#include "UpdaterBoxNPT.h"

#include <boost/python.hpp>
using namespace boost::python;

using namespace std;

/*! \file UpdaterBoxNPT.cc
    \brief Definition of UpdaterBoxNPT
*/

namespace hpmc
{

UpdaterBoxNPT::UpdaterBoxNPT(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<IntegratorHPMC> mc,
                             boost::shared_ptr<Variant> P,
                             Scalar dLx,
                             Scalar dLy,
                             Scalar dLz,
                             Scalar dxy,
                             Scalar dxz,
                             Scalar dyz,
                             Scalar move_ratio,
                             Scalar reduce,
                             bool isotropic,
                             const unsigned int seed)
        : Updater(sysdef),
          m_mc(mc),
          m_P(P),
          m_dLx(dLx),
          m_dLy(dLy),
          m_dLz(dLz),
          m_dxy(dxy),
          m_dxz(dxz),
          m_dyz(dyz),
          m_move_ratio(move_ratio),
          m_reduce(reduce),
          m_isotropic(isotropic),
          m_seed(seed)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterBoxNPT" << endl;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    // initialize logger and stats
    resetStats();

    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    m_maxparticlenumberchange_connection = m_pdata->connectMaxParticleNumberChange(boost::bind(&UpdaterBoxNPT::slotMaxNChange, this));

    if (isotropic)
        {
        computeAspectRatios();
        BoxDim curBox = m_pdata->getGlobalBox();
        m_dV = dLx * curBox.getLatticeVector(1).y * curBox.getLatticeVector(2).z;
        }
    }

UpdaterBoxNPT::~UpdaterBoxNPT()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterBoxNPT" << endl;
    m_maxparticlenumberchange_connection.disconnect();
    }

/*! hpmc::UpdaterBoxNPT provides:
    - hpmc_npt_trial_delta (Number of NPT box changes attempted during logger interval)
    - hpmc_npt_volume_acceptance (Ratio of volume change trials accepted during logger interval)
    - hpmc_npt_shear_acceptance (Ratio of shear trials accepted during logger interval)
    - hpmc_npt_move_ratio (Ratio of box length trials to total of box length and shear trials over logging period)
    - hpmc_npt_Lx (Current maximum trial length change of the first box vector)
    - hpmc_npt_Ly (Current maximum trial change of the y-component of the second box vector)
    - hpmc_npt_Lz (Current maximum trial change of the z-component of the third box vector)
    - hpmc_npt_xy (Current maximum trial change of the shear parameter for the second box vector)
    - hpmc_npt_xz (Current maximum trial change of the shear parameter for the third box vector in the x direction)
    - hpmc_npt_yz (Current maximum trial change of the shear parameter for the third box vector in the y direction)
    - hpmc_npt_pressure (Current value of beta*p parameter for the NpT updater)

    \returns a list of provided quantities
*/
std::vector< std::string > UpdaterBoxNPT::getProvidedLogQuantities()
    {
    // start with the updater provided quantities
    std::vector< std::string > result = Updater::getProvidedLogQuantities();

    // then add ours
    result.push_back("hpmc_npt_trial_count");
    result.push_back("hpmc_npt_volume_acceptance");
    result.push_back("hpmc_npt_shear_acceptance");
    result.push_back("hpmc_npt_move_ratio");
    result.push_back("hpmc_npt_dLx");
    result.push_back("hpmc_npt_dLz");
    result.push_back("hpmc_npt_dLy");
    result.push_back("hpmc_npt_dxy");
    result.push_back("hpmc_npt_dxz");
    result.push_back("hpmc_npt_dyz");
    result.push_back("hpmc_npt_pressure");
    return result;
    }

/*! Get logged quantity

    \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \returns the requested log quantity.
*/
Scalar UpdaterBoxNPT::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    hpmc_npt_counters_t counters = getCounters(1);

    // return requested log value
    if (quantity == "hpmc_npt_trial_count")
        {
        return counters.getNMoves();
        }
    else if (quantity == "hpmc_npt_volume_acceptance")
        {
        if (counters.volume_reject_count + counters.volume_accept_count == 0)
            return 0;
        else
            return counters.getVolumeAcceptance();
        }
    else if (quantity == "hpmc_npt_shear_acceptance")
        {
        if (counters.shear_reject_count + counters.shear_accept_count == 0)
            return 0;
        else
            return counters.getShearAcceptance();
        }
    else if (quantity == "hpmc_npt_move_ratio")
        {
        uint64_t total_volume = counters.volume_accept_count + counters.volume_reject_count;
        uint64_t total_shear = counters.shear_accept_count + counters.shear_reject_count;
        return (total_volume) / (total_volume + total_shear);
        }
    else if (quantity == "hpmc_npt_dLx")
        {
        return m_dLx;
        }
    else if (quantity == "hpmc_npt_dLy")
        {
        return m_dLy;
        }
    else if (quantity == "hpmc_npt_dLz")
        {
        return m_dLz;
        }
    else if (quantity == "hpmc_npt_dxy")
        {
        return m_dxy;
        }
    else if (quantity == "hpmc_npt_dxz")
        {
        return m_dxz;
        }
    else if (quantity == "hpmc_npt_dyz")
        {
        return m_dyz;
        }
    else if (quantity == "hpmc_npt_pressure")
        {
        return m_P->getValue(timestep);
        }
    else
        {
        return Updater::getLogValue(quantity, timestep);
        }
    }

/*! Determine if box exceeds a shearing threshold and needs to be lattice reduced.

    The maximum amount of shear to allow is somewhat arbitrary, but must be > 0.5. Small values mean the box is
    reconstructed more often, making it more confusing to track particle diffusion. Larger shear values mean
    parallel box planes can get closer together, reducing the number of cells possible in the cell list or increasing
    the number of images that must be checked for small boxes.

    Box is oversheared in direction \f$ \hat{e}_i \f$ if
    \f$ \bar{e}_j \cdot \hat{e}_i >= reduce * \left| \bar{e}_i \right| \f$
    or
    \f$ \bar{e}_j \cdot \bar{e}_i >= reduce * \left| \bar{e}_i \right| ^2 \f$
    \f$ = reduce * \bar{e}_i \cdot \bar{e}_i \f$

    \returns bool true if box is overly sheared
*/
inline bool UpdaterBoxNPT::is_oversheared()
    {
#ifdef ENABLE_MPI
    if (m_comm) return false; // lattice reduction not yet supported in MPI
#endif
    if (m_reduce <= 0.5) return false;

    const BoxDim curBox = m_pdata->getGlobalBox();
    const Scalar3 x = curBox.getLatticeVector(0);
    const Scalar3 y = curBox.getLatticeVector(1);
    const Scalar3 z = curBox.getLatticeVector(2);

    const Scalar y_x = y.x; // x component of y vector
    const Scalar max_y_x = x.x * m_reduce;
    const Scalar z_x = z.x; // x component of z vector
    const Scalar max_z_x = x.x * m_reduce;
    // z_y \left| y \right|
    const Scalar z_yy = dot(z,y);
    // MAX_SHEAR * left| y \right| ^2
    const Scalar max_z_y_2 = dot(y,y) * m_reduce;

    if (fabs(y_x) > max_y_x || fabs(z_x) > max_z_x || fabs(z_yy) > max_z_y_2)
        return true;
    else
        return false;
    }

/*! Perform lattice reduction.
    Remove excessive box shearing by finding a more cubic degenerate lattice
    when shearing is more half a lattice vector from cubic. The lattice reduction could make data needlessly complicated
    and may break detailed balance, use judiciously.

    \returns true if overshear was removed
*/
inline bool UpdaterBoxNPT::remove_overshear()
    {
    bool overshear = false; // initialize return value
    const Scalar NPT_MAX_SHEAR = Scalar(0.5f); // lattice can be reduced if shearing exceeds this value

    BoxDim newBox = m_pdata->getGlobalBox();
    Scalar3 x = newBox.getLatticeVector(0);
    Scalar3 y = newBox.getLatticeVector(1);
    Scalar3 z = newBox.getLatticeVector(2);
    Scalar xy = newBox.getTiltFactorXY();
    Scalar xz = newBox.getTiltFactorXZ();
    Scalar yz = newBox.getTiltFactorYZ();

    // Remove one lattice vector of shear if necessary. Only apply once so image doesn't change more than one.

    const Scalar y_x = y.x; // x component of y vector
    const Scalar max_y_x = x.x * NPT_MAX_SHEAR;
    if (y_x > max_y_x)
        {
        // Ly * xy_new = Ly * xy_old + sign*Lx --> xy_new = xy_old + sign*Lx/Ly
        xy -= x.x / y.y;
        y.x = xy * y.y;
        overshear = true;
        }
    if (y_x < -max_y_x)
        {
        xy += x.x / y.y;
        y.x = xy * y.y;
        overshear = true;
        }

    const Scalar z_x = z.x; // x component of z vector
    const Scalar max_z_x = x.x * NPT_MAX_SHEAR;
    if (z_x > max_z_x)
        {
        // Lz * xz_new = Lz * xz_old + sign*Lx --> xz_new = xz_old + sign*Lx/Lz
        xz -= x.x / z.z;
        z.x = xz * z.z;
        overshear = true;
        }
    if (z_x < -max_z_x)
        {
        // Lz * xz_new = Lz * xz_old + sign*Lx --> xz_new = xz_old + sign*Lx/Lz
        xz += x.x / z.z;
        z.x = xz * z.z;
        overshear = true;
        }

    // z_y \left| y \right|
    const Scalar z_yy = dot(z,y);
    // MAX_SHEAR * left| y \right| ^2
    const Scalar max_z_y_2 = dot(y,y) * NPT_MAX_SHEAR;
    if (z_yy > max_z_y_2)
        {
        // Lz * xz_new = Lz * xz_old + sign * y.x --> xz_new = = xz_old + sign * y.x / Lz
        xz -= y.x / z.z;
        // Lz * yz_new = Lz * yz_old + sign * y.y --> yz_new = yz_old + sign y.y /Lz
        yz -= y.y / z.z;
        overshear = true;
        }
    if (z_yy < -max_z_y_2)
        {
        // Lz * xz_new = Lz * xz_old + sign * y.x --> xz_new = = xz_old + sign * y.x / Lz
        xz += y.x / z.z;
        // Lz * yz_new = Lz * yz_old + sign * y.y --> yz_new = yz_old + sign y.y /Lz
        yz += y.y / z.z;
        overshear = true;
        }

    if (overshear)
        {
        newBox.setTiltFactors(xy, xz, yz);
        m_pdata->setGlobalBox(newBox);

        // Use lexical scope to make sure ArrayHandles get cleaned up
            {
            // Get particle positions and images
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            unsigned int N = m_pdata->getN();

            // move the particles to be inside the new box
            for (unsigned int i = 0; i < N; i++)
                {
                Scalar4 pos = h_pos.data[i];
                int3 image = h_image.data[i];
                newBox.wrap(pos, image);
                h_pos.data[i] = pos;
                h_image.data[i] = image;
                }
            } // end lexical scope

        // To get particles into the right domain in MPI, we will store and then reload a snapshot
        SnapshotParticleData<Scalar> snap;
        m_pdata->takeSnapshot(snap);

        // loading from snapshot will load particles into the proper MPI domain
        m_pdata->initializeFromSnapshot(snap);

        // we've moved the particles, communicate those changes
        m_mc->communicate(true);
        }
    return overshear;
    }

//! Try new box with particle positions scaled from previous box.
/*! If new box generates overlaps, restore original box and particle positions.
    \param Lx new Lx value
    \param Ly new Ly value
    \param Lz new Lz value
    \param xy new xy value
    \param xz new xz value
    \param yz new yz value
    \param timestep current simulation step

    \returns bool True if box resize was accepted

    If box is excessively sheared, subtract lattice vectors to make box more cubic.
*/
inline bool UpdaterBoxNPT::box_resize_trial(Scalar Lx,
                                            Scalar Ly,
                                            Scalar Lz,
                                            Scalar xy,
                                            Scalar xz,
                                            Scalar yz,
                                            unsigned int timestep,
                                            Scalar boltzmann,
                                            Saru& rng
                                            )
    {
    // Make a backup copy of position data
    unsigned int N_backup = m_pdata->getN();
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::readwrite);
        memcpy(h_pos_backup.data, h_pos.data, sizeof(Scalar4) * N_backup);
        }

    BoxDim curBox = m_pdata->getGlobalBox();

    // Attempt box resize and check for overlaps
    BoxDim newBox = m_pdata->getGlobalBox();

    newBox.setL(make_scalar3(Lx, Ly, Lz));
    newBox.setTiltFactors(xy, xz, yz);

    bool allowed = m_mc->attemptBoxResize(timestep, newBox);
    if (m_mc->getExternalField())
        {
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::readwrite);
        Scalar ext_boltzmann = m_mc->getExternalField()->calculateBoltzmannFactor(h_pos_backup.data, NULL, &curBox);
        boltzmann *= ext_boltzmann;
        }

    double p = rng.d();

    if (allowed && p < boltzmann)
        {
        return true;
        }
    else
        {
        // Restore original box and particle positions
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::read);
            unsigned int N = m_pdata->getN();
            if (N != N_backup)
                {
                this->m_exec_conf->msg->error() << "update.npt" << ": Number of particles mismatch when rejecting box resize" << std::endl;
                throw std::runtime_error("Error resizing box");
                // note, this error should never appear (because particles are not migrated after a box resize),
                // but is left here as a sanity check
                }
            memcpy(h_pos.data, h_pos_backup.data, sizeof(Scalar4) * N);
            }

        m_pdata->setGlobalBox(curBox);

        // we have moved particles, communicate those changes
        m_mc->communicate(false);
        return false;
        }
    }

/*! Perform Metropolis Monte Carlo box resizes and shearing
    \param timestep Current time step of the simulation
*/
void UpdaterBoxNPT::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("UpdaterBoxNPT");
    m_count_step_start = m_count_total;
    m_exec_conf->msg->notice(10) << "UpdaterBoxNPT: " << timestep << endl;

    // Get updater parameters for current timestep
    Scalar P = m_P->getValue(timestep);
    Scalar dLx_max = m_dLx;
    Scalar dLy_max = m_dLy;
    Scalar dLz_max = m_dLz;
    Scalar dV_max = m_dV;
    Scalar dxy_max = m_dxy;
    Scalar dxz_max = m_dxz;
    Scalar dyz_max = m_dyz;
    unsigned int move_ratio = m_move_ratio * 65536;

    // Get current particle data and box lattice parameters
    assert(m_pdata);
    unsigned int Ndim = m_sysdef->getNDimensions();
    unsigned int Nglobal = m_pdata->getNGlobal();
    BoxDim curBox = m_pdata->getGlobalBox();
    Scalar curL[3];
    Scalar newL[3]; // Lx, Ly, Lz
    newL[0] = curL[0] = curBox.getLatticeVector(0).x;
    newL[1] = curL[1] = curBox.getLatticeVector(1).y;
    newL[2] = curL[2] = curBox.getLatticeVector(2).z;
    Scalar newShear[3]; // xy, xz, yz
    newShear[0] = curBox.getTiltFactorXY();
    newShear[1] = curBox.getTiltFactorXZ();
    newShear[2] = curBox.getTiltFactorYZ();

    m_exec_conf->msg->notice(8) << "UpdaterBoxNPT read current box:"
                                << " Lx: " << curL[0]
                                << " Ly: " << curL[1]
                                << " Lz: " << curL[2]
                                << " XY: " << newShear[0]
                                << " XZ: " << newShear[1]
                                << " YZ: " << newShear[2] << endl;

    // Choose a move type
    Saru rng(m_seed, timestep, 0xf6a510ab);
    unsigned int move_type_select = rng.u32() & 0xffff;

    // Attempt and evaluate a move
    if (move_type_select < move_ratio)
        {
        // Volume change
        m_exec_conf->msg->notice(6) << "UpdaterBoxNPT attempting trial volume move" << endl;
        double Vold, dV, Vnew;
        Vold = double(curL[0]) * double(curL[1]);
        if (Ndim == 3) Vold *= curL[2];

        if (m_isotropic)
            {
            m_exec_conf->msg->notice(7) << "UpdaterBoxNPT: isotropic mode" << endl;
            dV = rng.s(-dV_max, dV_max);
            double L_scale = pow((Vold + dV)/Vold, 1./3.);
            // perform isotropic volume change
            newL[0] *= L_scale;
            // To avoid drift in aspect ratio, always set Ly and Lz as ratio of Lx
            newL[1] = m_rLy * newL[0];
            if (Ndim == 3) newL[2] = m_rLz * newL[0];
           }
        else
            {
            m_exec_conf->msg->notice(7) << "UpdaterBoxNPT: non-isotropic mode" << endl;
            // Choose a lattice vector if non-isotropic volume changes
            unsigned int i = rand_select(rng, Ndim - 1);
            Scalar dL_max(dLx_max);
            // Don't bother checking i==0: dL_max was already initialized to dLx_max
            if (i==1) dL_max = dLy_max;
            if (i==2) dL_max = dLz_max;
            // Choose a length change
            Scalar dL = rng.s(-dL_max, dL_max);
            // perform volume change by applying a delta to one dimension
            newL[i] += dL;
            }
        m_exec_conf->msg->notice(8) << "UpdaterBoxNPT new box proposed:"
                                << " Lx: " << newL[0]
                                << " Ly: " << newL[1]
                                << " Lz: " << newL[2] << endl;
        bool safe_box = true;
        Scalar min_allowed_size = m_mc->getMaxTransMoveSize();
        for (unsigned int j = 0; j < Ndim; j++)
            {
            if ((newL[j]) < min_allowed_size)
                {
                // volume must be kept positive
                m_exec_conf->msg->notice(5) << "Box resize rejected because dimension " << j << " would be < translation distance." << endl;
                m_count_total.volume_reject_count++;
                safe_box = false;
                }
            }

        if (safe_box)
            {
            // Calculate volume change for 2 or 3 dimensions.
            Vnew = double(newL[0]) * double(newL[1]);
            if (Ndim ==3) Vnew *= newL[2];

            if (m_mc->getExternalField() && m_mc->getExternalField()->hasVolume())
                {
                double ext_scale = Vnew/Vold;
                Vold = m_mc->getExternalField()->getVolume();
                Vnew = Vold*ext_scale;
                }

            dV = Vnew - Vold;
            m_exec_conf->msg->notice(10) << "UpdaterBoxNPT: dV = Vnew - Vold = " << " = " << Vnew << " - " << Vold << " = " << dV << endl;

            // Calculate Boltzmann factor
            // For precision reasons, it is probably best to gather all terms in the psuedo-Halmitonian
            // before applying a single exponential rather than to multiply multiple exponentials.
            double dBetaH = -P * dV + Nglobal * log(Vnew/Vold);
            m_exec_conf->msg->notice(9) << "UpdaterBoxNPT evaluating change in pseudo-Hamiltonian " << dBetaH << endl;
            double Boltzmann = exp(dBetaH);
            m_exec_conf->msg->notice(9) << "Pseudo-Boltzmann factor " << Boltzmann << endl;

            bool accept = false;
            // attempt box change
            accept = box_resize_trial(newL[0],
                                      newL[1],
                                      newL[2],
                                      newShear[0],
                                      newShear[1],
                                      newShear[2],
                                      timestep,
                                      Boltzmann,
                                      rng);

            if (accept)
                {
                m_exec_conf->msg->notice(5) << "UpdaterBoxNPT: accepting box change" << endl;
                m_count_total.volume_accept_count++;
                }
            else
                {
                m_exec_conf->msg->notice(5) << "UpdaterBoxNPT: rejecting box change" << endl;
                m_count_total.volume_reject_count++;
                }
            }
        }
    else
        {
        // Shearing change
        m_exec_conf->msg->notice(6) << "UpdaterBoxNPT attempting trial shearing move" << endl;

        Scalar dA, dA_max(0);
        // Choose a tilt factor and randomly perturb it
        unsigned int i(0);
        if (Ndim == 3)
            {
            i = rand_select(rng, 2);
            }
        if (i == 0) dA_max = dxy_max;
        if (i == 1) dA_max = dxz_max;
        if (i == 2) dA_max = dyz_max;
        dA = rng.s(-dA_max, dA_max);
        newShear[i] += dA;
        m_exec_conf->msg->notice(8) << "UpdaterBoxNPT new box proposed:"
                                << " XY: " << newShear[0]
                                << " XZ: " << newShear[1]
                                << " YZ: " << newShear[2] << endl;

        // To do: check if we've sheared the box too far in a direction and shift to a more cubic degenerate lattice
        // while (xy * Ly > Lx) subtract Lx from xy*Ly, etc.

        Scalar boltzmann = Scalar(1);
        // Attempt box resize
        bool trial_success = box_resize_trial(newL[0],
                                              newL[1],
                                              newL[2],
                                              newShear[0],
                                              newShear[1],
                                              newShear[2],
                                              timestep,
                                              boltzmann,
                                              rng
                                              );
        if (trial_success)
            {
            m_exec_conf->msg->notice(5) << "UpdaterBoxNPT: accepting box change" << endl;
            m_count_total.shear_accept_count++;
            }
        else
            {
            m_exec_conf->msg->notice(5) << "UpdaterBoxNPT: rejecting box change" << endl;
            m_count_total.shear_reject_count++;
            }
        }


    if (m_prof) m_prof->push("UpdaterBoxNPT: examining shear");
    if (is_oversheared())
        {
        while (remove_overshear()) {}; // lattice reduction, possibly in several steps
        m_exec_conf->msg->notice(5) << "Lattice reduction performed at step " << timestep << endl;
        }
    if (m_prof) m_prof->pop();

    if (m_prof) m_prof->pop();
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    UpdaterBoxNPT maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
hpmc_npt_counters_t UpdaterBoxNPT::getCounters(unsigned int mode)
    {
    hpmc_npt_counters_t result;

    if (mode == 0)
        result = m_count_total;
    else if (mode == 1)
        result = m_count_total - m_count_run_start;
    else
        result = m_count_total - m_count_step_start;

    // don't MPI_AllReduce counters because all ranks count the same thing
    return result;
    }

void export_UpdaterBoxNPT()
    {
    class_< UpdaterBoxNPT, boost::shared_ptr< UpdaterBoxNPT >, bases<Updater>, boost::noncopyable>
    ("UpdaterBoxNPT", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<IntegratorHPMC>,
                         boost::shared_ptr<Variant>,
                         Scalar,
                         Scalar,
                         Scalar,
                         Scalar,
                         Scalar,
                         Scalar,
                         Scalar,
                         Scalar,
                         bool,
                         const unsigned int >())
    .def("setParams", &UpdaterBoxNPT::setParams)
    .def("printStats", &UpdaterBoxNPT::printStats)
    .def("resetStats", &UpdaterBoxNPT::resetStats)
    .def("getP", &UpdaterBoxNPT::getP)
    .def("getdLx", &UpdaterBoxNPT::getdLx)
    .def("getdLy", &UpdaterBoxNPT::getdLy)
    .def("getdLz", &UpdaterBoxNPT::getdLz)
    .def("getdxy", &UpdaterBoxNPT::getdxy)
    .def("getdxz", &UpdaterBoxNPT::getdxz)
    .def("getdyz", &UpdaterBoxNPT::getdyz)
    .def("getMoveRatio", &UpdaterBoxNPT::getMoveRatio)
    .def("getReduce", &UpdaterBoxNPT::getReduce)
    .def("getIsotropic", &UpdaterBoxNPT::getIsotropic)
    .def("computeAspectRatios", &UpdaterBoxNPT::computeAspectRatios)
    .def("getCounters", &UpdaterBoxNPT::getCounters)
    ;

    class_< hpmc_npt_counters_t >("hpmc_npt_counters_t")
    .def_readwrite("volume_accept_count", &hpmc_npt_counters_t::volume_accept_count)
    .def_readwrite("volume_reject_count", &hpmc_npt_counters_t::volume_reject_count)
    .def_readwrite("shear_accept_count", &hpmc_npt_counters_t::shear_accept_count)
    .def_readwrite("shear_reject_count", &hpmc_npt_counters_t::shear_reject_count)
    .def("getVolumeAcceptance", &hpmc_npt_counters_t::getVolumeAcceptance)
    .def("getShearAcceptance", &hpmc_npt_counters_t::getShearAcceptance)
    .def("getNMoves", &hpmc_npt_counters_t::getNMoves)
    ;
    }

} // end namespace hpmc
