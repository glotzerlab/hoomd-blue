// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "UpdaterBoxMC.h"
#include "hoomd/RNGIdentifiers.h"

namespace py = pybind11;

/*! \file UpdaterBoxMC.cc
    \brief Definition of UpdaterBoxMC
*/

namespace hpmc
{

UpdaterBoxMC::UpdaterBoxMC(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMC> mc,
                             std::shared_ptr<Variant> P,
                             const Scalar frequency,
                             const unsigned int seed)
        : Updater(sysdef),
          m_mc(mc),
          m_P(P),
          m_frequency(frequency),
          m_Volume_delta(0.0),
          m_Volume_weight(0.0),
          m_lnVolume_delta(0.0),
          m_lnVolume_weight(0.0),
          m_Volume_A1(0.0),
          m_Volume_A2(0.0),
          m_Length_delta {0.0, 0.0, 0.0},
          m_Length_weight(0.0),
          m_Shear_delta {0.0, 0.0, 0.0},
          m_Shear_weight(0.0),
          m_Shear_reduce(0.0),
          m_Aspect_delta(0.0),
          m_Aspect_weight(0.0),
          m_seed(seed)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterBoxMC" << std::endl;

    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif

    // initialize logger and stats
    resetStats();

    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal().connect<UpdaterBoxMC, &UpdaterBoxMC::slotMaxNChange>(this);

    }

UpdaterBoxMC::~UpdaterBoxMC()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterBoxMC" << std::endl;
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<UpdaterBoxMC, &UpdaterBoxMC::slotMaxNChange>(this);
    }

/*! hpmc::UpdaterBoxMC provides:
    - hpmc_boxmc_trial_delta (Number of MC box changes attempted during logger interval)
    - hpmc_boxmc_volume_acceptance (Ratio of volume change trials accepted during logger interval)
    - hpmc_boxmc_shear_acceptance (Ratio of shear trials accepted during logger interval)
    - hpmc_boxmc_aspect_acceptance (Ratio of aspect trials accepted during logger interval)
    - hpmc_boxmc_betaP (Current value of beta*p parameter for the box updater)

    \returns a list of provided quantities
*/
std::vector< std::string > UpdaterBoxMC::getProvidedLogQuantities()
    {
    // start with the updater provided quantities
    std::vector< std::string > result = Updater::getProvidedLogQuantities();

    // then add ours
    result.push_back("hpmc_boxmc_trial_count");
    result.push_back("hpmc_boxmc_volume_acceptance");
    result.push_back("hpmc_boxmc_ln_volume_acceptance");
    result.push_back("hpmc_boxmc_shear_acceptance");
    result.push_back("hpmc_boxmc_aspect_acceptance");
    result.push_back("hpmc_boxmc_betaP");
    return result;
    }

/*! Get logged quantity

    \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \returns the requested log quantity.
*/
Scalar UpdaterBoxMC::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    hpmc_boxmc_counters_t counters = getCounters(1);

    // return requested log value
    if (quantity == "hpmc_boxmc_trial_count")
        {
        return counters.getNMoves();
        }
    else if (quantity == "hpmc_boxmc_volume_acceptance")
        {
        if (counters.volume_reject_count + counters.volume_accept_count == 0)
            return 0;
        else
            return counters.getVolumeAcceptance();
        }
    else if (quantity == "hpmc_boxmc_ln_volume_acceptance")
        {
        if (counters.ln_volume_reject_count + counters.ln_volume_accept_count == 0)
            return 0;
        else
            return counters.getLogVolumeAcceptance();
        }
    else if (quantity == "hpmc_boxmc_shear_acceptance")
        {
        if (counters.shear_reject_count + counters.shear_accept_count == 0)
            return 0;
        else
            return counters.getShearAcceptance();
        }
    else if (quantity == "hpmc_boxmc_aspect_acceptance")
        {
        if (counters.aspect_reject_count + counters.aspect_accept_count == 0)
            return 0;
        else
            return counters.getAspectAcceptance();
        }
    else if (quantity == "hpmc_boxmc_betaP")
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
inline bool UpdaterBoxMC::is_oversheared()
    {
    if (m_Shear_reduce <= 0.5) return false;

    const BoxDim curBox = m_pdata->getGlobalBox();
    const Scalar3 x = curBox.getLatticeVector(0);
    const Scalar3 y = curBox.getLatticeVector(1);
    const Scalar3 z = curBox.getLatticeVector(2);

    const Scalar y_x = y.x; // x component of y vector
    const Scalar max_y_x = x.x * m_Shear_reduce;
    const Scalar z_x = z.x; // x component of z vector
    const Scalar max_z_x = x.x * m_Shear_reduce;
    // z_y \left| y \right|
    const Scalar z_yy = dot(z,y);
    // MAX_SHEAR * left| y \right| ^2
    const Scalar max_z_y_2 = dot(y,y) * m_Shear_reduce;

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
inline bool UpdaterBoxMC::remove_overshear()
    {
    bool overshear = false; // initialize return value
    const Scalar MAX_SHEAR = Scalar(0.5f); // lattice can be reduced if shearing exceeds this value

    BoxDim newBox = m_pdata->getGlobalBox();
    Scalar3 x = newBox.getLatticeVector(0);
    Scalar3 y = newBox.getLatticeVector(1);
    Scalar3 z = newBox.getLatticeVector(2);
    Scalar xy = newBox.getTiltFactorXY();
    Scalar xz = newBox.getTiltFactorXZ();
    Scalar yz = newBox.getTiltFactorYZ();

    // Remove one lattice vector of shear if necessary. Only apply once so image doesn't change more than one.

    const Scalar y_x = y.x; // x component of y vector
    const Scalar max_y_x = x.x * MAX_SHEAR;
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
    const Scalar max_z_x = x.x * MAX_SHEAR;
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
    const Scalar max_z_y_2 = dot(y,y) * MAX_SHEAR;
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
inline bool UpdaterBoxMC::box_resize_trial(Scalar Lx,
                                          Scalar Ly,
                                          Scalar Lz,
                                          Scalar xy,
                                          Scalar xz,
                                          Scalar yz,
                                          unsigned int timestep,
                                          Scalar deltaE,
                                          hoomd::RandomGenerator& rng
                                          )
    {
    // Make a backup copy of position data
    unsigned int N_backup = m_pdata->getN();
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::overwrite);
        memcpy(h_pos_backup.data, h_pos.data, sizeof(Scalar4) * N_backup);
        }

    BoxDim curBox = m_pdata->getGlobalBox();

    if (m_mc->getPatchInteraction())
        {
        // energy of old configuration
        deltaE -= m_mc->computePatchEnergy(timestep);
        }

    // Attempt box resize and check for overlaps
    BoxDim newBox = m_pdata->getGlobalBox();

    newBox.setL(make_scalar3(Lx, Ly, Lz));
    newBox.setTiltFactors(xy, xz, yz);

    bool allowed = m_mc->attemptBoxResize(timestep, newBox);

    if (allowed && m_mc->getPatchInteraction())
        {
        deltaE += m_mc->computePatchEnergy(timestep);
        }

    if (allowed && m_mc->getExternalField())
        {
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::readwrite);
        Scalar ext_energy = m_mc->getExternalField()->calculateDeltaE(h_pos_backup.data, NULL, &curBox);
        // The exponential is a very fast function and we may do better to add pseudo-Hamiltonians and exponentiate only once...
        deltaE += ext_energy;
        }

    double p = hoomd::detail::generate_canonical<double>(rng);

    if (allowed && p < fast::exp(-deltaE))
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
                this->m_exec_conf->msg->error() << "update.boxmc" << ": Number of particles mismatch when rejecting box resize" << std::endl;
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

inline bool UpdaterBoxMC::safe_box(const Scalar newL[3], const unsigned int& Ndim)
    {
    //Scalar min_allowed_size = m_mc->getMaxTransMoveSize(); // This is dealt with elsewhere
    const Scalar min_allowed_size(0.0); // volume must be kept positive
    for (unsigned int j = 0; j < Ndim; j++)
        {
        if ((newL[j]) < min_allowed_size)
            {
            // volume must be kept positive
            m_exec_conf->msg->notice(10) << "Box unsafe because dimension " << j << " would be negative." << std::endl;
            return false;
            }
        }
    return true;
    }

/*! Perform Metropolis Monte Carlo box resizes and shearing
    \param timestep Current time step of the simulation
*/
void UpdaterBoxMC::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("UpdaterBoxMC");
    m_count_step_start = m_count_total;
    m_exec_conf->msg->notice(10) << "UpdaterBoxMC: " << timestep << std::endl;

    // Create a prng instance for this timestep
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterBoxMC, m_seed, timestep);

    // Choose a move type
    // This seems messy and can hopefully be simplified and generalized.
    // This line will need to be rewritten or updated when move types are added to the updater.
    float range = m_Volume_weight + m_lnVolume_weight + m_Length_weight + m_Shear_weight + m_Aspect_weight;
    if (range == 0.0)
        {
        // Attempt to execute with no move types set.
        m_exec_conf->msg->warning() << "No move types with non-zero weight. UpdaterBoxMC has nothing to do." << std::endl;
        if (m_prof) m_prof->pop();
        return;
        }
    float move_type_select = hoomd::detail::generate_canonical<float>(rng) * range; // generate a number on (0, range]

    // Attempt and evaluate a move
    // This section will need to be updated when move types are added.
    if (move_type_select < m_Volume_weight)
        {
        // Isotropic volume change
        m_exec_conf->msg->notice(8) << "Volume move performed at step " << timestep << std::endl;
        update_V(timestep, rng);
        }
    else if (move_type_select < m_Volume_weight + m_lnVolume_weight)
        {
        // Isotropic volume change in logarithmic steps
        m_exec_conf->msg->notice(8) << "lnV move performed at step " << timestep << std::endl;
        update_lnV(timestep, rng);
        }
    else if (move_type_select < m_Volume_weight + m_lnVolume_weight + m_Length_weight)
        {
        // Volume change in distribution of box lengths
        m_exec_conf->msg->notice(8) << "Box length move performed at step " << timestep << std::endl;
        update_L(timestep, rng);
        }
    else if (move_type_select < m_Volume_weight + m_lnVolume_weight + m_Length_weight + m_Shear_weight)
        {
        // Shear change
        m_exec_conf->msg->notice(8) << "Box shear move performed at step " << timestep << std::endl;
        update_shear(timestep, rng);
        }
    else if (move_type_select <= m_Volume_weight + m_lnVolume_weight + m_Length_weight + m_Shear_weight + m_Aspect_weight)
        {
        // Volume conserving aspect change
        m_exec_conf->msg->notice(8) << "Box aspect move performed at step " << timestep << std::endl;
        update_aspect(timestep, rng);
        }
    else
        {
        // Should not reach this point
        m_exec_conf->msg->warning() << "UpdaterBoxMC selected an unassigned move type. Selected " << move_type_select << " from range " << range << std::endl;
        if (m_prof) m_prof->pop();
        return;
        }

    if (m_prof) m_prof->push("UpdaterBoxMC: examining shear");
    if (is_oversheared())
        {
        while (remove_overshear()) {}; // lattice reduction, possibly in several steps
        m_exec_conf->msg->notice(5) << "Lattice reduction performed at step " << timestep << std::endl;
        }
    if (m_prof) m_prof->pop();

    if (m_prof) m_prof->pop();
    }

void UpdaterBoxMC::update_L(unsigned int timestep, hoomd::RandomGenerator& rng)
    {
    if (m_prof) m_prof->push("UpdaterBoxMC: update_L");
    // Get updater parameters for current timestep
    Scalar P = m_P->getValue(timestep);

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

    // Volume change

    // Choose a lattice vector if non-isotropic volume changes
    unsigned int nonzero_dim = 0;
    for (unsigned int i = 0; i < Ndim; ++i)
        if (m_Length_delta[i] != 0.0)
            nonzero_dim++;

    unsigned int i = hoomd::UniformIntDistribution(nonzero_dim-1)(rng);
    for (unsigned int j = 0; j < Ndim; ++j)
        if (m_Length_delta[j] == 0.0 && i == j)
            ++i;

    if (i == Ndim)
        {
        // all dimensions have delta==0, just count as accepted and return
        m_count_total.volume_accept_count++;
        return;
        }

    Scalar dL_max(m_Length_delta[i]);

    // Choose a length change
    Scalar dL = hoomd::UniformDistribution<Scalar>(-dL_max, dL_max)(rng);
    // perform volume change by applying a delta to one dimension
    newL[i] += dL;

    if (!safe_box(newL, Ndim))
        {
        m_count_total.volume_reject_count++;
        }
    else
        {
        // Calculate volume change for 2 or 3 dimensions.
        double Vold, dV, Vnew;
        Vold = curL[0] * curL[1];
        if (Ndim == 3) Vold *= curL[2];
        Vnew = newL[0] * newL[1];
        if (Ndim ==3) Vnew *= newL[2];
        dV = Vnew - Vold;

        // Calculate Boltzmann factor
        double dBetaH = P * dV - Nglobal * log(Vnew/Vold);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                  newL[1],
                                  newL[2],
                                  newShear[0],
                                  newShear[1],
                                  newShear[2],
                                  timestep,
                                  dBetaH,
                                  rng
                                  );

        if (accept)
            {
            m_count_total.volume_accept_count++;
            }
        else
            {
            m_count_total.volume_reject_count++;
            }
        }
    if (m_prof) m_prof->pop();
    }

//! Update the box volume in logarithmic steps
void UpdaterBoxMC::update_lnV(unsigned int timestep, hoomd::RandomGenerator& rng)
    {
    if (m_prof) m_prof->push("UpdaterBoxMC: update_lnV");
    // Get updater parameters for current timestep
    Scalar P = m_P->getValue(timestep);

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

    // original volume
    double V = curL[0] * curL[1];
    if (Ndim == 3)
        {
        V *= curL[2];
        }
    // Aspect ratios
    Scalar A1 = m_Volume_A1;
    Scalar A2 = m_Volume_A2;

    // Volume change
    Scalar dlnV_max(m_lnVolume_delta);

    // Choose a volume change
    Scalar dlnV = hoomd::UniformDistribution<Scalar>(-dlnV_max, dlnV_max)(rng);
    Scalar new_V = V*exp(dlnV);

    // perform isotropic volume change
    if (Ndim == 3)
        {
        newL[0] = pow(A1 * A2 * new_V,(1./3.));
        newL[1] = newL[0]/A1;
        newL[2] = newL[0]/A2;
        }
    else // Ndim ==2
        {
        newL[0] = pow(A1*new_V,(1./2.));
        newL[1] = newL[0]/A1;
        // newL[2] is already assigned to curL[2]
        }

    if (!safe_box(newL, Ndim))
        {
        m_count_total.ln_volume_reject_count++;
        }
    else
        {
        // Calculate Boltzmann factor
        double dBetaH = P * (new_V-V) - (Nglobal+1) * log(new_V/V);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                      newL[1],
                                      newL[2],
                                      newShear[0],
                                      newShear[1],
                                      newShear[2],
                                      timestep,
                                      dBetaH,
                                      rng);

        if (accept)
            {
            m_count_total.ln_volume_accept_count++;
            }
        else
            {
            m_count_total.ln_volume_reject_count++;
            }
        }
    if (m_prof) m_prof->pop();
    }

void UpdaterBoxMC::update_V(unsigned int timestep, hoomd::RandomGenerator& rng)
    {
    if (m_prof) m_prof->push("UpdaterBoxMC: update_V");
    // Get updater parameters for current timestep
    Scalar P = m_P->getValue(timestep);

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

    // original volume
    double V = curL[0] * curL[1];
    if (Ndim == 3)
        {
        V *= curL[2];
        }
    // Aspect ratios
    Scalar A1 = m_Volume_A1;
    Scalar A2 = m_Volume_A2;

    // Volume change
    Scalar dV_max(m_Volume_delta);

    // Choose a volume change
    Scalar dV = hoomd::UniformDistribution<Scalar>(-dV_max, dV_max)(rng);

    // perform isotropic volume change
    if (Ndim == 3)
        {
        newL[0] = pow((A1 * A2 * (V + dV)),(1./3.));
        newL[1] = newL[0]/A1;
        newL[2] = newL[0]/A2;
        }
    else // Ndim ==2
        {
        newL[0] = pow((A1*(V+dV)),(1./2.));
        newL[1] = newL[0]/A1;
        // newL[2] is already assigned to curL[2]
        }

    if (!safe_box(newL, Ndim))
        {
        m_count_total.volume_reject_count++;
        }
    else
        {
        // Calculate new volume
        double Vnew = newL[0] * newL[1];
        if (Ndim == 3)
            {
            Vnew *= newL[2];
            }
        // Calculate Boltzmann factor
        double dBetaH = P * dV - Nglobal * log(Vnew/V);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                      newL[1],
                                      newL[2],
                                      newShear[0],
                                      newShear[1],
                                      newShear[2],
                                      timestep,
                                      dBetaH,
                                      rng);

        if (accept)
            {
            m_count_total.volume_accept_count++;
            }
        else
            {
            m_count_total.volume_reject_count++;
            }
        }
    if (m_prof) m_prof->pop();
    }

void UpdaterBoxMC::update_shear(unsigned int timestep, hoomd::RandomGenerator& rng)
    {
    if (m_prof) m_prof->push("UpdaterBoxMC: update_shear");
    // Get updater parameters for current timestep
    // Get current particle data and box lattice parameters
    assert(m_pdata);
    unsigned int Ndim = m_sysdef->getNDimensions();
    //unsigned int Nglobal = m_pdata->getNGlobal();
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

    Scalar dA, dA_max;
    // Choose a tilt factor and randomly perturb it
    unsigned int i(0);
    if (Ndim == 3)
        {
        i = hoomd::UniformIntDistribution(2)(rng);
        }
    dA_max = m_Shear_delta[i];
    dA = hoomd::UniformDistribution<Scalar>(-dA_max, dA_max)(rng);
    newShear[i] += dA;

    // Attempt box resize
    bool trial_success = box_resize_trial(newL[0],
                                          newL[1],
                                          newL[2],
                                          newShear[0],
                                          newShear[1],
                                          newShear[2],
                                          timestep,
                                          Scalar(0.0),
                                          rng);
    if (trial_success)
        {
        m_count_total.shear_accept_count++;
        }
    else
        {
        m_count_total.shear_reject_count++;
        }
    if (m_prof) m_prof->pop();
    }

void UpdaterBoxMC::update_aspect(unsigned int timestep, hoomd::RandomGenerator& rng)
    {
    // We have not established what ensemble this samples:
    // This is not a thermodynamic updater.
    // There is also room for improvement in enforcing volume conservation.
    if (m_prof) m_prof->push("UpdaterBoxMC: update_aspect");
    // Get updater parameters for current timestep
    // Get current particle data and box lattice parameters
    assert(m_pdata);
    unsigned int Ndim = m_sysdef->getNDimensions();
    //unsigned int Nglobal = m_pdata->getNGlobal();
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

    // Choose an aspect ratio and randomly perturb it
    unsigned int i = hoomd::UniformIntDistribution(Ndim - 1)(rng);
    Scalar dA = Scalar(1.0) + hoomd::UniformDistribution<Scalar>(Scalar(0.0), m_Aspect_delta)(rng);
    if (hoomd::UniformIntDistribution(1)(rng))
        {
        dA = Scalar(1.0)/dA;
        }
    newL[i] *= dA;
    Scalar lambda = curL[i] / newL[i];
    if (Ndim == 3)
        {
        lambda = sqrt(lambda);
        }
    for (unsigned int j=0; j < Ndim; j++)
        {
        if (i != j)
            {
            newL[j] = lambda * curL[j];
            }
        }

    // Attempt box resize
    bool trial_success = box_resize_trial(newL[0],
                                          newL[1],
                                          newL[2],
                                          newShear[0],
                                          newShear[1],
                                          newShear[2],
                                          timestep,
                                          Scalar(0.0),
                                          rng);
    if (trial_success)
        {
        m_count_total.aspect_accept_count++;
        }
    else
        {
        m_count_total.aspect_reject_count++;
        }

    if (m_prof) m_prof->pop();
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    UpdaterBoxMC maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
hpmc_boxmc_counters_t UpdaterBoxMC::getCounters(unsigned int mode)
    {
    hpmc_boxmc_counters_t result;

    if (mode == 0)
        result = m_count_total;
    else if (mode == 1)
        result = m_count_total - m_count_run_start;
    else
        result = m_count_total - m_count_step_start;

    // don't MPI_AllReduce counters because all ranks count the same thing
    return result;
    }

void export_UpdaterBoxMC(py::module& m)
    {
   py::class_< UpdaterBoxMC, std::shared_ptr< UpdaterBoxMC > >(m, "UpdaterBoxMC", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr<IntegratorHPMC>,
                         std::shared_ptr<Variant>,
                         Scalar,
                         const unsigned int >())
    .def("volume", &UpdaterBoxMC::volume)
    .def("ln_volume", &UpdaterBoxMC::ln_volume)
    .def("length", &UpdaterBoxMC::length)
    .def("shear", &UpdaterBoxMC::shear)
    .def("aspect", &UpdaterBoxMC::aspect)
    .def("printStats", &UpdaterBoxMC::printStats)
    .def("resetStats", &UpdaterBoxMC::resetStats)
    .def("getP", &UpdaterBoxMC::getP)
    .def("setP", &UpdaterBoxMC::setP)
    .def("get_volume_delta", &UpdaterBoxMC::get_volume_delta)
    .def("get_ln_volume_delta", &UpdaterBoxMC::get_ln_volume_delta)
    .def("get_length_delta", &UpdaterBoxMC::get_length_delta)
    .def("get_shear_delta", &UpdaterBoxMC::get_shear_delta)
    .def("get_aspect_delta", &UpdaterBoxMC::get_aspect_delta)
//    .def("getMoveRatio", &UpdaterBoxMC::getMoveRatio)
//    .def("getReduce", &UpdaterBoxMC::getReduce)
//    .def("getIsotropic", &UpdaterBoxMC::getIsotropic)
    .def("computeAspectRatios", &UpdaterBoxMC::computeAspectRatios)
    .def("getCounters", &UpdaterBoxMC::getCounters)
    ;

   py::class_< hpmc_boxmc_counters_t >(m, "hpmc_boxmc_counters_t")
    .def_readwrite("volume_accept_count", &hpmc_boxmc_counters_t::volume_accept_count)
    .def_readwrite("volume_reject_count", &hpmc_boxmc_counters_t::volume_reject_count)
    .def_readwrite("ln_volume_accept_count", &hpmc_boxmc_counters_t::ln_volume_accept_count)
    .def_readwrite("ln_volume_reject_count", &hpmc_boxmc_counters_t::ln_volume_reject_count)
    .def_readwrite("shear_accept_count", &hpmc_boxmc_counters_t::shear_accept_count)
    .def_readwrite("shear_reject_count", &hpmc_boxmc_counters_t::shear_reject_count)
    .def_readwrite("aspect_accept_count", &hpmc_boxmc_counters_t::aspect_accept_count)
    .def_readwrite("aspect_reject_count", &hpmc_boxmc_counters_t::aspect_reject_count)
    .def("getVolumeAcceptance", &hpmc_boxmc_counters_t::getVolumeAcceptance)
    .def("getLogVolumeAcceptance", &hpmc_boxmc_counters_t::getLogVolumeAcceptance)
    .def("getShearAcceptance", &hpmc_boxmc_counters_t::getShearAcceptance)
    .def("getAspectAcceptance", &hpmc_boxmc_counters_t::getAspectAcceptance)
    .def("getNMoves", &hpmc_boxmc_counters_t::getNMoves)
    ;
    }

} // end namespace hpmc
