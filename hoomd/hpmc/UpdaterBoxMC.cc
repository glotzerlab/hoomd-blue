// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "UpdaterBoxMC.h"
#include "hoomd/RNGIdentifiers.h"
#include <numeric>
#include <vector>

/*! \file UpdaterBoxMC.cc
    \brief Definition of UpdaterBoxMC
*/

namespace hoomd
    {
namespace hpmc
    {
UpdaterBoxMC::UpdaterBoxMC(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Trigger> trigger,
                           std::shared_ptr<IntegratorHPMC> mc,
                           std::shared_ptr<Variant> P)
    : Updater(sysdef, trigger), m_mc(mc), m_beta_P(P), m_volume_delta(0.0), m_volume_weight(0.0),
      m_ln_volume_delta(0.0), m_ln_volume_weight(0.0), m_volume_mode("standard"), m_volume_A1(0.0),
      m_volume_A2(0.0), m_length_delta {0.0, 0.0, 0.0}, m_length_weight(0.0),
      m_shear_delta {0.0, 0.0, 0.0}, m_shear_weight(0.0), m_shear_reduce(0.0), m_aspect_delta(0.0),
      m_aspect_weight(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterBoxMC" << std::endl;

    // initialize stats
    resetStats();

    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<UpdaterBoxMC, &UpdaterBoxMC::slotMaxNChange>(this);

    updateChangedWeights();
    }

UpdaterBoxMC::~UpdaterBoxMC()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterBoxMC" << std::endl;
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<UpdaterBoxMC, &UpdaterBoxMC::slotMaxNChange>(this);
    }

/*! Determine if box exceeds a shearing threshold and needs to be lattice reduced.

    The maximum amount of shear to allow is somewhat arbitrary, but must be > 0.5. Small values mean
   the box is reconstructed more often, making it more confusing to track particle diffusion. Larger
   shear values mean parallel box planes can get closer together, reducing the number of cells
   possible in the cell list or increasing the number of images that must be checked for small
   boxes.

    Box is oversheared in direction \f$ \hat{e}_i \f$ if
    \f$ \bar{e}_j \cdot \hat{e}_i >= reduce * \left| \bar{e}_i \right| \f$
    or
    \f$ \bar{e}_j \cdot \bar{e}_i >= reduce * \left| \bar{e}_i \right| ^2 \f$
    \f$ = reduce * \bar{e}_i \cdot \bar{e}_i \f$

    \returns bool true if box is overly sheared
*/
inline bool UpdaterBoxMC::is_oversheared()
    {
    if (m_shear_reduce <= 0.5)
        return false;

    const BoxDim curBox = m_pdata->getGlobalBox();
    const Scalar3 x = curBox.getLatticeVector(0);
    const Scalar3 y = curBox.getLatticeVector(1);
    const Scalar3 z = curBox.getLatticeVector(2);

    const Scalar y_x = y.x; // x component of y vector
    const Scalar max_y_x = x.x * m_shear_reduce;
    const Scalar z_x = z.x; // x component of z vector
    const Scalar max_z_x = x.x * m_shear_reduce;
    // z_y \left| y \right|
    const Scalar z_yy = dot(z, y);
    // MAX_SHEAR * left| y \right| ^2
    const Scalar max_z_y_2 = dot(y, y) * m_shear_reduce;

    if (fabs(y_x) > max_y_x || fabs(z_x) > max_z_x || fabs(z_yy) > max_z_y_2)
        return true;
    else
        return false;
    }

/*! Perform lattice reduction.
    Remove excessive box shearing by finding a more cubic degenerate lattice
    when shearing is more half a lattice vector from cubic. The lattice reduction could make data
   needlessly complicated and may break detailed balance, use judiciously.

    \returns true if overshear was removed
*/
inline bool UpdaterBoxMC::remove_overshear()
    {
    bool overshear = false;                // initialize return value
    const Scalar MAX_SHEAR = Scalar(0.5f); // lattice can be reduced if shearing exceeds this value

    BoxDim newBox = m_pdata->getGlobalBox();
    Scalar3 x = newBox.getLatticeVector(0);
    Scalar3 y = newBox.getLatticeVector(1);
    Scalar3 z = newBox.getLatticeVector(2);
    Scalar xy = newBox.getTiltFactorXY();
    Scalar xz = newBox.getTiltFactorXZ();
    Scalar yz = newBox.getTiltFactorYZ();

    // Remove one lattice vector of shear if necessary. Only apply once so image doesn't change more
    // than one.

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
    const Scalar z_yy = dot(z, y);
    // MAX_SHEAR * left| y \right| ^2
    const Scalar max_z_y_2 = dot(y, y) * MAX_SHEAR;
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
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(),
                                      access_location::host,
                                      access_mode::readwrite);
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
                                           uint64_t timestep,
                                           double delta_beta_H,
                                           double log_V_term,
                                           hoomd::RandomGenerator& rng)
    {
    // Make a backup copy of position data
    unsigned int N_backup = m_pdata->getN();
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup,
                                          access_location::host,
                                          access_mode::overwrite);
        memcpy(h_pos_backup.data, h_pos.data, sizeof(Scalar4) * N_backup);
        }

    BoxDim curBox = m_pdata->getGlobalBox();
    double delta_U_pair = 0;
    double delta_U_external = 0;

    // energy of old configuration
    delta_U_pair -= m_mc->computeTotalPairEnergy(timestep);
    delta_U_external -= m_mc->computeTotalExternalEnergy(false);

    // Attempt box resize and check for overlaps
    BoxDim newBox = m_pdata->getGlobalBox();

    newBox.setL(make_scalar3(Lx, Ly, Lz));
    newBox.setTiltFactors(xy, xz, yz);

    Scalar3 old_origin = m_pdata->getOrigin();
    bool allowed = m_mc->attemptBoxResize(timestep, newBox);
    Scalar3 new_origin = m_pdata->getOrigin();
    Scalar3 origin_shift = new_origin - old_origin;

    if (allowed)
        {
        delta_U_pair += m_mc->computeTotalPairEnergy(timestep);
        delta_U_external += m_mc->computeTotalExternalEnergy(true);
        }

    if (allowed && m_mc->getExternalField())
        {
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup,
                                          access_location::host,
                                          access_mode::readwrite);
        Scalar ext_energy = m_mc->getExternalField()->calculateDeltaE(timestep,
                                                                      h_pos_backup.data,
                                                                      NULL,
                                                                      curBox,
                                                                      old_origin);
        delta_U_external += ext_energy;
        }

    double p = hoomd::detail::generate_canonical<double>(rng);

    const Scalar kT = (*m_mc->getKT())(timestep);

    if (allowed
        && p < (exp(-(delta_U_pair + delta_U_external) / kT) * exp(-delta_beta_H + log_V_term)))
        {
        return true;
        }
    else
        {
            // Restore original box and particle positions
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_backup(m_pos_backup,
                                              access_location::host,
                                              access_mode::read);
            unsigned int N = m_pdata->getN();
            if (N != N_backup)
                {
                this->m_exec_conf->msg->error()
                    << "update.boxmc" << ": Number of particles mismatch when rejecting box resize"
                    << std::endl;
                throw std::runtime_error("Error resizing box");
                // note, this error should never appear (because particles are not migrated after a
                // box resize), but is left here as a sanity check
                }
            memcpy(h_pos.data, h_pos_backup.data, sizeof(Scalar4) * N);
            }

        m_pdata->setGlobalBox(curBox);

        // reset origin
        m_pdata->translateOrigin(-origin_shift);

        // we have moved particles, communicate those changes
        m_mc->communicate(false);
        return false;
        }
    }

inline bool UpdaterBoxMC::safe_box(const Scalar newL[3], const unsigned int& Ndim)
    {
    // Scalar min_allowed_size = m_mc->getMaxTransMoveSize(); // This is dealt with elsewhere
    const Scalar min_allowed_size(0.0); // volume must be kept positive
    for (unsigned int j = 0; j < Ndim; j++)
        {
        if ((newL[j]) < min_allowed_size)
            {
            // volume must be kept positive
            m_exec_conf->msg->notice(10)
                << "Box unsafe because dimension " << j << " would be negative." << std::endl;
            return false;
            }
        }
    return true;
    }

/*! Perform Metropolis Monte Carlo box resizes and shearing
    \param timestep Current time step of the simulation
*/
void UpdaterBoxMC::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_count_step_start = m_count_total;
    m_exec_conf->msg->notice(10) << "UpdaterBoxMC: " << timestep << std::endl;

    // Create a prng instance for this timestep
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::UpdaterBoxMC, timestep, m_sysdef->getSeed()),
        hoomd::Counter(m_instance));

    // Choose a move type
    auto const weight_total = m_weight_partial_sums.back();
    if (weight_total == 0.0)
        {
        // Attempt to execute with all move weights equal to zero.
        m_exec_conf->msg->warning()
            << "No move types with non-zero weight. UpdaterBoxMC has nothing to do." << std::endl;
        return;
        }

    // Generate a number between (0, weight_total]
    auto const selected = hoomd::detail::generate_canonical<Scalar>(rng) * weight_total;
    // Select the first move type whose partial sum of weights is greater than
    // or equal to the generated value.
    auto const move_type_select = std::distance(
        m_weight_partial_sums.cbegin(),
        std::lower_bound(m_weight_partial_sums.cbegin(), m_weight_partial_sums.cend(), selected));

    // Attempt and evaluate a move
    // This section will need to be updated when move types are added.
    if (move_type_select == 0)
        {
        // Isotropic volume change
        m_exec_conf->msg->notice(8) << "Volume move performed at step " << timestep << std::endl;
        update_V(timestep, rng);
        }
    else if (move_type_select == 1)
        {
        // Isotropic volume change in logarithmic steps
        m_exec_conf->msg->notice(8) << "lnV move performed at step " << timestep << std::endl;
        update_lnV(timestep, rng);
        }
    else if (move_type_select == 2)
        {
        // Volume change in distribution of box lengths
        m_exec_conf->msg->notice(8)
            << "Box length move performed at step " << timestep << std::endl;
        update_L(timestep, rng);
        }
    else if (move_type_select == 3)
        {
        // Shear change
        m_exec_conf->msg->notice(8) << "Box shear move performed at step " << timestep << std::endl;
        update_shear(timestep, rng);
        }
    else if (move_type_select == 4)
        {
        // Volume conserving aspect change
        m_exec_conf->msg->notice(8)
            << "Box aspect move performed at step " << timestep << std::endl;
        update_aspect(timestep, rng);
        }
    else
        {
        // Should not reach this point
        m_exec_conf->msg->warning()
            << "UpdaterBoxMC selected an unassigned move type. Selected " << move_type_select
            << " from range " << weight_total << std::endl;
        return;
        }

    if (is_oversheared())
        {
        while (remove_overshear())
            {
            }; // lattice reduction, possibly in several steps
        m_exec_conf->msg->notice(5)
            << "Lattice reduction performed at step " << timestep << std::endl;
        }
    }

void UpdaterBoxMC::update_L(uint64_t timestep, hoomd::RandomGenerator& rng)
    {
    // Get updater parameters for current timestep
    Scalar beta_P = (*m_beta_P)(timestep);

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
        if (m_length_delta[i] != 0.0)
            nonzero_dim++;

    if (nonzero_dim == 0)
        {
        // all dimensions have delta==0, just count as accepted and return
        m_count_total.volume_accept_count++;
        return;
        }

    unsigned int chosen_nonzero_dim = hoomd::UniformIntDistribution(nonzero_dim - 1)(rng);
    unsigned int nonzero_dim_count = 0;
    unsigned int i = 0;
    for (unsigned int j = 0; j < Ndim; ++j)
        {
        if (m_length_delta[j] != 0.0)
            {
            if (nonzero_dim_count == chosen_nonzero_dim)
                {
                i = j;
                break;
                }
            ++nonzero_dim_count;
            }
        }

    Scalar dL_max(m_length_delta[i]);

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
        if (Ndim == 3)
            Vold *= curL[2];
        Vnew = newL[0] * newL[1];
        if (Ndim == 3)
            Vnew *= newL[2];
        dV = Vnew - Vold;

        // Calculate Boltzmann factor
        double delta_beta_H = beta_P * dV;
        double log_V_term = Nglobal * log(Vnew / Vold);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                       newL[1],
                                       newL[2],
                                       newShear[0],
                                       newShear[1],
                                       newShear[2],
                                       timestep,
                                       delta_beta_H,
                                       log_V_term,
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
    }

//! Update the box volume in logarithmic steps
void UpdaterBoxMC::update_lnV(uint64_t timestep, hoomd::RandomGenerator& rng)
    {
    // Get updater parameters for current timestep
    Scalar beta_P = (*m_beta_P)(timestep);

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
    Scalar A1 = m_volume_A1;
    Scalar A2 = m_volume_A2;

    // Volume change
    Scalar dlnV_max(m_ln_volume_delta);

    // Choose a volume change
    Scalar dlnV = hoomd::UniformDistribution<Scalar>(-dlnV_max, dlnV_max)(rng);
    Scalar new_V = V * exp(dlnV);

    // perform isotropic volume change
    if (Ndim == 3)
        {
        newL[0] = pow(A1 * A2 * new_V, (1. / 3.));
        newL[1] = newL[0] / A1;
        newL[2] = newL[0] / A2;
        }
    else // Ndim ==2
        {
        newL[0] = pow(A1 * new_V, (1. / 2.));
        newL[1] = newL[0] / A1;
        // newL[2] is already assigned to curL[2]
        }

    if (!safe_box(newL, Ndim))
        {
        m_count_total.ln_volume_reject_count++;
        }
    else
        {
        // Calculate Boltzmann factor
        double delta_beta_H = beta_P * (new_V - V);
        double log_V_term = (Nglobal + 1) * log(new_V / V);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                       newL[1],
                                       newL[2],
                                       newShear[0],
                                       newShear[1],
                                       newShear[2],
                                       timestep,
                                       delta_beta_H,
                                       log_V_term,
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
    }

void UpdaterBoxMC::update_V(uint64_t timestep, hoomd::RandomGenerator& rng)
    {
    // Get updater parameters for current timestep
    Scalar beta_P = (*m_beta_P)(timestep);

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
    Scalar A1 = m_volume_A1;
    Scalar A2 = m_volume_A2;

    // Volume change
    Scalar dV_max(m_volume_delta);

    // Choose a volume change
    Scalar dV = hoomd::UniformDistribution<Scalar>(-dV_max, dV_max)(rng);

    // perform isotropic volume change
    if (Ndim == 3)
        {
        newL[0] = pow((A1 * A2 * (V + dV)), (1. / 3.));
        newL[1] = newL[0] / A1;
        newL[2] = newL[0] / A2;
        }
    else // Ndim ==2
        {
        newL[0] = pow((A1 * (V + dV)), (1. / 2.));
        newL[1] = newL[0] / A1;
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
        double delta_beta_H = beta_P * dV;
        double log_V_term = Nglobal * log(Vnew / V);

        // attempt box change
        bool accept = box_resize_trial(newL[0],
                                       newL[1],
                                       newL[2],
                                       newShear[0],
                                       newShear[1],
                                       newShear[2],
                                       timestep,
                                       delta_beta_H,
                                       log_V_term,
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
    }

void UpdaterBoxMC::update_shear(uint64_t timestep, hoomd::RandomGenerator& rng)
    {
    // Get updater parameters for current timestep
    // Get current particle data and box lattice parameters
    assert(m_pdata);
    unsigned int Ndim = m_sysdef->getNDimensions();
    // unsigned int Nglobal = m_pdata->getNGlobal();
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
    dA_max = m_shear_delta[i];
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
    }

void UpdaterBoxMC::update_aspect(uint64_t timestep, hoomd::RandomGenerator& rng)
    {
    // We have not established what ensemble this samples:
    // This is not a thermodynamic updater.
    // There is also room for improvement in enforcing volume conservation.
    // Get updater parameters for current timestep
    // Get current particle data and box lattice parameters
    assert(m_pdata);
    unsigned int Ndim = m_sysdef->getNDimensions();
    // unsigned int Nglobal = m_pdata->getNGlobal();
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
    Scalar dA = Scalar(1.0) + hoomd::UniformDistribution<Scalar>(Scalar(0.0), m_aspect_delta)(rng);
    if (hoomd::UniformIntDistribution(1)(rng))
        {
        dA = Scalar(1.0) / dA;
        }
    newL[i] *= dA;
    Scalar lambda = curL[i] / newL[i];
    if (Ndim == 3)
        {
        lambda = sqrt(lambda);
        }
    for (unsigned int j = 0; j < Ndim; j++)
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
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the
   last executed step \return The current state of the acceptance counters

    UpdaterBoxMC maintains a count of the number of accepted and rejected moves since instantiation.
   getCounters() provides the current value. The parameter *mode* controls whether the returned
   counts are absolute, relative to the start of the run, or relative to the start of the last
   executed step.
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

namespace detail
    {
void export_UpdaterBoxMC(pybind11::module& m)
    {
    pybind11::class_<UpdaterBoxMC, Updater, std::shared_ptr<UpdaterBoxMC>>(m, "UpdaterBoxMC")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMC>,
                            std::shared_ptr<Variant>>())
        .def_property("volume", &UpdaterBoxMC::getVolumeParams, &UpdaterBoxMC::setVolumeParams)
        .def_property("length", &UpdaterBoxMC::getLengthParams, &UpdaterBoxMC::setLengthParams)
        .def_property("shear", &UpdaterBoxMC::getShearParams, &UpdaterBoxMC::setShearParams)
        .def_property("aspect", &UpdaterBoxMC::getAspectParams, &UpdaterBoxMC::setAspectParams)
        .def_property("betaP", &UpdaterBoxMC::getBetaP, &UpdaterBoxMC::setBetaP)
        .def("getCounters", &UpdaterBoxMC::getCounters)
        .def_property("instance", &UpdaterBoxMC::getInstance, &UpdaterBoxMC::setInstance);

    pybind11::class_<hpmc_boxmc_counters_t>(m, "hpmc_boxmc_counters_t")
        .def_property_readonly("volume",
                               [](const hpmc_boxmc_counters_t& a)
                               {
                                   pybind11::tuple result;
                                   result = pybind11::make_tuple(a.volume_accept_count,
                                                                 a.volume_reject_count);
                                   return result;
                               })
        .def_property_readonly("ln_volume",
                               [](const hpmc_boxmc_counters_t& a)
                               {
                                   pybind11::tuple result;
                                   result = pybind11::make_tuple(a.ln_volume_accept_count,
                                                                 a.ln_volume_reject_count);
                                   return result;
                               })
        .def_property_readonly("aspect",
                               [](const hpmc_boxmc_counters_t& a)
                               {
                                   pybind11::tuple result;
                                   result = pybind11::make_tuple(a.aspect_accept_count,
                                                                 a.aspect_reject_count);
                                   return result;
                               })
        .def_property_readonly("shear",
                               [](const hpmc_boxmc_counters_t& a)
                               {
                                   pybind11::tuple result;
                                   result = pybind11::make_tuple(a.shear_accept_count,
                                                                 a.shear_reject_count);
                                   return result;
                               });
    }

    } // end namespace detail

void UpdaterBoxMC::updateChangedWeights()
    {
    // This line will need to be rewritten or updated when move types are added to the updater.
    auto const weights = std::vector<Scalar> {m_volume_weight,
                                              m_ln_volume_weight,
                                              m_length_weight,
                                              m_shear_weight,
                                              m_aspect_weight};
    m_weight_partial_sums = std::vector<Scalar>(weights.size());
    std::partial_sum(weights.cbegin(), weights.cend(), m_weight_partial_sums.begin());
    }

    } // end namespace hpmc
    } // end namespace hoomd
