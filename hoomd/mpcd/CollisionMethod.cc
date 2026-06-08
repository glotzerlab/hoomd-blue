// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Definition of mpcd::CollisionMethod
 */

#include "CollisionMethod.h"
#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif // ENABLE_MPI
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param seed Seed to pseudo-random number generator
 */
mpcd::CollisionMethod::CollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                                       uint64_t cur_timestep,
                                       uint64_t period,
                                       int phase)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(sysdef->getMPCDParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_period(period), m_checked_collision_warnings(false), m_needs_temperature(false),
      m_initial_velocity(m_exec_conf), m_linmom_accum(m_exec_conf), m_angmom_accum(m_exec_conf)
    {
    // setup next timestep for collision
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        uint64_t multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_drawrandvec_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                   m_exec_conf,
                                                   "mpcd_rigid_rand"));
        m_netvelo_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "mpcd_rigid_netvelo"));
        m_applyrandvec_tuner.reset(
            new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                             m_exec_conf,
                             "mpcd_rigid_applyrand"));
        m_store_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                             m_exec_conf,
                                             "mpcd_rigid_store"));
        m_accumulate_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                  m_exec_conf,
                                                  "mpcd_rigid_accum"));
        m_transfer_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                m_exec_conf,
                                                "mpcd_rigid_transfer"));
        }
#endif // ENABLE_HIP
    }

void mpcd::CollisionMethod::collide(uint64_t timestep)
    {
    if (!shouldCollide(timestep))
        return;

    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // sync the embedded group
    m_cl->setEmbeddedGroup(m_embed_group);

    // check for collision warnings the first time collide is run
    if (!m_checked_collision_warnings)
        {
        checkCollisionWarnings(timestep);
        m_checked_collision_warnings = true;
        }

    if (m_needs_temperature && !m_T)
        {
        throw std::runtime_error("Temperature required by collision method");
        }

    // check the GPU autotuners
#ifdef ENABLE_HIP
    if (m_check_rigid_tuners)
        {
        checkRigidAutotuners();
        m_check_rigid_tuners = false;
        }
#endif // ENABLE_HIP

    // set random grid shift
    m_cl->drawGridShift(timestep);

    // update cell list
    m_cl->compute(timestep);

    // setup rigid bodies before collision happens
    const bool rigid_body_collision = m_embed_group && m_rigid_bodies;
    if (rigid_body_collision)
        {
        // resize initial velocity array
        const unsigned int num_group = m_embed_group->getNumMembers();
        if (num_group > m_initial_velocity.getNumElements())
            {
            GPUArray<Scalar4> initial_velocity(num_group, m_exec_conf);
            m_initial_velocity.swap(initial_velocity);
            }

        const unsigned int num_total = m_pdata->getN() + m_pdata->getNGhosts();
        if (num_total > m_linmom_accum.getNumElements())
            {
            GPUArray<Scalar3> linmom_accum(num_total, m_exec_conf);
            m_linmom_accum.swap(linmom_accum);
            GPUArray<Scalar3> angmom_accum(num_total, m_exec_conf);
            m_angmom_accum.swap(angmom_accum);
            }

#ifdef ENABLE_HIP
        // thermalize rigid bodies and store constituent velocities
        if (m_exec_conf->isCUDAEnabled())
            {
            beginThermalizeConstituentParticlesGPU(timestep);
            finishThermalizeConstituentParticlesGPU(timestep);
            storeInitialEmbeddedGroupVelocitiesGPU(timestep);
            }
        else
#endif
            {
            beginThermalizeConstituentParticles(timestep);
            finishThermalizeConstituentParticles(timestep);
            storeInitialEmbeddedGroupVelocities(timestep);
            }
        }

    // apply collisions
    rule(timestep);

    // apply collisions to rigid bodies
    if (rigid_body_collision)
        {
#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            accumulateRigidBodyMomentaGPU(timestep);
            transferRigidBodyMomentaGPU(timestep);
            }
        else
#endif
            {
            accumulateRigidBodyMomenta(timestep);
            transferRigidBodyMomenta(timestep);
            }
        m_rigid_bodies->updateCompositeParticles(timestep);
        }
    }

/*! Check for warnings and errors related to collision step and rigid bodies
 * \warning If the central particle of a rigid body participates in collision
 * \throws If any embedded particles have zero mass and no momentum to transfer
 * \throws If rigid body particles participate in collision but cannot be thermostatted
 * \throws If center of mass of rigid body particles is not at the central particle
 * \throws If sum of constituent particles' masses is not equal to mass of central particle
 */
void mpcd::CollisionMethod::checkCollisionWarnings(uint64_t timestep)
    {
    if (m_embed_group)
        {
        // initialize variables for if warnings or errors exist
        bool invalid_mass = false;
        bool needs_thermostat = false;
        bool central_interacting = false;
        bool invalid_center_of_mass = false;
        bool invalid_mass_sum = false;
        std::set<unsigned int> rigid_types;

            // check embedded particles for warnings
            {
            const unsigned int num_group = m_embed_group->getNumMembers();
            ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                                    access_location::host,
                                                    access_mode::read);
            ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_body_embed(m_pdata->getBodies(),
                                                   access_location::host,
                                                   access_mode::read);
            ArrayHandle<unsigned int> h_rtag_embed(m_pdata->getRTags(),
                                                   access_location::host,
                                                   access_mode::read);
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);

            for (unsigned int idx = 0; idx < num_group; ++idx)
                {
                // get the index from the embedded group
                const unsigned int particle_index = h_embed_group.data[idx];

                // check particle has mass
                const Scalar4 vel_mass = h_velocity.data[particle_index];
                const Scalar mass = vel_mass.w;
                if (mass <= Scalar(0))
                    {
                    invalid_mass = true;
                    }

                // check if particle is central particle in rigid body
                const unsigned int central_tag = h_body_embed.data[particle_index];
                if (central_tag < MIN_FLOPPY)
                    {
                    needs_thermostat = true;
                    const unsigned int central_idx = h_rtag_embed.data[central_tag];
                    if (particle_index == central_idx)
                        {
                        central_interacting = true;
                        }
                    const unsigned int central_type
                        = __scalar_as_int(h_postype.data[central_idx].w);
                    rigid_types.insert(central_type);
                    }
                }
            }
        if (m_sysdef->isDomainDecomposed() && m_rigid_bodies && !rigid_types.empty())
            {
            throw std::runtime_error("Rigid bodies with MPCD unavailable for decomposed domain");
            }
        // go through each molecule and check if the center of mass is in right location
        if (m_rigid_bodies && !rigid_types.empty())
            {
            const Scalar tol(1e-6);
            // access molecule order
            ArrayHandle<unsigned int> h_molecule_order(m_rigid_bodies->getMoleculeOrder(),
                                                       access_location::host,
                                                       access_mode::read);
            ArrayHandle<unsigned int> h_molecule_len(m_rigid_bodies->getMoleculeLengths(),
                                                     access_location::host,
                                                     access_mode::read);
            ArrayHandle<unsigned int> h_molecule_idx(m_rigid_bodies->getMoleculeIndex(),
                                                     access_location::host,
                                                     access_mode::read);
            ArrayHandle<unsigned int> h_molecule_list(m_rigid_bodies->getMoleculeList(),
                                                      access_location::host,
                                                      access_mode::read);
            Index2D molecule_indexer = m_rigid_bodies->getMoleculeIndexer();
            const unsigned int nmol = molecule_indexer.getH();
            // access body definitions
            ArrayHandle<Scalar3> h_body_pos(m_rigid_bodies->getBodyOffsets(),
                                            access_location::host,
                                            access_mode::read);
            Index2D h_body_idx = m_rigid_bodies->getBodyIndexer();
            // accesss particle data
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);
            ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                             access_location::host,
                                             access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                             access_location::host,
                                             access_mode::read);
            // loop over all molecules
            const unsigned int n_particles_local = m_pdata->getN() + m_pdata->getNGhosts();
            for (unsigned int ibody = 0; ibody < nmol && !invalid_center_of_mass; ++ibody)
                {
                // get central particle tag from first particle in molecule
                assert(h_molecule_len.data[ibody] > 0);
                const unsigned int first_idx = h_molecule_list.data[molecule_indexer(0, ibody)];
                if (first_idx > n_particles_local)
                    continue;
                const unsigned int central_tag = h_body.data[first_idx];
                const unsigned int central_idx = h_rtag.data[central_tag];
                if (central_idx >= n_particles_local)
                    continue;

                // only do molecules participating in the collision
                const unsigned int type = __scalar_as_int(h_postype.data[central_idx].w);
                if (rigid_types.find(type) == rigid_types.end())
                    {
                    continue;
                    }

                // find center of mass position of particle
                Scalar3 center_of_mass = make_scalar3(0.0, 0.0, 0.0);
                Scalar mass_sum = Scalar(0.0);
                const Scalar center_mass = h_velocity.data[central_idx].w;
                for (unsigned int constituent_index = 0;
                     constituent_index < h_molecule_len.data[ibody];
                     ++constituent_index)
                    {
                    const unsigned int idxj
                        = h_molecule_list.data[molecule_indexer(constituent_index, ibody)];
                    assert(idxj < m_pdata->getN() + m_pdata->getNGhosts());
                    if (idxj == central_idx)
                        continue;

                    // get relative position and mass of constituent
                    const unsigned int idx_in_body = h_molecule_order.data[idxj] - 1;
                    const Scalar3 local_pos(h_body_pos.data[h_body_idx(type, idx_in_body)]);
                    const Scalar mass = h_velocity.data[idxj].w;
                    // add to accumulating center of mass
                    center_of_mass += mass * local_pos;
                    mass_sum += mass;
                    }
                // check if center of mass in body frame is (0, 0, 0)
                invalid_center_of_mass
                    |= !((center_of_mass.x >= -tol && center_of_mass.x <= tol)
                         && (center_of_mass.y >= -tol && center_of_mass.y <= tol)
                         && (center_of_mass.z >= -tol && center_of_mass.z <= tol));

                // check if center of mass is the same as the sum of masses
                const Scalar mass_diff = center_mass - mass_sum;
                const bool sum_in_tol = mass_diff >= -tol && mass_diff <= tol;
                invalid_mass_sum |= !sum_in_tol;
                }
            }

#ifdef ENABLE_MPI
        if (m_exec_conf->getNRanks() > 1)
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &invalid_mass,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                          &needs_thermostat,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                          &central_interacting,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                          &invalid_center_of_mass,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                          &invalid_mass_sum,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            }
#endif // ENABLE_MPI

        if (needs_thermostat)
            {
            requireTemperature();
            }
        if (central_interacting)
            {
            m_exec_conf->msg->warning() << "Central particle of rigid body included in "
                                           "MPCD collision. Check if this is intentional."
                                        << std::endl;
            }
        if (invalid_mass)
            {
            throw std::runtime_error("Some particles have a mass <= 0.");
            }
        if (invalid_center_of_mass)
            {
            throw std::runtime_error(
                "Some rigid bodies do not have center of mass at central particle.");
            }
        if (invalid_mass_sum)
            {
            throw std::runtime_error(
                "Mass of some rigid bodies not equal to sum of constituent particle masses.");
            }
        }
    }

void mpcd::CollisionMethod::beginThermalizeConstituentParticles(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

    const Scalar T_set = (*m_T)(timestep);

    ArrayHandle<unsigned int> h_lookup_center(m_rigid_bodies->getLookupCenters(),
                                              access_location::host,
                                              access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> h_alt_vel(m_pdata->getAltVelocities(),
                                   access_location::host,
                                   access_mode::overwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);

    const unsigned int num_total = m_pdata->getN();
    const uint16_t seed = m_sysdef->getSeed();

    // get random velocities and accumulate the resulting change in momentum
    for (unsigned int idx = 0; idx < num_total; ++idx)
        {
        // get the index from the particle and check if it is a constituent
        const unsigned int central_idx = h_lookup_center.data[idx];
        if (central_idx == NO_BODY || idx == central_idx)
            {
            continue;
            }

        const Scalar mass_const = h_velocity.data[idx].w;
        // don't thermalize particles with zero mass
        if (mass_const == Scalar(0))
            {
            continue;
            }

        // draw random velocities from normal distribution
        const unsigned int tag = h_tag.data[idx];
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::CollisionMethod, timestep, seed),
            hoomd::Counter(tag, 1));
        hoomd::NormalDistribution<Scalar> gen(fast::sqrt(T_set / mass_const), 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        h_alt_vel.data[idx] = make_scalar4(vel.x, vel.y, vel.z, mass_const);

        // add to net momentum
        const Scalar3 linmom_change = mass_const * vel;
        h_linmom_accum.data[central_idx] += linmom_change;

        // get displacement
        const Scalar4 postype_const = h_postype.data[idx];
        const vec3<Scalar> pos_const(postype_const);
        const int3 img_const = h_image.data[idx];

        const Scalar4 postype_central = h_postype.data[central_idx];
        const vec3<Scalar> pos_central(postype_central);
        const int3 img_central = h_image.data[central_idx];

        vec3<Scalar> displacement = pos_const - pos_central;
        const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                                img_const.y - img_central.y,
                                                img_const.z - img_central.z);
        displacement = global_box.shift(displacement, displacement_img);

        // add to net momentum
        const vec3<Scalar> angmom_change = cross(displacement, vec3<Scalar>(linmom_change));
        h_angmom_accum.data[central_idx] += vec_to_scalar3(angmom_change);
        }
    }

void mpcd::CollisionMethod::finishThermalizeConstituentParticles(uint64_t timestep)
    {
    ArrayHandle<unsigned int> h_lookup_center(m_rigid_bodies->getLookupCenters(),
                                              access_location::host,
                                              access_mode::read);
    ArrayHandle<unsigned int> h_rigid_center(m_rigid_bodies->getRigidCenters(),
                                             access_location::host,
                                             access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> h_alt_vel(m_pdata->getAltVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);

    const unsigned int num_total = m_pdata->getN();
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    // get net linear velocity and net angular velocity of central particles
    const unsigned int num_centers = m_rigid_bodies->getNLocal();
    for (unsigned int idx = 0; idx < num_centers; ++idx)
        {
        // get index of central particle
        const unsigned int central_idx = h_rigid_center.data[idx];

        // store the net linear velocity in AltVelocities
        const Scalar mass = h_velocity.data[central_idx].w;
        const Scalar3 net_velocity = h_linmom_accum.data[central_idx] / mass;
        h_alt_vel.data[central_idx]
            = make_scalar4(net_velocity.x, net_velocity.y, net_velocity.z, mass);

        // get net angular momentum
        const vec3<Scalar> net_angmom(h_angmom_accum.data[central_idx]);
        const quat<Scalar> orientation(h_orientation.data[central_idx]);
        const vec3<Scalar> inertia(h_inertia.data[central_idx]);

        // rotate to body frame
        vec3<Scalar> net_angvel_body = rotate(conj(orientation), net_angmom);
        if (inertia.x != Scalar(0))
            {
            net_angvel_body.x = net_angvel_body.x / inertia.x;
            }
        else
            {
            net_angvel_body.x = Scalar(0);
            }
        if (inertia.y != Scalar(0))
            {
            net_angvel_body.y = net_angvel_body.y / inertia.y;
            }
        else
            {
            net_angvel_body.y = Scalar(0);
            }
        if (inertia.z != Scalar(0))
            {
            net_angvel_body.z = net_angvel_body.z / inertia.z;
            }
        else
            {
            net_angvel_body.z = Scalar(0);
            }
        const vec3<Scalar> net_angvel_space = rotate(orientation, net_angvel_body);

        // set net angular velocity in space frame
        h_angmom_accum.data[central_idx] = vec_to_scalar3(net_angvel_space);
        }

    // Subtract off net linear and angular velocity and apply to constituents
    for (unsigned int idx = 0; idx < num_total; ++idx)
        {
        // get the index from the particle and check if it is a constituent
        const unsigned int central_idx = h_lookup_center.data[idx];
        if (central_idx == NO_BODY || central_idx == idx)
            {
            continue;
            }

        // get velocities and masses
        Scalar4 vel_constituent = h_velocity.data[idx];
        if (vel_constituent.w == Scalar(0))
            {
            // constituents with zero mass don't get thermalized
            continue;
            }
        const Scalar4 thermal_vel_mass = h_alt_vel.data[idx];
        vec3<Scalar> thermal_vel(thermal_vel_mass);

        // get net angular and linear velocity
        const Scalar4 net_vel_mass = h_alt_vel.data[central_idx];
        const vec3<Scalar> net_vel(net_vel_mass);
        const vec3<Scalar> net_angvel_space(h_angmom_accum.data[central_idx]);

        // get displacement
        const Scalar4 postype_const = h_postype.data[idx];
        const vec3<Scalar> pos_const(postype_const);
        const int3 img_const = h_image.data[idx];

        const Scalar4 postype_central = h_postype.data[central_idx];
        const vec3<Scalar> pos_central(postype_central);
        const int3 img_central = h_image.data[central_idx];

        vec3<Scalar> displacement = pos_const - pos_central;
        const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                                img_const.y - img_central.y,
                                                img_const.z - img_central.z);
        displacement = global_box.shift(displacement, displacement_img);

        // subtract net velocities
        thermal_vel -= net_vel + cross(net_angvel_space, displacement);

        // add to constituent
        vel_constituent.x += thermal_vel.x;
        vel_constituent.y += thermal_vel.y;
        vel_constituent.z += thermal_vel.z;
        h_velocity.data[idx] = vel_constituent;
        }
    }

void mpcd::CollisionMethod::storeInitialEmbeddedGroupVelocities(uint64_t timestep)
    {
    const unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::read);
    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // collect the initial velocities of the embedded particles
        const unsigned int particle_index = h_embed_group.data[idx];
        h_initial_vel.data[idx] = h_velocity.data[particle_index];
        }
    }

void mpcd::CollisionMethod::accumulateRigidBodyMomenta(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

    ArrayHandle<unsigned int> h_lookup_center(m_rigid_bodies->getLookupCenters(),
                                              access_location::host,
                                              access_mode::read);

    const unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);

    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // get the index from the embedded group and check if in a rigid body
        const unsigned int particle_index = h_embed_group.data[idx];

        // get the index from the particle and check if it is a constituent
        const unsigned int central_idx = h_lookup_center.data[particle_index];
        if (central_idx == NO_BODY || central_idx == particle_index)
            {
            continue;
            }

        // get velocities and masses
        const Scalar4 vel_mass_const = h_velocity.data[particle_index];
        const vec3<Scalar> vel_const(vel_mass_const);
        const Scalar mass_const = vel_mass_const.w;

        // get displacement
        const Scalar4 postype_const = h_postype.data[particle_index];
        const vec3<Scalar> pos_const(postype_const);
        const int3 img_const = h_image.data[particle_index];

        const Scalar4 postype_central = h_postype.data[central_idx];
        const vec3<Scalar> pos_central(postype_central);
        const int3 img_central = h_image.data[central_idx];

        vec3<Scalar> displacement = pos_const - pos_central;
        const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                                img_const.y - img_central.y,
                                                img_const.z - img_central.z);
        displacement = global_box.shift(displacement, displacement_img);

        // change in linear and angular momentum
        const vec3<Scalar> initial_vel_const(h_initial_vel.data[idx]);
        const vec3<Scalar> linmom_change = mass_const * (vel_const - initial_vel_const);
        const vec3<Scalar> angmom_change = cross(displacement, linmom_change);

        // accumulate onto central particle
        h_linmom_accum.data[central_idx] += vec_to_scalar3(linmom_change);
        h_angmom_accum.data[central_idx] += vec_to_scalar3(angmom_change);
        }
    }

void mpcd::CollisionMethod::transferRigidBodyMomenta(uint64_t timestep)
    {
    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rigid_center(m_rigid_bodies->getRigidCenters(),
                                             access_location::host,
                                             access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    // add accumulated momentum to the central particle
    const unsigned int num_centers = m_rigid_bodies->getNLocal();
    for (unsigned int idx = 0; idx < num_centers; ++idx)
        {
        // get index of central particle
        const unsigned int central_idx = h_rigid_center.data[idx];

        // get accumulated momentum for particle
        const Scalar3 linmom_accum(h_linmom_accum.data[central_idx]);
        const vec3<Scalar> angmom_accum(h_angmom_accum.data[central_idx]);

        // compute and store new velocity
        Scalar4 vel_mass = h_velocity.data[central_idx];
        const Scalar mass = vel_mass.w;
        if (mass > 0)
            {
            vel_mass.x += linmom_accum.x / mass;
            vel_mass.y += linmom_accum.y / mass;
            vel_mass.z += linmom_accum.z / mass;
            h_velocity.data[central_idx] = vel_mass;
            }

        // compute new angular momentum
        quat<Scalar> angmom(h_angmom.data[central_idx]);
        const quat<Scalar> orientation(h_orientation.data[central_idx]);

        // convert angular momentum to quaternion and update
        const vec3<Scalar> inertia(h_inertia.data[central_idx]);
        vec3<Scalar> angmom_change_body = rotate(conj(orientation), angmom_accum);
        if (inertia.x == Scalar(0))
            {
            angmom_change_body.x = Scalar(0);
            }

        if (inertia.y == Scalar(0))
            {
            angmom_change_body.y = Scalar(0);
            }

        if (inertia.z == Scalar(0))
            {
            angmom_change_body.z = Scalar(0);
            }
        angmom += Scalar(2.0) * orientation * quat(0.0, angmom_change_body);

        // save update
        h_angmom.data[central_idx] = quat_to_scalar4(angmom);
        }
    }

#ifdef ENABLE_HIP
//! Create autotuners
void mpcd::CollisionMethod::checkRigidAutotuners()
    {
    std::vector<std::shared_ptr<AutotunerBase>> rigid_autotuners {m_drawrandvec_tuner,
                                                                  m_netvelo_tuner,
                                                                  m_applyrandvec_tuner,
                                                                  m_store_tuner,
                                                                  m_accumulate_tuner,
                                                                  m_transfer_tuner};
    if (m_exec_conf->isCUDAEnabled() && m_embed_group && m_rigid_bodies)
        {
        // add tuners if they aren't already there
        for (auto& rigid_tuner : rigid_autotuners)
            {
            if (std::find(m_autotuners.begin(), m_autotuners.end(), rigid_tuner)
                == m_autotuners.end())
                {
                m_autotuners.push_back(rigid_tuner);
                }
            }
        }
    else
        {
        // remove rigid tuners if present
        for (auto& rigid_tuner : rigid_autotuners)
            {
            auto it = std::find(m_autotuners.begin(), m_autotuners.end(), rigid_tuner);
            if (it != m_autotuners.end())
                {
                m_autotuners.erase(it);
                }
            }
        }
    }

void mpcd::CollisionMethod::beginThermalizeConstituentParticlesGPU(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

        {
        ArrayHandle<unsigned int> d_lookup_center(m_rigid_bodies->getLookupCenters(),
                                                  access_location::device,
                                                  access_mode::read);
        ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum,
                                            access_location::device,
                                            access_mode::readwrite);
        ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum,
                                            access_location::device,
                                            access_mode::readwrite);
        ArrayHandle<Scalar4> d_alt_vel(m_pdata->getAltVelocities(),
                                       access_location::device,
                                       access_mode::overwrite);
        ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                        access_location::device,
                                        access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                        access_location::device,
                                        access_mode::read);

        m_drawrandvec_tuner->begin();
        mpcd::gpu::draw_velocities_constituent_particles(d_linmom_accum.data,
                                                         d_angmom_accum.data,
                                                         d_alt_vel.data,
                                                         d_postype.data,
                                                         d_velocity.data,
                                                         d_image.data,
                                                         d_tag.data,
                                                         d_lookup_center.data,
                                                         m_pdata->getGlobalBox(),
                                                         timestep,
                                                         m_sysdef->getSeed(),
                                                         (*m_T)(timestep),
                                                         m_pdata->getN(),
                                                         m_drawrandvec_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_drawrandvec_tuner->end();
        }
    }

void mpcd::CollisionMethod::finishThermalizeConstituentParticlesGPU(uint64_t timestep)
    {
        {
        ArrayHandle<unsigned int> d_rigid_center(m_rigid_bodies->getRigidCenters(),
                                                 access_location::device,
                                                 access_mode::read);
        ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum,
                                            access_location::device,
                                            access_mode::read);
        ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum,
                                            access_location::device,
                                            access_mode::readwrite);
        ArrayHandle<Scalar4> d_alt_vel(m_pdata->getAltVelocities(),
                                       access_location::device,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                        access_location::device,
                                        access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        m_netvelo_tuner->begin();
        mpcd::gpu::get_net_velocity_rigid_body(d_linmom_accum.data,
                                               d_angmom_accum.data,
                                               d_alt_vel.data,
                                               d_velocity.data,
                                               d_orientation.data,
                                               d_inertia.data,
                                               d_rigid_center.data,
                                               m_rigid_bodies->getNLocal(),
                                               m_netvelo_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_netvelo_tuner->end();
        }
        {
        ArrayHandle<unsigned int> d_lookup_center(m_rigid_bodies->getLookupCenters(),
                                                  access_location::device,
                                                  access_mode::read);
        ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum,
                                            access_location::device,
                                            access_mode::read);
        ArrayHandle<Scalar4> d_alt_vel(m_pdata->getAltVelocities(),
                                       access_location::device,
                                       access_mode::read);

        ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);

        ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                        access_location::device,
                                        access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);

        m_applyrandvec_tuner->begin();
        mpcd::gpu::apply_thermalized_velocity_vectors(d_angmom_accum.data,
                                                      d_alt_vel.data,
                                                      d_postype.data,
                                                      d_velocity.data,
                                                      d_image.data,
                                                      d_lookup_center.data,
                                                      m_pdata->getGlobalBox(),
                                                      m_pdata->getN(),
                                                      m_applyrandvec_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_applyrandvec_tuner->end();
        }
    }

//! Begin process of applying collisions to rigid bodies (GPU version)
void mpcd::CollisionMethod::storeInitialEmbeddedGroupVelocitiesGPU(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_initial_vel(m_initial_velocity,
                                       access_location::device,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> d_embed_group(m_embed_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::read);
    m_store_tuner->begin();
    mpcd::gpu::store_initial_embedded_group_velocities(d_initial_vel.data,
                                                       d_velocity.data,
                                                       d_embed_group.data,
                                                       m_embed_group->getNumMembers(),
                                                       m_store_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_store_tuner->end();
    }

//! Accumulate momenta changes of constituent particles of rigid bodies (GPU version)
void mpcd::CollisionMethod::accumulateRigidBodyMomentaGPU(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

    ArrayHandle<unsigned int> d_lookup_center(m_rigid_bodies->getLookupCenters(),
                                              access_location::device,
                                              access_mode::read);
    ArrayHandle<unsigned int> d_embed_group(m_embed_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar4> d_initial_vel(m_initial_velocity,
                                       access_location::device,
                                       access_mode::read);

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum,
                                        access_location::device,
                                        access_mode::readwrite);
    m_accumulate_tuner->begin();
    mpcd::gpu::accumulate_rigid_body_momenta(d_linmom_accum.data,
                                             d_angmom_accum.data,
                                             d_initial_vel.data,
                                             d_embed_group.data,
                                             d_postype.data,
                                             d_velocity.data,
                                             d_image.data,
                                             d_lookup_center.data,
                                             global_box,
                                             m_embed_group->getNumMembers(),
                                             m_accumulate_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_accumulate_tuner->end();
    }

//! Finish process of applying collisions to rigid bodies (GPU version)
void mpcd::CollisionMethod::transferRigidBodyMomentaGPU(uint64_t timestep)
    {
    ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum, access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_rigid_center(m_rigid_bodies->getRigidCenters(),
                                             access_location::device,
                                             access_mode::read);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::device,
                                  access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::device,
                                   access_mode::read);

    m_transfer_tuner->begin();
    mpcd::gpu::transfer_rigid_body_momenta(d_linmom_accum.data,
                                           d_angmom_accum.data,
                                           d_velocity.data,
                                           d_orientation.data,
                                           d_angmom.data,
                                           d_inertia.data,
                                           d_rigid_center.data,
                                           m_rigid_bodies->getNLocal(),
                                           m_transfer_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_transfer_tuner->end();
    }

#endif // ENABLE_HIP

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * Using a multiple allows the collision method to be disabled and then reenabled later if the \a
 * timestep has already exceeded the \a m_next_timestep.
 */
bool mpcd::CollisionMethod::peekCollide(uint64_t timestep) const
    {
    if (timestep < m_next_timestep)
        return false;
    else
        return ((timestep - m_next_timestep) % m_period == 0);
    }

/*!
 * \param cur_timestep Current simulation timestep
 * \param period New period
 *
 * The collision method period is updated to \a period only if collision would occur at \a
 * cur_timestep. It is the caller's responsibility to ensure this condition is valid.
 */
void mpcd::CollisionMethod::setPeriod(uint64_t cur_timestep, uint64_t period)
    {
    if (!peekCollide(cur_timestep))
        {
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of original period"
            << std::endl;
        throw std::runtime_error(
            "Collision period can only be changed on multiple of original period");
        }

    // try to update the period
    const uint64_t old_period = m_period;
    m_period = period;

    // validate the new period, resetting to the old one before erroring out if it doesn't match
    if (!peekCollide(cur_timestep))
        {
        m_period = old_period;
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of new period"
            << std::endl;
        throw std::runtime_error("Collision period can only be changed on multiple of new period");
        }
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * \post The next timestep is also advanced to the next timestep the collision should occur after \a
 * timestep. If this behavior is not desired, then use peekCollide() instead.
 */
bool mpcd::CollisionMethod::shouldCollide(uint64_t timestep)
    {
    if (peekCollide(timestep))
        {
        m_next_timestep = timestep + m_period;
        return true;
        }
    else
        {
        return false;
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_CollisionMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::CollisionMethod, std::shared_ptr<mpcd::CollisionMethod>>(
        m,
        "CollisionMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t, uint64_t, int>())
        .def_property_readonly("embedded_particles",
                               [](const std::shared_ptr<mpcd::CollisionMethod> method)
                               {
                                   auto group = method->getEmbeddedGroup();
                                   return (group) ? group->getFilter()
                                                  : std::shared_ptr<hoomd::ParticleFilter>();
                               })
        .def("setEmbeddedGroup", &mpcd::CollisionMethod::setEmbeddedGroup)
        .def_property_readonly("period", &mpcd::CollisionMethod::getPeriod)
        .def_property("kT",
                      &mpcd::CollisionMethod::getTemperature,
                      &mpcd::CollisionMethod::setTemperature);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
