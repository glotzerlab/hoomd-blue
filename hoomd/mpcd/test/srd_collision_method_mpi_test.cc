// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/Communicator.h"
#include "hoomd/mpcd/SRDCollisionMethod.h"
#include "utils.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CommunicatorGPU.h"
#include "hoomd/mpcd/SRDCollisionMethodGPU.h"
#endif // ENABLE_HIP

#include "hoomd/Communicator.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/filter/ParticleFilterAll.h"
#include "hoomd/mpcd/Communicator.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for basic setup and functionality of the SRD collision method
template<class CM>
void srd_collision_method_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(2.0);
    snap->particle_data.type_mapping.push_back("A");
    // 4 particle system
    snap->mpcd_data.resize(4);
    snap->mpcd_data.type_mapping.push_back("A");

    snap->mpcd_data.position[0] = vec3<Scalar>(-0.1, -0.1, -0.1);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.1, -0.1, -0.1);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.1, 0.1, 0.1);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.1, 0.1, 0.1);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    // initialize system and collision method
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<mpcd::ParticleData> pdata_4 = sysdef->getMPCDParticleData();
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);
    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(sysdef, 0, 2, 1, 130.);
    collide->setCellList(cl);
    AllThermoRequest thermo_req(cl);

    // stash initial velocities for reference
    std::vector<Scalar3> orig_vel(pdata_4->getN());
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < pdata_4->getN(); ++i)
            {
            orig_vel[i] = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            }
        }
    // Save original momentum for comparison as well
    const Scalar3 orig_mom = make_scalar3(7.0, 0.0, -2.0);
    const Scalar orig_energy = 36.5;
    const Scalar orig_temp = 9.75;

    UP_ASSERT(!collide->peekCollide(0));
    collide->collide(0);
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < pdata_4->getN(); ++i)
            {
            CHECK_CLOSE(h_vel.data[i].x, orig_vel[i].x, tol_small);
            CHECK_CLOSE(h_vel.data[i].y, orig_vel[i].y, tol_small);
            CHECK_CLOSE(h_vel.data[i].z, orig_vel[i].z, tol_small);
            }
        }

        {
        // check net properties of cells, which should match our inputs
        cl->compute(0);
        const Scalar3 mom = cl->getNetMomentum();
        CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
        CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
        CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

        const Scalar energy = cl->getNetEnergy();
        CHECK_CLOSE(energy, orig_energy, tol_small);

        const Scalar temp = cl->getTemperature();
        CHECK_CLOSE(temp, orig_temp, tol_small);
        }

    UP_ASSERT(collide->peekCollide(1));
    collide->collide(1);
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<double3> h_rotvec(collide->getRotationVectors(),
                                      access_location::host,
                                      access_mode::read);
        ArrayHandle<double4> h_cell_vel(cl->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        for (unsigned int i = 0; i < pdata_4->getN(); ++i)
            {
            // all rotation vectors should be unit norm
            const unsigned int cell = __scalar_as_int(h_vel.data[i].w);
            const Scalar3 avg_vel = make_scalar3(h_cell_vel.data[cell].x,
                                                 h_cell_vel.data[cell].y,
                                                 h_cell_vel.data[cell].z);
            const Scalar3 rot_vec
                = make_scalar3(h_rotvec.data[cell].x, h_rotvec.data[cell].y, h_rotvec.data[cell].z);
            CHECK_CLOSE(dot(rot_vec, rot_vec), 1.0, tol_small);

            // norm of velocity relative to average is unchanged by rotation
            const Scalar3 vel = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            const Scalar norm = dot(vel - avg_vel, vel - avg_vel);

            if (norm < 1)
                {
                CHECK_CLOSE(norm, 0.25, tol_small);
                }
            else
                {
                CHECK_CLOSE(norm, 3.0 * 3.0 + 2.0 * 2.0 + 4.0 * 4.0, tol_small);
                }

            // compute the angle between the two vectors relative to the cell average velocity
            // which should be the same before and after rotation
            Scalar3 v1 = vel - avg_vel;
            Scalar3 v2 = orig_vel[i] - avg_vel;
            CHECK_CLOSE(dot(v1, rot_vec), dot(v2, rot_vec), tol_small);

            // check the rotation angle of the velocities by projecting the velocities orthogonally
            // into the plane that the rotation vector is the normal of. Given the plane is through
            // the origin with normal n, the projection of v is: q = v - dot(v,n) * n
            Scalar3 q1 = v1 - dot(v1, rot_vec) * rot_vec;
            Scalar3 q2 = v2 - dot(v2, rot_vec) * rot_vec;
            Scalar cos_angle = dot(q1, q2) / (sqrt(dot(q1, q1)) * sqrt(dot(q2, q2)));
            CHECK_CLOSE(cos_angle, slow::cos(collide->getRotationAngle() * M_PI / 180.), tol_small);
            }
        }

    // recompute net properties, and make sure they are still the same
    cl->compute(2);
    Scalar3 mom = cl->getNetMomentum();
    CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
    CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
    CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

    Scalar energy = cl->getNetEnergy();
    CHECK_CLOSE(energy, orig_energy, tol_small);

    Scalar temp = cl->getTemperature();
    CHECK_CLOSE(temp, orig_temp, tol_small);

        // stash current velocities for reference
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < pdata_4->getN(); ++i)
            {
            orig_vel[i] = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            }
        }
    // shift particles so that they are all in the same cell to force communication
    const Scalar3 shift = (Scalar(0.5) / 3) * make_scalar3(1, 1, 1);
    cl->setGridShift(-shift);
    cl->compute(3);

    // update the temperature
    const Scalar orig_temp_shift = cl->getTemperature();

    UP_ASSERT(collide->peekCollide(3));
    collide->collide(3);
        {
        const unsigned int num_ghosts = cl->getNMPCDGhosts();
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<double3> h_rotvec(collide->getRotationVectors(),
                                      access_location::host,
                                      access_mode::read);
        ArrayHandle<double4> h_cell_vel(cl->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);

        // communicate the rot_vec that is relevant to the particles
        Scalar3 rot_vec;
        Scalar3 avg_vel;
        if (num_ghosts)
            {
            rot_vec = make_scalar3(h_rotvec.data[0].x, h_rotvec.data[0].y, h_rotvec.data[0].z);
            avg_vel
                = make_scalar3(h_cell_vel.data[0].x, h_cell_vel.data[0].y, h_cell_vel.data[0].z);
            }
        else
            {
            rot_vec = make_scalar3(0, 0, 0);
            avg_vel = make_scalar3(0, 0, 0);
            }
        MPI_Allreduce(MPI_IN_PLACE,
                      &rot_vec,
                      int(sizeof(Scalar3)),
                      MPI_BYTE,
                      MPI_SUM,
                      exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE,
                      &avg_vel,
                      int(sizeof(Scalar3)),
                      MPI_BYTE,
                      MPI_SUM,
                      exec_conf->getMPICommunicator());

        for (unsigned int i = 0; i < pdata_4->getN(); ++i)
            {
            // all rotation vectors should be unit norm
            CHECK_CLOSE(dot(rot_vec, rot_vec), 1.0, tol_small);

            const Scalar3 vel = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

            // ensure that the velocity is different
            UP_ASSERT(vel.x != orig_vel[i].x);
            UP_ASSERT(vel.y != orig_vel[i].y);
            UP_ASSERT(vel.z != orig_vel[i].z);

            // compute the angle between the two vectors relative to the cell average velocity
            // which should be the same before and after rotation
            Scalar3 v1 = vel - avg_vel;
            Scalar3 v2 = orig_vel[i] - avg_vel;
            CHECK_CLOSE(dot(v1, rot_vec), dot(v2, rot_vec), tol_small);

            // check the rotation angle of the velocities by projecting the velocities orthogonally
            // into the plane that the rotation vector is the normal of. Given the plane is through
            // the origin with normal n, the projection of v is: q = v - dot(v,n) * n
            Scalar3 q1 = v1 - dot(v1, rot_vec) * rot_vec;
            Scalar3 q2 = v2 - dot(v2, rot_vec) * rot_vec;
            Scalar cos_angle = dot(q1, q2) / (sqrt(dot(q1, q1)) * sqrt(dot(q2, q2)));
            CHECK_CLOSE(cos_angle, slow::cos(collide->getRotationAngle() * M_PI / 180.), tol_small);
            }
        }

    // check net properties of momentum and energy still match
    mom = cl->getNetMomentum();
    CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
    CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
    CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

    energy = cl->getNetEnergy();
    CHECK_CLOSE(energy, orig_energy, tol_small);

    temp = cl->getTemperature();
    CHECK_CLOSE(temp, orig_temp_shift, tol_small);
    }

//! Test that embedding a particle keeps conservation
/*!
 * Because of the way the rotations occur, we only need to check that an update
 * is made properly, and that the normal properties are conserved.
 */
template<class CM>
void srd_collision_method_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(2.0);
    snap->particle_data.type_mapping.push_back("A");
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.resize(2);
        pdata_snap.pos[0] = vec3<Scalar>(-0.1, -0.1, -0.1);
        pdata_snap.vel[0] = vec3<Scalar>(1.0, 2.0, 3.0);
        pdata_snap.mass[0] = 2.0;
        pdata_snap.pos[1] = vec3<Scalar>(0.1, 0.1, 0.1);
        pdata_snap.vel[1] = vec3<Scalar>(-1.0, -1.0, -2.0);
        pdata_snap.mass[1] = 2.0;
        }

    // 4 particle system
    snap->mpcd_data.resize(4);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.1, -0.1, -0.1);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.1, -0.1, -0.1);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.1, 0.1, 0.1);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.1, 0.1, 0.1);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);

    // initialize system and collision method
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<mpcd::ParticleData> pdata_4 = sysdef->getMPCDParticleData();
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);
    std::shared_ptr<mpcd::SRDCollisionMethod> collide
        = std::make_shared<CM>(sysdef, 0, 1, -1, 130.);
    collide->setCellList(cl);
    AllThermoRequest thermo_req(cl);

    // embed the particle group into the mpcd system
    std::shared_ptr<ParticleFilter> selector_one(new ParticleFilterAll());
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_one));
    collide->setEmbeddedGroup(group_all);

    // stash current embedded velocities for reference
    std::vector<Scalar3> orig_vel(sysdef->getParticleData()->getN());
        {
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < sysdef->getParticleData()->getN(); ++i)
            {
            orig_vel[i] = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            }
        }

    // Save original momentum for comparison as well
    cl->compute(0);
    const Scalar orig_energy = cl->getNetEnergy();
    const Scalar orig_temp = cl->getTemperature();
    const Scalar3 orig_mom = cl->getNetMomentum();
    collide->collide(0);
        {
        // velocity should be different now, but the mass should stay the same
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < sysdef->getParticleData()->getN(); ++i)
            {
            UP_ASSERT(h_vel.data[i].x != orig_vel[i].x);
            UP_ASSERT(h_vel.data[i].y != orig_vel[i].y);
            UP_ASSERT(h_vel.data[i].z != orig_vel[i].z);
            CHECK_CLOSE(h_vel.data[i].w, 2.0, tol_small);
            }
        }

    // compute properties after rotation
    cl->compute(1);
    Scalar energy = cl->getNetEnergy();
    Scalar temp = cl->getTemperature();
    Scalar3 mom = cl->getNetMomentum();

    // energy (temperature) and momentum should be conserved after a collision
    CHECK_CLOSE(orig_energy, energy, tol_small);
    CHECK_CLOSE(orig_temp, temp, tol_small);
    CHECK_CLOSE(orig_mom.x, mom.x, tol_small);
    CHECK_CLOSE(orig_mom.y, mom.y, tol_small);
    CHECK_CLOSE(orig_mom.z, mom.z, tol_small);

        // stash current embedded velocities for reference
        {
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < sysdef->getParticleData()->getN(); ++i)
            {
            orig_vel[i] = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            }
        }

    // shift particles so that they are all in the same cell to force communication
    const Scalar3 shift = (Scalar(0.5) / 3) * make_scalar3(1, 1, 1);
    cl->setGridShift(-shift);
    cl->compute(2);

    // update the temperature
    const Scalar orig_temp_shift = cl->getTemperature();

    collide->collide(1);
        {
        // velocity should be different now, but the mass should stay the same
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = 0; i < sysdef->getParticleData()->getN(); ++i)
            {
            UP_ASSERT(h_vel.data[i].x != orig_vel[i].x);
            UP_ASSERT(h_vel.data[i].y != orig_vel[i].y);
            UP_ASSERT(h_vel.data[i].z != orig_vel[i].z);
            CHECK_CLOSE(h_vel.data[i].w, 2.0, tol_small);
            }
        }

    // check net properties of momentum and energy still match
    mom = cl->getNetMomentum();
    CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
    CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
    CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

    energy = cl->getNetEnergy();
    CHECK_CLOSE(energy, orig_energy, tol_small);

    temp = cl->getTemperature();
    CHECK_CLOSE(temp, orig_temp_shift, tol_small);
    }

//! Test that the thermostat can generate the correct temperature
template<class CM>
void srd_collision_method_thermostat_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    auto box = std::make_shared<BoxDim>(10.0);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, box->getL(), 2, 2, 2));
    auto sysdef = std::make_shared<hoomd::SystemDefinition>(0,
                                                            box,
                                                            1,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            exec_conf,
                                                            decomposition);
    auto pdata
        = std::make_shared<mpcd::ParticleData>(10000, box, 1.0, 42, 3, exec_conf, decomposition);
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setMPCDParticleData(pdata);
    sysdef->setCommunicator(pdata_comm);

    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);
    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(sysdef, 0, 1, -1, 827);
    collide->setCellList(cl);
    AllThermoRequest thermo_req(cl);

    // timestep counter and number of samples to make
    uint64_t timestep = 0;
    const unsigned int N = 1000;

        // set the temperature to 2.0 and check
        {
        std::shared_ptr<Variant> T = std::make_shared<VariantConstant>(2.0);
        collide->setTemperature(T);
        double mean(0.0);
        for (unsigned int i = 0; i < N; ++i)
            {
            cl->compute(timestep);
            mean += cl->getTemperature();
            collide->collide(timestep++);
            }
        mean /= N;
        CHECK_CLOSE(mean, 2.0, tol);
        }

        // change the temperature and check again
        {
        std::shared_ptr<Variant> T = std::make_shared<VariantConstant>(4.0);
        collide->setTemperature(T);
        double mean(0.0);
        for (unsigned int i = 0; i < N; ++i)
            {
            cl->compute(timestep);
            mean += cl->getTemperature();
            collide->collide(timestep++);
            }
        mean /= N;
        CHECK_CLOSE(mean, 4.0, tol);
        }
    }

//! basic test case for MPCD SRDCollisionMethod class
UP_TEST(srd_collision_method_basic)
    {
    srd_collision_method_basic_test<mpcd::SRDCollisionMethod>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
//! test embedding of particles into the MPCD SRDCollisionMethod class
// UP_TEST(srd_collision_method_embed)
//     {
//     srd_collision_method_embed_test<mpcd::SRDCollisionMethod>(
//         std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
//     }
UP_TEST(srd_collision_method_thermostat)
    {
    srd_collision_method_thermostat_test<mpcd::SRDCollisionMethod>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
//! basic test case for MPCD SRDCollisionMethodGPU class
UP_TEST(srd_collision_method_basic_gpu)
    {
    srd_collision_method_basic_test<mpcd::SRDCollisionMethodGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
//! test embedding of particles into the MPCD SRDCollisionMethodGPU class
// UP_TEST(srd_collision_method_embed_gpu)
//     {
//     srd_collision_method_embed_test<mpcd::SRDCollisionMethodGPU>(
//         std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
//     }
UP_TEST(srd_collision_method_thermostat_gpu)
    {
    srd_collision_method_thermostat_test<mpcd::SRDCollisionMethodGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
