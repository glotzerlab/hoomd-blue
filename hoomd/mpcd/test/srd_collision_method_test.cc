// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "utils.h"
#include "hoomd/mpcd/SRDCollisionMethod.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/SRDCollisionMethodGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

//! Test for basic setup and functionality of the SRD collision method
template<class CM>
void srd_collision_method_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // 4 particle system
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
    std::vector<Scalar3> orig_vel;
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(4);

        mpcd_snap->position[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[1] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
        mpcd_snap->position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);

        orig_vel.resize(mpcd_snap->size);
        // stash initial velocities for reference
        for (unsigned int i=0; i < mpcd_snap->size; ++i)
            {
            orig_vel[i] = make_scalar3(mpcd_snap->velocity[i].x, mpcd_snap->velocity[i].y, mpcd_snap->velocity[i].z);
            }
        }
    // Save original momentum for comparison as well
    const Scalar3 orig_mom = make_scalar3(7.0, 0.0, -2.0);
    const Scalar orig_energy = 36.5;
    const Scalar orig_temp = 9.75;

    // initialize system and collision method
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    std::shared_ptr<mpcd::ParticleData> pdata_4 = mpcd_sys->getParticleData();

    // create a thermo, and use it to check the current
    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);

    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 2, 1, 42, thermo);
    collide->enableGridShifting(false);
    // 130 degrees, forces all components of the rotation matrix to act
    const double rot_angle = 2.2689280275926285;
    collide->setRotationAngle(rot_angle);

    UP_ASSERT(!collide->peekCollide(0));
    collide->collide(0);
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(), access_location::host, access_mode::read);
        for (unsigned int i=0; i < pdata_4->getN(); ++i)
            {
            CHECK_CLOSE(h_vel.data[i].x, orig_vel[i].x, tol_small);
            CHECK_CLOSE(h_vel.data[i].y, orig_vel[i].y, tol_small);
            CHECK_CLOSE(h_vel.data[i].z, orig_vel[i].z, tol_small);
            }

        // check net properties of cells, which should match our inputs
        thermo->compute(0);
        const Scalar3 mom = thermo->getNetMomentum();
        CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
        CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
        CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

        const Scalar energy = thermo->getNetEnergy();
        CHECK_CLOSE(energy, orig_energy, tol_small);

        const Scalar temp = thermo->getTemperature();
        CHECK_CLOSE(temp, orig_temp, tol_small);
        }

    UP_ASSERT(collide->peekCollide(1));
    collide->collide(1);
        {
        ArrayHandle<Scalar4> h_vel(pdata_4->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<double3> h_rotvec(collide->getRotationVectors(), access_location::host, access_mode::read);

        for (unsigned int i=0; i < pdata_4->getN(); ++i)
            {
            Scalar3 avg_vel;
            if (i < 2)
                {
                avg_vel.x = 1.5;
                avg_vel.y = 0.0;
                avg_vel.z = 0.0;
                }
            else
                {
                avg_vel.x = 2.0;
                avg_vel.y = 0.0;
                avg_vel.z = -1.0;
                }

            // all rotation vectors should be unit norm
            const unsigned int cell = __scalar_as_int(h_vel.data[i].w);
            const Scalar3 rot_vec = make_scalar3(h_rotvec.data[cell].x, h_rotvec.data[cell].y, h_rotvec.data[cell].z);
            CHECK_CLOSE(dot(rot_vec,rot_vec), 1.0, tol_small);

            // norm of velocity relative to average is unchanged by rotation
            const Scalar3 vel = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            const Scalar norm = dot(vel - avg_vel, vel - avg_vel);
            if (i < 2)
                {
                CHECK_CLOSE(norm, 0.25, tol_small);
                }
            else
                {
                CHECK_CLOSE(norm, 3.0*3.0 + 2.0*2.0 + 4.0*4.0, tol_small);
                }

            // compute the angle between the two vectors relative to the cell average velocity
            // which should be the same before and after rotation
            Scalar3 v1 = vel - avg_vel;
            Scalar3 v2 = orig_vel[i] - avg_vel;
            CHECK_CLOSE(dot(v1, rot_vec), dot(v2, rot_vec), tol_small);

            // check the rotation angle of the velocities by projecting the velocities orthogonally into the plane
            // that the rotation vector is the normal of. Given the plane is through the origin with normal n,
            // the projection of v is: q = v - dot(v,n) * n
            Scalar3 q1 = v1 - dot(v1, rot_vec)*rot_vec;
            Scalar3 q2 = v2 - dot(v2, rot_vec)*rot_vec;
            Scalar cos_angle = dot(q1, q2)/(sqrt(dot(q1,q1))*sqrt(dot(q2,q2)));
            CHECK_CLOSE(cos_angle, slow::cos(rot_angle), tol_small);
            }
        }

    // recompute net properties, and make sure they are still the same
    thermo->compute(2);
    const Scalar3 mom = thermo->getNetMomentum();
    CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
    CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
    CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

    const Scalar energy = thermo->getNetEnergy();
    CHECK_CLOSE(energy, orig_energy, tol_small);

    const Scalar temp = thermo->getTemperature();
    CHECK_CLOSE(temp, orig_temp, tol_small);
    }

//! Test that rotation vectors are drawn with the correct distribution on the unit sphere
template<class CM>
void srd_collision_method_rotvec_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // initialize a big empty system
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(50.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    // mpcd system, thermo, srd collision method
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 1, -1, 42, thermo);

    // initialize the histograms
    const double mpcd_pi = 3.141592653589793;
    const unsigned int nbins = 25;
    const double dphi = mpcd_pi/static_cast<double>(nbins); // [0, pi)
    const double dtheta = 2.0*mpcd_pi/static_cast<double>(nbins); // [0, 2pi)
    std::vector<unsigned int> fphi(nbins, 0), ftheta(nbins, 0);

    // size cell list and count number of cells
    mpcd_sys->getCellList()->computeDimensions();
    const unsigned int ncells = mpcd_sys->getCellList()->getNCells();

    const unsigned int nsamples = 10;
    for (unsigned int sample_i = 0; sample_i < nsamples; ++sample_i)
        {
        collide->collide(sample_i);
        ArrayHandle<double3> h_rotvec(collide->getRotationVectors(), access_location::host, access_mode::read);

        for (unsigned int cell_i = 0; cell_i < ncells; ++cell_i)
            {
            const double3 rotvec = h_rotvec.data[cell_i];
            const double r = slow::sqrt(rotvec.x * rotvec.x + rotvec.y*rotvec.y +rotvec.z*rotvec.z);
            CHECK_CLOSE(r, 1.0, tol_small);

            // z = r cos(phi)
            const double phi = std::acos(rotvec.z / r);
            const unsigned int phi_bin = static_cast<unsigned int>(phi/dphi);
            UP_ASSERT(phi_bin < nbins);
            fphi[phi_bin] += 1;

            // bin theta
            double theta = std::atan2(rotvec.y, rotvec.x);
            if (theta < 0.0)
                {
                theta += 2.0*mpcd_pi;
                }
            const unsigned int theta_bin = static_cast<unsigned int>(theta/dtheta);
            UP_ASSERT(theta_bin < nbins);
            ftheta[theta_bin] += 1;
            }
        }

    /* When drawing uniformly on a sphere, the pdf should satisfy
     * \f$ \int f(\omega) d\omega = 1 = \int d\theta d\phi f(\theta, \phi) \f$.
     * The proper distribution satisfying this is:
     * \f$ f(\theta, \phi) = sin(\phi) / 4\pi \f$
     * because \f$ d\omega = sin(\phi) d\theta d\phi \f$.
     *
     * The marginal probability of each spherical coordinate is then
     *
     * \f$ f(\theta) = 1/2\pi \f$
     * \f$ f(\phi) = sin(\phi)/2 \f$
     *
     * Verify this with a loose (2%) tolerance since there will just be some random noise as well.
     */
    for (unsigned int bin_i = 0; bin_i < nbins; ++bin_i)
        {
        const double ftheta_i = static_cast<double>(ftheta[bin_i]) / (dtheta * static_cast<double>(nsamples*ncells));
        const double fphi_i = static_cast<double>(fphi[bin_i]) / (dphi * static_cast<double>(nsamples*ncells));
        CHECK_CLOSE(ftheta_i, 1.0/(2.0*mpcd_pi), 2.0);
        CHECK_CLOSE(fphi_i, 0.5*sin(dphi*(0.5+static_cast<double>(bin_i))), 2.0);
        }
    }

//! Test that embedding a particle keeps conservation
/*!
 * Because of the way the rotations occur, we only need to check that an update
 * is made properly, and that the normal properties are conserved.
 */
template<class CM>
void srd_collision_method_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.resize(1);
        pdata_snap.pos[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        pdata_snap.vel[0] = vec3<Scalar>(1.0, 2.0, 3.0);
        pdata_snap.mass[0] = 2.0;
        }
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // 4 particle system
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(4);

        mpcd_snap->position[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[1] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
        mpcd_snap->position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);
        }
    // initialize system and collision method
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    std::shared_ptr<mpcd::ParticleData> pdata_4 = mpcd_sys->getParticleData();

    // create a thermo, and use it to check the current
    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);

    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 1, -1, 827, thermo);
    collide->enableGridShifting(false);
    // 130 degrees, forces all components of the rotation matrix to act
    const double rot_angle = 2.2689280275926285;
    collide->setRotationAngle(rot_angle);

    // embed the particle group into the mpcd system
    std::shared_ptr<ParticleSelector> selector_one(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_one));
    collide->setEmbeddedGroup(group_all);

    // Save original momentum for comparison as well
    thermo->compute(0);
    const Scalar orig_energy = thermo->getNetEnergy();
    const Scalar orig_temp = thermo->getTemperature();
    const Scalar3 orig_mom = thermo->getNetMomentum();
    collide->collide(0);
        {
        // velocity should be different now, but the mass should stay the same
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(), access_location::host, access_mode::read);
        UP_ASSERT(h_vel.data[0].x != 1.0);
        UP_ASSERT(h_vel.data[0].y != 2.0);
        UP_ASSERT(h_vel.data[0].z != 3.0);
        CHECK_CLOSE(h_vel.data[0].w, 2.0, tol_small);
        }

    // compute properties after rotation
    thermo->compute(1);
    Scalar energy = thermo->getNetEnergy();
    Scalar temp = thermo->getTemperature();
    Scalar3 mom = thermo->getNetMomentum();

    // energy (temperature) and momentum should be conserved after a collision
    CHECK_CLOSE(orig_energy, energy, tol_small);
    CHECK_CLOSE(orig_temp, temp, tol_small);
    CHECK_CLOSE(orig_mom.x, mom.x, tol_small);
    CHECK_CLOSE(orig_mom.y, mom.y, tol_small);
    CHECK_CLOSE(orig_mom.z, mom.z, tol_small);
    }

//! Test that the thermostat can generate the correct temperature
template<class CM>
void srd_collision_method_thermostat_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const BoxDim box(10.0);
    auto sysdef = std::make_shared<::SystemDefinition>(0, box, 1, 0, 0, 0, 0, exec_conf);
    auto pdata = std::make_shared<mpcd::ParticleData>(10000, box, 1.0, 42, 3, exec_conf);
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(sysdef, pdata);

    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    std::shared_ptr<mpcd::SRDCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 1, -1, 827, thermo);

    // timestep counter and number of samples to make
    unsigned int timestep = 0;
    const unsigned int N = 1000;

    // set the temperature to 2.0 and check
        {
        std::shared_ptr<::Variant> T = std::make_shared<::VariantConst>(2.0);
        collide->setTemperature(T);
        double mean(0.0);
        for (unsigned int i=0; i < N; ++i)
            {
            collide->collide(timestep++);
            mean += thermo->getTemperature();
            }
        mean /= N;
        CHECK_CLOSE(mean, 2.0, tol);
        }

    // change the temperature and check again
        {
        std::shared_ptr<::Variant> T = std::make_shared<::VariantConst>(4.0);
        collide->setTemperature(T);
        double mean(0.0);
        for (unsigned int i=0; i < N; ++i)
            {
            collide->collide(timestep++);
            mean += thermo->getTemperature();
            }
        mean /= N;
        CHECK_CLOSE(mean, 4.0, tol);
        }
    }

//! basic test case for MPCD SRDCollisionMethod class
UP_TEST( srd_collision_method_basic )
    {
    srd_collision_method_basic_test<mpcd::SRDCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
//! test distribution of random rotation vectors for the MPCD SRDCollisionMethod class
UP_TEST( srd_collision_method_rotvec )
    {
    srd_collision_method_rotvec_test<mpcd::SRDCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
//! test embedding of particles into the MPCD SRDCollisionMethod class
UP_TEST( srd_collision_method_embed )
    {
    srd_collision_method_embed_test<mpcd::SRDCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
UP_TEST( srd_collision_method_thermostat )
    {
    srd_collision_method_thermostat_test<mpcd::SRDCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_CUDA
//! basic test case for MPCD SRDCollisionMethodGPU class
UP_TEST( srd_collision_method_basic_gpu )
    {
    srd_collision_method_basic_test<mpcd::SRDCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
//! test distribution of random rotation vectors for the MPCD SRDCollisionMethodGPU class
UP_TEST( srd_collision_method_rotvec_gpu )
    {
    srd_collision_method_rotvec_test<mpcd::SRDCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
//! test embedding of particles into the MPCD SRDCollisionMethodGPU class
UP_TEST( srd_collision_method_embed_gpu )
    {
    srd_collision_method_embed_test<mpcd::SRDCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
UP_TEST( srd_collision_method_thermostat_gpu )
    {
    srd_collision_method_thermostat_test<mpcd::SRDCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_CUDA
