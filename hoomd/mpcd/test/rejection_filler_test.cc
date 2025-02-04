// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/RejectionVirtualParticleFiller.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.h"
#endif // ENABLE_HIP
#include "hoomd/mpcd/SphereGeometry.h"

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

template<class F>
void sphere_rejection_fill_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(20.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.type_mapping.push_back("B");
    snap->mpcd_data.position[0] = vec3<Scalar>(1, -2, 3);
    snap->mpcd_data.velocity[0] = vec3<Scalar>(123, 456, 789);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    auto pdata = sysdef->getMPCDParticleData();
    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);
    // we should have no virtual particle in the system at this point.
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);

    // create a spherical confinement of radius 5.0
    const Scalar r = 5.0;
    auto sphere = std::make_shared<const mpcd::SphereGeometry>(r, true);
    std::shared_ptr<Variant> kT = std::make_shared<VariantConstant>(1.5);
    std::shared_ptr<mpcd::RejectionVirtualParticleFiller<mpcd::SphereGeometry>> filler
        = std::make_shared<F>(sysdef, "B", 2.0, kT, sphere);
    filler->setCellList(cl);

    /*
     * Test basic filling up for this cell list
     */
    unsigned int Nfill_0(0);
    filler->fill(0);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        // ensure first particle did not get overwritten
        UP_ASSERT_CLOSE(h_pos.data[0].x, Scalar(1), tol_small);
        UP_ASSERT_CLOSE(h_pos.data[0].y, Scalar(-2), tol_small);
        UP_ASSERT_CLOSE(h_pos.data[0].z, Scalar(3), tol_small);
        UP_ASSERT_CLOSE(h_vel.data[0].x, Scalar(123), tol_small);
        UP_ASSERT_CLOSE(h_vel.data[0].y, Scalar(456), tol_small);
        UP_ASSERT_CLOSE(h_vel.data[0].z, Scalar(789), tol_small);
        UP_ASSERT_EQUAL(h_tag.data[0], 0);

        // check if the particles have been placed outside the confinement
        unsigned int N_out(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);
            // type should be set
            UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[i].w), 1);

            Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r * r)
                ++N_out;
            }
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        Nfill_0 = N_out;
        }

    /*
     * Fill the volume again, which should approximately double the number of virtual particles
     */
    filler->fill(1);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        // check if the particles have been placed outside the confinement
        unsigned int N_out(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);
            // type should be set
            UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[i].w), 1);

            Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r * r)
                ++N_out;
            }
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        UP_ASSERT_GREATER(N_out, Nfill_0);
        }

    /*
     * Test the average properties of the virtual particles.
     */
    // initialize variables for storing avg data
    Scalar N_avg(0);
    Scalar N_shell(0);
    Scalar3 vel_avg_net = make_scalar3(0, 0, 0);
    Scalar T_avg(0);
    // repeat filling 1000 times
    unsigned int num_samples(1000);
    for (unsigned int t = 0; t < num_samples; ++t)
        {
        pdata->removeVirtualParticles();
        filler->fill(2 + t);

        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        // local variables
        unsigned int N_out(0);
        Scalar temp(0);
        Scalar3 vel_avg = make_scalar3(0, 0, 0);

        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            const Scalar3 vel = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            const Scalar r2 = dot(pos, pos);
            if (r2 > r * r)
                {
                ++N_out;
                if (r2 < (r + 1) * (r + 1))
                    ++N_shell;
                }
            temp += dot(vel, vel);
            vel_avg += vel;
            }

        temp /= (3 * (N_out - 1));
        vel_avg_net += vel_avg / N_out;
        // Check whether all virtual particles are outside the sphere
        UP_ASSERT_EQUAL(N_out, pdata->getNVirtual());
        N_avg += N_out;
        T_avg += temp;
        }
    N_avg /= num_samples;
    N_shell /= num_samples;
    T_avg /= num_samples;
    vel_avg_net /= num_samples;

    /*
     * Expected number of virtual particles = int( density * volume outside sphere )
     * volume outside sphere = sim-box volume - sphere volume
     * N_exptd = int(density*(L^3 - 4*pi*r^3/3))
     *         = 14952
     */
    UP_ASSERT_CLOSE(N_avg, Scalar(14952.0), 2);

    // expected number in the shell is density * (4 pi/3)*((r+1)^3-r^3) = 1906
    UP_ASSERT_CLOSE(N_shell, Scalar(1906.0), 2);

    UP_ASSERT_SMALL(vel_avg_net.x, tol_small);
    UP_ASSERT_SMALL(vel_avg_net.y, tol_small);
    UP_ASSERT_SMALL(vel_avg_net.z, tol_small);
    UP_ASSERT_CLOSE(T_avg, 1.5, tol_small);
    }

UP_TEST(sphere_rejection_fill_basic)
    {
    sphere_rejection_fill_basic_test<mpcd::RejectionVirtualParticleFiller<mpcd::SphereGeometry>>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
UP_TEST(sphere_rejection_fill_basic_gpu)
    {
    sphere_rejection_fill_basic_test<mpcd::RejectionVirtualParticleFillerGPU<mpcd::SphereGeometry>>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
