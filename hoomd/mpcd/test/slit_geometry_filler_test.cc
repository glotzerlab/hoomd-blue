// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/SlitGeometryFiller.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/SlitGeometryFillerGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

template<class F> void slit_fill_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(20.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(1, -2, 3);
    snap->mpcd_data.velocity[0] = vec3<Scalar>(123, 456, 789);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    auto pdata = sysdef->getMPCDParticleData();
    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    cl->setCellSize(2.0);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);

    // create slit channel with half width 5
    auto slit = std::make_shared<const mpcd::detail::SlitGeometry>(5.0,
                                                                   1.0,
                                                                   mpcd::detail::boundary::no_slip);
    std::shared_ptr<Variant> kT = std::make_shared<VariantConstant>(1.5);
    std::shared_ptr<mpcd::SlitGeometryFiller> filler
        = std::make_shared<F>(sysdef, 2.0, 1, kT, slit);
    filler->setCellList(cl);

    /*
     * Test basic filling up for this cell list
     */
    filler->fill(0);
    // volume to fill is from 5->7 (2) on + side, with cross section of 20^2, mirrored on bottom
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2 * (2 * 20 * 20) * 2);
        // count that particles have been placed on the right sides
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        // ensure first particle did not get overwritten
        CHECK_CLOSE(h_pos.data[0].x, 1, tol_small);
        CHECK_CLOSE(h_pos.data[0].y, -2, tol_small);
        CHECK_CLOSE(h_pos.data[0].z, 3, tol_small);
        CHECK_CLOSE(h_vel.data[0].x, 123, tol_small);
        CHECK_CLOSE(h_vel.data[0].y, 456, tol_small);
        CHECK_CLOSE(h_vel.data[0].z, 789, tol_small);
        UP_ASSERT_EQUAL(h_tag.data[0], 0);

        unsigned int N_lo(0), N_hi(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);
            // type should be set
            UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[i].w), 1);

            const Scalar z = h_pos.data[i].z;
            if (z < Scalar(-5.0))
                ++N_lo;
            else if (z >= Scalar(5.0))
                ++N_hi;
            }
        UP_ASSERT_EQUAL(N_lo, 2 * (2 * 20 * 20));
        UP_ASSERT_EQUAL(N_hi, 2 * (2 * 20 * 20));
        }

    /*
     * Fill the volume again, which should double the number of virtual particles
     */
    filler->fill(1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2 * 2 * (2 * 20 * 20) * 2);
        // count that particles have been placed on the right sides
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);

        unsigned int N_lo(0), N_hi(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            // tag should equal index on one rank with one filler
            UP_ASSERT_EQUAL(h_tag.data[i], i);

            const Scalar z = h_pos.data[i].z;
            if (z < Scalar(-5.0))
                ++N_lo;
            else if (z >= Scalar(5.0))
                ++N_hi;
            }
        UP_ASSERT_EQUAL(N_lo, 2 * 2 * (2 * 20 * 20));
        UP_ASSERT_EQUAL(N_hi, 2 * 2 * (2 * 20 * 20));
        }

    /*
     * Change the cell size so that we lie exactly on a boundary.
     */
    pdata->removeVirtualParticles();
    cl->setCellSize(1.0);
    filler->fill(2);
    // volume to fill is now from 5->5.5 on + side, same other parameters
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2 * (20 * 20 / 2) * 2);
        // count that particles have been placed on the right sides
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        unsigned int N_lo(0), N_hi(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar z = h_pos.data[i].z;
            if (z < Scalar(-5.0))
                ++N_lo;
            else if (z >= Scalar(5.0))
                ++N_hi;
            }
        UP_ASSERT_EQUAL(N_lo, 2 * (20 * 20 / 2));
        UP_ASSERT_EQUAL(N_hi, 2 * (20 * 20 / 2));
        }

    /*
     * Test the average properties of the virtual particles.
     */
    cl->setCellSize(2.0);
    unsigned int N_lo(0), N_hi(0);
    Scalar3 v_lo = make_scalar3(0, 0, 0);
    Scalar3 v_hi = make_scalar3(0, 0, 0);
    Scalar T_avg(0);
    for (unsigned int t = 0; t < 500; ++t)
        {
        pdata->removeVirtualParticles();
        filler->fill(3 + t);

        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar z = h_pos.data[i].z;
            const Scalar4 vel_cell = h_vel.data[i];
            const Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
            if (z < Scalar(-5.0))
                {
                v_lo += vel;
                T_avg += dot(vel - make_scalar3(-1.0, 0, 0), vel - make_scalar3(-1.0, 0, 0));
                ++N_lo;
                }
            else if (z >= Scalar(5.0))
                {
                v_hi += vel;
                T_avg += dot(vel - make_scalar3(1.0, 0, 0), vel - make_scalar3(1.0, 0, 0));
                ++N_hi;
                }
            }
        }
    // make averages
    UP_ASSERT_EQUAL(N_lo, 500 * 2 * (2 * 20 * 20));
    UP_ASSERT_EQUAL(N_hi, 500 * 2 * (2 * 20 * 20));
    v_lo /= N_lo;
    v_hi /= N_hi;
    T_avg /= (3 * (N_lo + N_hi - 1));

    CHECK_CLOSE(v_lo.x, -1.0, tol);
    CHECK_SMALL(v_lo.y, tol);
    CHECK_SMALL(v_lo.z, tol);
    CHECK_CLOSE(v_hi.x, 1.0, tol);
    CHECK_SMALL(v_hi.y, tol);
    CHECK_SMALL(v_hi.z, tol);
    CHECK_CLOSE(T_avg, 1.5, tol);
    }

UP_TEST(slit_fill_basic)
    {
    slit_fill_basic_test<mpcd::SlitGeometryFiller>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
UP_TEST(slit_fill_basic_gpu)
    {
    slit_fill_basic_test<mpcd::SlitGeometryFillerGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
