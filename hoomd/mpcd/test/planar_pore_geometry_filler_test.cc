// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/PlanarPoreGeometryFiller.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/PlanarPoreGeometryFillerGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

template<class F>
void planar_pore_fill_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
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
    auto cl = std::make_shared<mpcd::CellList>(sysdef, 2.0, false);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);

    // create slit pore channel with half width 5, half length 8
    auto slit = std::make_shared<const mpcd::PlanarPoreGeometry>(10.0, 16.0, true);
    // fill density 2, temperature 1.5
    std::shared_ptr<Variant> kT = std::make_shared<VariantConstant>(1.5);
    std::shared_ptr<mpcd::PlanarPoreGeometryFiller> filler
        = std::make_shared<F>(sysdef, "B", 2.0, kT, slit);
    filler->setCellList(cl);

    /*
     * Test basic filling up for this cell list
     *
     * The fill volume is a U-shape |___|. The width of the sides is 1, since they align with the
     * grid. The width of the bottom is 2, since the wall slices halfway through the cell. The area
     * is the sum of the 3 rectangles, and the volume is multiplied by the box size.
     */
    filler->fill(0);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2 * 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        // count that particles have been placed on the right sides, and in right spaces
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

            const Scalar4 r = h_pos.data[i];
            if (r.x >= Scalar(-8.0) && r.x <= Scalar(8.0))
                {
                if (r.y < Scalar(-5.0))
                    ++N_lo;
                else if (r.y >= Scalar(5.0))
                    ++N_hi;
                }
            }
        UP_ASSERT_EQUAL(N_lo, 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        UP_ASSERT_EQUAL(N_hi, 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        }

    /*
     * Fill the volume again with double the density, which should triple the number of virtual
     * particles
     */
    filler->setDensity(4.0);
    filler->fill(1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 6 * 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
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

            const Scalar4 r = h_pos.data[i];
            if (r.x >= Scalar(-8.0) && r.x <= Scalar(8.0))
                {
                if (r.y < Scalar(-5.0))
                    ++N_lo;
                else if (r.y >= Scalar(5.0))
                    ++N_hi;
                }
            }
        UP_ASSERT_EQUAL(N_lo, 6 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        UP_ASSERT_EQUAL(N_hi, 6 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        }

    /*
     * Change the cell size so that we lie exactly on a boundary.
     *
     * Now, all sides of the U have thickness 0.5.
     */
    pdata->removeVirtualParticles();
    cl->setCellSize(1.0);
    filler->fill(2);
    UP_ASSERT_EQUAL(pdata->getNVirtual(),
                    (unsigned int)(4 * 2 * (0.5 * 4.5 + 0.5 * 16 + 0.5 * 4.5) * 20));
        // count that particles have been placed on the right sides
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        unsigned int N_lo(0), N_hi(0);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar4 r = h_pos.data[i];
            if (r.x >= Scalar(-8.0) && r.x <= Scalar(8.0))
                {
                if (r.y < Scalar(-5.0))
                    ++N_lo;
                else if (r.y >= Scalar(5.0))
                    ++N_hi;
                }
            }
        UP_ASSERT_EQUAL(N_lo, (unsigned int)(4 * (8 + 4.5) * 20));
        UP_ASSERT_EQUAL(N_hi, (unsigned int)(4 * (8 + 4.5) * 20));
        }

    /*
     * Test the average fill properties of the virtual particles.
     */
    filler->setDensity(2.0);
    cl->setCellSize(2.0);
    unsigned int N_avg(0);
    Scalar3 v_avg = make_scalar3(0, 0, 0);
    Scalar T_avg(0);
    for (unsigned int t = 0; t < 500; ++t)
        {
        pdata->removeVirtualParticles();
        filler->fill(3 + t);

        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int i = pdata->getN(); i < pdata->getN() + pdata->getNVirtual(); ++i)
            {
            const Scalar4 vel_cell = h_vel.data[i];
            const Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);

            ++N_avg;
            v_avg += vel;
            T_avg += dot(vel, vel);
            }
        }
    // make averages
    v_avg /= N_avg;
    T_avg /= (3 * (N_avg - 1));

    CHECK_SMALL(v_avg.x, tol);
    CHECK_SMALL(v_avg.y, tol);
    CHECK_SMALL(v_avg.z, tol);
    CHECK_CLOSE(T_avg, 1.5, tol);
    }

UP_TEST(planar_pore_fill_basic)
    {
    planar_pore_fill_basic_test<mpcd::PlanarPoreGeometryFiller>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
UP_TEST(planar_pore_fill_basic_gpu)
    {
    planar_pore_fill_basic_test<mpcd::PlanarPoreGeometryFillerGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
