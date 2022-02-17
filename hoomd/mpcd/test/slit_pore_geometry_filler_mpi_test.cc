// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/Communicator.h"
#include "hoomd/mpcd/SlitPoreGeometryFiller.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/SlitPoreGeometryFillerGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

template<class F> void slit_pore_fill_mpi_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(20.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        std::shared_ptr<mpcd::ParticleDataSnapshot> mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(1);

        mpcd_snap->position[0] = vec3<Scalar>(1, 1, 1);
        mpcd_snap->velocity[0] = vec3<Scalar>(123, 456, 789);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    auto pdata = mpcd_sys->getParticleData();
    mpcd_sys->getCellList()->setCellSize(2.0);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);

    // create slit channel with half width 5 and half length 8
    auto slit
        = std::make_shared<const mpcd::detail::SlitPoreGeometry>(5.0,
                                                                 8.0,
                                                                 mpcd::detail::boundary::no_slip);
    std::shared_ptr<Variant> kT = std::make_shared<VariantConstant>(1.0);
    std::shared_ptr<mpcd::SlitPoreGeometryFiller> filler
        = std::make_shared<F>(mpcd_sys, 2.0, 0, kT, 42, slit);

    /*
     * Test basic filling up for this cell list
     *
     * The fill volume is a U-shape |___|. The width of the sides is 1, since they align with the
     * grid. The width of the bottom is 2, since the wall slices halfway through the cell. The are
     * is the sum of the 3 rectangles, and the volume is multiplied by the box size.
     *
     * Each rank only owns half of this geometry, and then half of the y box size..
     */
    filler->fill(0);
    // volume to fill is from 5->5.5 (0.5) on + side, with cross section of 10^2 locally
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2 * (1 * 3 + 2 * 8) * 10);
    // globally, all ranks should have particles (8x larger)
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 2 * 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        const BoxDim& box = sysdef->getParticleData()->getBox();
        for (unsigned int i = 0; i < pdata->getNVirtual(); ++i)
            {
            const unsigned int idx = pdata->getN() + i;
            // each rank held 380 particles, so range is easy to determine
            UP_ASSERT_EQUAL(h_tag.data[idx], 1 + exec_conf->getRank() * 380 + i);
            UP_ASSERT(h_pos.data[idx].x >= box.getLo().x && h_pos.data[idx].x < box.getHi().x);
            UP_ASSERT(h_pos.data[idx].y >= box.getLo().y && h_pos.data[idx].y < box.getHi().y);
            UP_ASSERT(h_pos.data[idx].z >= box.getLo().z && h_pos.data[idx].z < box.getHi().z);
            }
        }

    /*
     * Fill up a second time
     */
    filler->fill(1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 4 * (1 * 3 + 2 * 8) * 10);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 4 * 2 * (1 * 3 + 2 * 16 + 1 * 3) * 20);

    pdata->removeVirtualParticles();
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);
    }

UP_TEST(slit_pore_fill_mpi)
    {
    slit_pore_fill_mpi_test<mpcd::SlitPoreGeometryFiller>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
UP_TEST(slit_pore_fill_mpi_gpu)
    {
    slit_pore_fill_mpi_test<mpcd::SlitPoreGeometryFillerGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
