// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "hoomd/mpcd/SlitGeometryFiller.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/SlitGeometryFillerGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

template<class F>
void slit_fill_mpi_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(20.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf,snap->global_box.getL(),2,2,2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));

    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        std::shared_ptr<mpcd::ParticleDataSnapshot> mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(1);

        mpcd_snap->position[0] = vec3<Scalar>(1,1,1);
        mpcd_snap->velocity[0] = vec3<Scalar>(123, 456, 789);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    auto pdata = mpcd_sys->getParticleData();
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 0);
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);

    // create slit channel with half width 5
    auto slit = std::make_shared<const mpcd::detail::SlitGeometry>(5.0, 0.0, mpcd::detail::boundary::no_slip);
    std::shared_ptr<::Variant> kT = std::make_shared<::VariantConst>(1.0);
    std::shared_ptr<mpcd::SlitGeometryFiller> filler = std::make_shared<F>(mpcd_sys, 2.0, 0, kT, 42, slit);

    /*
     * Test basic filling up for this cell list
     */
    filler->fill(0);
    // volume to fill is from 5->5.5 (0.5) on + side, with cross section of 10^2 locally
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2*(10*10/2));
    // globally, cross section is 20^2 globally and also mirrored on bottom
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 2*(20*20/2)*2);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        const BoxDim& box = sysdef->getParticleData()->getBox();
        for (unsigned int i = 0; i < pdata->getNVirtual(); ++i)
            {
            const unsigned int idx = pdata->getN() + i;
            // each rank held 100 particles, so range is easy to determine
            UP_ASSERT_EQUAL(h_tag.data[idx] , 1 + exec_conf->getRank()*100 + i);
            UP_ASSERT(h_pos.data[idx].x >= box.getLo().x && h_pos.data[idx].x < box.getHi().x);
            UP_ASSERT(h_pos.data[idx].y >= box.getLo().y && h_pos.data[idx].y < box.getHi().y);
            UP_ASSERT(h_pos.data[idx].z >= box.getLo().z && h_pos.data[idx].z < box.getHi().z);
            }
        }

    /*
     * Fill up a second time
     */
    filler->fill(1);
    UP_ASSERT_EQUAL(pdata->getNVirtual(), 2*2*(10*10/2));
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 2*2*(20*20/2)*2);

    pdata->removeVirtualParticles();
    UP_ASSERT_EQUAL(pdata->getNVirtualGlobal(), 0);
    }

UP_TEST( slit_fill_mpi )
    {
    slit_fill_mpi_test<mpcd::SlitGeometryFiller>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_CUDA
UP_TEST( slit_fill_mpi_gpu )
    {
    slit_fill_mpi_test<mpcd::SlitGeometryFillerGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_CUDA
