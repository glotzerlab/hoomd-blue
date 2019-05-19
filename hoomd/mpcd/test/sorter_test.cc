// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "hoomd/mpcd/Sorter.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/SorterGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

//! Test for basic MPCD sort functions
template<class T>
void sorter_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // default initialize an empty snapshot in the reference box
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
        {
        // embed one particle
        snap->particle_data.resize(1);
        snap->particle_data.pos[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        }
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // place eight mpcd particles, one per cell
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->type_mapping.push_back("M");
        mpcd_snap->type_mapping.push_back("P");
        mpcd_snap->type_mapping.push_back("H");
        mpcd_snap->type_mapping.push_back("R");
        mpcd_snap->type_mapping.push_back("L");
        mpcd_snap->type_mapping.push_back("G");
        mpcd_snap->type_mapping.push_back("PSU");
        mpcd_snap->type_mapping.push_back("PU");

        mpcd_snap->resize(8);
        mpcd_snap->position[7] = vec3<Scalar>(-0.5,-0.5,-0.5);
        mpcd_snap->position[6] = vec3<Scalar>(0.5,-0.5,-0.5);
        mpcd_snap->position[5] = vec3<Scalar>(-0.5, 0.5,-0.5);
        mpcd_snap->position[4] = vec3<Scalar>(0.5, 0.5,-0.5);
        mpcd_snap->position[3] = vec3<Scalar>(-0.5,-0.5, 0.5);
        mpcd_snap->position[2] = vec3<Scalar>(0.5,-0.5, 0.5);
        mpcd_snap->position[1] = vec3<Scalar>(-0.5, 0.5, 0.5);
        mpcd_snap->position[0] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[7] = vec3<Scalar>(0., -0.5, 0.5);
        mpcd_snap->velocity[6] = vec3<Scalar>(1., -1.5, 1.5);
        mpcd_snap->velocity[5] = vec3<Scalar>(2., -2.5, 2.5);
        mpcd_snap->velocity[4] = vec3<Scalar>(3., -3.5, 3.5);
        mpcd_snap->velocity[3] = vec3<Scalar>(4., -4.5, 4.5);
        mpcd_snap->velocity[2] = vec3<Scalar>(5., -5.5, 5.5);
        mpcd_snap->velocity[1] = vec3<Scalar>(6., -6.5, 6.5);
        mpcd_snap->velocity[0] = vec3<Scalar>(7., -7.5, 7.5);

        mpcd_snap->type[7] = 0;
        mpcd_snap->type[6] = 1;
        mpcd_snap->type[5] = 2;
        mpcd_snap->type[4] = 3;
        mpcd_snap->type[3] = 4;
        mpcd_snap->type[2] = 5;
        mpcd_snap->type[1] = 6;
        mpcd_snap->type[0] = 7;
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);

    // add an embedded group
    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group(new ParticleGroup(sysdef, selector));
    mpcd_sys->getCellList()->setEmbeddedGroup(group);

    // run the sorter
    std::shared_ptr<T> sorter = std::make_shared<T>(mpcd_sys,0,1);
    sorter->update(0);

    // check that all particles are properly ordered
        {
        std::shared_ptr<mpcd::ParticleData> pdata = mpcd_sys->getParticleData();

        // tag order should be reversed
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_tag.data[0], 7);
        UP_ASSERT_EQUAL(h_tag.data[1], 6);
        UP_ASSERT_EQUAL(h_tag.data[2], 5);
        UP_ASSERT_EQUAL(h_tag.data[3], 4);
        UP_ASSERT_EQUAL(h_tag.data[4], 3);
        UP_ASSERT_EQUAL(h_tag.data[5], 2);
        UP_ASSERT_EQUAL(h_tag.data[6], 1);
        UP_ASSERT_EQUAL(h_tag.data[7], 0);

        // positions should be in order now
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, -0.5, tol); CHECK_CLOSE(h_pos.data[0].y, -0.5, tol); CHECK_CLOSE(h_pos.data[0].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[1].x,  0.5, tol); CHECK_CLOSE(h_pos.data[1].y, -0.5, tol); CHECK_CLOSE(h_pos.data[1].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[2].x, -0.5, tol); CHECK_CLOSE(h_pos.data[2].y,  0.5, tol); CHECK_CLOSE(h_pos.data[2].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[3].x,  0.5, tol); CHECK_CLOSE(h_pos.data[3].y,  0.5, tol); CHECK_CLOSE(h_pos.data[3].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[4].x, -0.5, tol); CHECK_CLOSE(h_pos.data[4].y, -0.5, tol); CHECK_CLOSE(h_pos.data[4].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[5].x,  0.5, tol); CHECK_CLOSE(h_pos.data[5].y, -0.5, tol); CHECK_CLOSE(h_pos.data[5].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[6].x, -0.5, tol); CHECK_CLOSE(h_pos.data[6].y,  0.5, tol); CHECK_CLOSE(h_pos.data[6].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[7].x,  0.5, tol); CHECK_CLOSE(h_pos.data[7].y,  0.5, tol); CHECK_CLOSE(h_pos.data[7].z,  0.5, tol);
        // types were set to the actual order of things
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[0].w), 0);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[1].w), 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[2].w), 2);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[3].w), 3);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[4].w), 4);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[5].w), 5);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[6].w), 6);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[7].w), 7);

        // velocities should also be sorted
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_vel.data[0].x, 0., tol); CHECK_CLOSE(h_vel.data[0].y, -0.5, tol); CHECK_CLOSE(h_vel.data[0].z, 0.5, tol);
        CHECK_CLOSE(h_vel.data[1].x, 1., tol); CHECK_CLOSE(h_vel.data[1].y, -1.5, tol); CHECK_CLOSE(h_vel.data[1].z, 1.5, tol);
        CHECK_CLOSE(h_vel.data[2].x, 2., tol); CHECK_CLOSE(h_vel.data[2].y, -2.5, tol); CHECK_CLOSE(h_vel.data[2].z, 2.5, tol);
        CHECK_CLOSE(h_vel.data[3].x, 3., tol); CHECK_CLOSE(h_vel.data[3].y, -3.5, tol); CHECK_CLOSE(h_vel.data[3].z, 3.5, tol);
        CHECK_CLOSE(h_vel.data[4].x, 4., tol); CHECK_CLOSE(h_vel.data[4].y, -4.5, tol); CHECK_CLOSE(h_vel.data[4].z, 4.5, tol);
        CHECK_CLOSE(h_vel.data[5].x, 5., tol); CHECK_CLOSE(h_vel.data[5].y, -5.5, tol); CHECK_CLOSE(h_vel.data[5].z, 5.5, tol);
        CHECK_CLOSE(h_vel.data[6].x, 6., tol); CHECK_CLOSE(h_vel.data[6].y, -6.5, tol); CHECK_CLOSE(h_vel.data[6].z, 6.5, tol);
        CHECK_CLOSE(h_vel.data[7].x, 7., tol); CHECK_CLOSE(h_vel.data[7].y, -7.5, tol); CHECK_CLOSE(h_vel.data[7].z, 7.5, tol);
        // cells should be in the right order now too
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), 0);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[1].w), 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[2].w), 2);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[3].w), 3);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[4].w), 4);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[5].w), 5);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[6].w), 6);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[7].w), 7);
        }

    // check that the cell list has been updated as well
        {
        auto cl = mpcd_sys->getCellList();
        ArrayHandle<unsigned int> h_cl(cl->getCellList(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        const Index3D& ci = cl->getCellIndexer();
        const Index2D& cli = cl->getCellListIndexer();

        // all cells should have one particle, except the first cell, which has the embedded one
        UP_ASSERT_EQUAL(h_np.data[ci(0,0,0)], 2);
        UP_ASSERT_EQUAL(h_np.data[ci(1,0,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,1,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,1,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,0,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,0,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,1,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,1,1)], 1);

        // the particles should be in ascending order
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,0,0))], 1);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,1,0))], 2);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,1,0))], 3);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,0,1))], 4);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,0,1))], 5);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,1,1))], 6);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,1,1))], 7);
        // do first cell separately, since it needs to be a sorted list
        std::vector<unsigned int> cell_0 = {h_cl.data[cli(0,ci(0,0,0))], h_cl.data[cli(1,ci(0,0,0))]};
        std::sort(cell_0.begin(), cell_0.end());
        UP_ASSERT_EQUAL(cell_0, std::vector<unsigned int>{0,8});
        }
    }

//! Test for MPCD sorting with virtual particles
template<class T>
void sorter_virtual_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // default initialize an empty snapshot in the reference box
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // place eight mpcd particles, one per cell
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->type_mapping.push_back("M");
        mpcd_snap->type_mapping.push_back("P");
        mpcd_snap->type_mapping.push_back("H");
        mpcd_snap->type_mapping.push_back("R");
        mpcd_snap->type_mapping.push_back("L");
        mpcd_snap->type_mapping.push_back("G");
        mpcd_snap->type_mapping.push_back("PSU");
        mpcd_snap->type_mapping.push_back("PU");

        mpcd_snap->resize(6);
        mpcd_snap->position[5] = vec3<Scalar>(-0.5,-0.5,-0.5);
        mpcd_snap->position[4] = vec3<Scalar>(-0.5, 0.5,-0.5);
        mpcd_snap->position[3] = vec3<Scalar>(-0.5,-0.5, 0.5);
        mpcd_snap->position[2] = vec3<Scalar>(0.5,-0.5, 0.5);
        mpcd_snap->position[1] = vec3<Scalar>(-0.5, 0.5, 0.5);
        mpcd_snap->position[0] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[5] = vec3<Scalar>(0., -0.5, 0.5);
        mpcd_snap->velocity[4] = vec3<Scalar>(2., -2.5, 2.5);
        mpcd_snap->velocity[3] = vec3<Scalar>(4., -4.5, 4.5);
        mpcd_snap->velocity[2] = vec3<Scalar>(5., -5.5, 5.5);
        mpcd_snap->velocity[1] = vec3<Scalar>(6., -6.5, 6.5);
        mpcd_snap->velocity[0] = vec3<Scalar>(7., -7.5, 7.5);

        mpcd_snap->type[5] = 0;
        mpcd_snap->type[4] = 2;
        mpcd_snap->type[3] = 4;
        mpcd_snap->type[2] = 5;
        mpcd_snap->type[1] = 6;
        mpcd_snap->type[0] = 7;
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);

    // add 2 virtual particles to fill in the rest of the cells
    auto pdata = mpcd_sys->getParticleData();
    pdata->addVirtualParticles(2);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::readwrite);

        h_pos.data[pdata->getN()+0] = make_scalar4(0.5,-0.5,-0.5,__int_as_scalar(1));
        h_vel.data[pdata->getN()+0] = make_scalar4(1., -1.5, 1.5,__int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pdata->getN()+0] = 6;

        h_pos.data[pdata->getN()+1] = make_scalar4(0.5, 0.5,-0.5,__int_as_scalar(3));
        h_vel.data[pdata->getN()+1] = make_scalar4(3., -3.5, 3.5,__int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pdata->getN()+1] = 7;
        }

    // run the sorter
    std::shared_ptr<T> sorter = std::make_shared<T>(mpcd_sys,0,1);
    sorter->update(0);

    // check that all particles are properly ordered
        {
        // tag order should be reversed, with virtual particles at the end unsorted
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_tag.data[0], 5);
        UP_ASSERT_EQUAL(h_tag.data[1], 4);
        UP_ASSERT_EQUAL(h_tag.data[2], 3);
        UP_ASSERT_EQUAL(h_tag.data[3], 2);
        UP_ASSERT_EQUAL(h_tag.data[4], 1);
        UP_ASSERT_EQUAL(h_tag.data[5], 0);
        // virtual particles
        UP_ASSERT_EQUAL(h_tag.data[6], 6);
        UP_ASSERT_EQUAL(h_tag.data[7], 7);

        // positions should be in order now, with virtual particles at the end unsorted
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, -0.5, tol); CHECK_CLOSE(h_pos.data[0].y, -0.5, tol); CHECK_CLOSE(h_pos.data[0].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[1].x, -0.5, tol); CHECK_CLOSE(h_pos.data[1].y,  0.5, tol); CHECK_CLOSE(h_pos.data[1].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[2].x, -0.5, tol); CHECK_CLOSE(h_pos.data[2].y, -0.5, tol); CHECK_CLOSE(h_pos.data[2].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[3].x,  0.5, tol); CHECK_CLOSE(h_pos.data[3].y, -0.5, tol); CHECK_CLOSE(h_pos.data[3].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[4].x, -0.5, tol); CHECK_CLOSE(h_pos.data[4].y,  0.5, tol); CHECK_CLOSE(h_pos.data[4].z,  0.5, tol);
        CHECK_CLOSE(h_pos.data[5].x,  0.5, tol); CHECK_CLOSE(h_pos.data[5].y,  0.5, tol); CHECK_CLOSE(h_pos.data[5].z,  0.5, tol);
        // virtual particles
        CHECK_CLOSE(h_pos.data[6].x,  0.5, tol); CHECK_CLOSE(h_pos.data[6].y, -0.5, tol); CHECK_CLOSE(h_pos.data[6].z, -0.5, tol);
        CHECK_CLOSE(h_pos.data[7].x,  0.5, tol); CHECK_CLOSE(h_pos.data[7].y,  0.5, tol); CHECK_CLOSE(h_pos.data[7].z, -0.5, tol);

        // types were set to the actual order of things
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[0].w), 0);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[1].w), 2);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[2].w), 4);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[3].w), 5);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[4].w), 6);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[5].w), 7);
        // VPs
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[6].w), 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_pos.data[7].w), 3);

        // velocities should also be sorted
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_vel.data[0].x, 0., tol); CHECK_CLOSE(h_vel.data[0].y, -0.5, tol); CHECK_CLOSE(h_vel.data[0].z, 0.5, tol);
        CHECK_CLOSE(h_vel.data[1].x, 2., tol); CHECK_CLOSE(h_vel.data[1].y, -2.5, tol); CHECK_CLOSE(h_vel.data[1].z, 2.5, tol);
        CHECK_CLOSE(h_vel.data[2].x, 4., tol); CHECK_CLOSE(h_vel.data[2].y, -4.5, tol); CHECK_CLOSE(h_vel.data[2].z, 4.5, tol);
        CHECK_CLOSE(h_vel.data[3].x, 5., tol); CHECK_CLOSE(h_vel.data[3].y, -5.5, tol); CHECK_CLOSE(h_vel.data[3].z, 5.5, tol);
        CHECK_CLOSE(h_vel.data[4].x, 6., tol); CHECK_CLOSE(h_vel.data[4].y, -6.5, tol); CHECK_CLOSE(h_vel.data[4].z, 6.5, tol);
        CHECK_CLOSE(h_vel.data[5].x, 7., tol); CHECK_CLOSE(h_vel.data[5].y, -7.5, tol); CHECK_CLOSE(h_vel.data[5].z, 7.5, tol);
        // VPs
        CHECK_CLOSE(h_vel.data[6].x, 1., tol); CHECK_CLOSE(h_vel.data[6].y, -1.5, tol); CHECK_CLOSE(h_vel.data[6].z, 1.5, tol);
        CHECK_CLOSE(h_vel.data[7].x, 3., tol); CHECK_CLOSE(h_vel.data[7].y, -3.5, tol); CHECK_CLOSE(h_vel.data[7].z, 3.5, tol);
        // cells should be in the right order now too
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), 0);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[1].w), 2);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[2].w), 4);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[3].w), 5);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[4].w), 6);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[5].w), 7);
        // VPs
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[6].w), 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[7].w), 3);
        }

    // check that the cell list has been updated as well
        {
        auto cl = mpcd_sys->getCellList();
        ArrayHandle<unsigned int> h_cl(cl->getCellList(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        const Index3D& ci = cl->getCellIndexer();
        const Index2D& cli = cl->getCellListIndexer();

        // all cells should have one particle
        UP_ASSERT_EQUAL(h_np.data[ci(0,0,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,0,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,1,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,1,0)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,0,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,0,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(0,1,1)], 1);
        UP_ASSERT_EQUAL(h_np.data[ci(1,1,1)], 1);

        // the particles should be in ascending order, with VPs interleaved unsorted
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,0,0))], 0);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,0,0))], 6);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,1,0))], 1);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,1,0))], 7);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,0,1))], 2);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,0,1))], 3);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(0,1,1))], 4);
        UP_ASSERT_EQUAL(h_cl.data[cli(0,ci(1,1,1))], 5);
        }
    }

//! basic test case for MPCD sorter
UP_TEST( mpcd_sorter_test )
    {
    sorter_test<mpcd::Sorter>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! test case for MPCD sorter with virtual particles
UP_TEST( mpcd_sorter_virtual_test )
    {
    sorter_virtual_test<mpcd::Sorter>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
#ifdef ENABLE_CUDA
UP_TEST( mpcd_sorter_test_gpu )
    {
    sorter_test<mpcd::SorterGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
UP_TEST( mpcd_sorter_virtual_test_gpu )
    {
    sorter_virtual_test<mpcd::SorterGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif // ENABLE_CUDA
