// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "hoomd/mpcd/CellList.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/CellListGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

//! Test for correct calculation of MPCD grid dimensions
template<class CL>
void celllist_dimension_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(6.0, 8.0, 10.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // initialize mpcd system
    auto mpcd_snap = std::make_shared<mpcd::ParticleDataSnapshot>(1);
    auto pdata_1 = std::make_shared<mpcd::ParticleData>(mpcd_snap, snap->global_box, exec_conf);

    // define a system of different edge lengths
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, pdata_1));

    // compute the cell list dimensions
    cl->computeDimensions();

    // check the dimensions
    uint3 dim = cl->getDim();
    CHECK_EQUAL_UINT(dim.x, 6);
    CHECK_EQUAL_UINT(dim.y, 8);
    CHECK_EQUAL_UINT(dim.z, 10);

    // check the indexers
    Index3D cell_indexer = cl->getCellIndexer();
    CHECK_EQUAL_UINT(cell_indexer.getNumElements(), 6*8*10);
    UP_ASSERT(cl->getCellSizeArray().getNumElements() >= 6*8*10);    // Each cell has one number

    unsigned int Nmax = cl->getNmax();
    CHECK_EQUAL_UINT(Nmax, 4);    // Default is 4 particles per cell, ensure this happens if there's only one

    Index2D cli = cl->getCellListIndexer();
    CHECK_EQUAL_UINT(cli.getNumElements(), 6*8*10*Nmax);    // Cell list indexer has Nmax entries per cell

    // Cell list uses amortized sizing, so must only be at least this big
    UP_ASSERT(cl->getCellList().getNumElements() >= 6*8*10*Nmax);

    /*******************/
    // Change the cell size, and ensure everything stays up to date
    cl->setCellSize(2.0);
    cl->computeDimensions();

    dim = cl->getDim();
    CHECK_EQUAL_UINT(dim.x, 3);
    CHECK_EQUAL_UINT(dim.y, 4);
    CHECK_EQUAL_UINT(dim.z, 5);

    // check the indexers
    cell_indexer = cl->getCellIndexer();
    CHECK_EQUAL_UINT(cell_indexer.getNumElements(), 3*4*5);
    UP_ASSERT(cl->getCellSizeArray().getNumElements() >= 3*4*5);    // Each cell has one number

    cli = cl->getCellListIndexer();
    CHECK_EQUAL_UINT(cli.getNumElements(), 3*4*5*Nmax);    // Cell list indexer has Nmax entries per cell

    // Cell list uses amortized sizing, so must only be at least this big
    UP_ASSERT(cl->getCellList().getNumElements() >= 3*4*5*Nmax);

    /*******************/
    // Change the cell size to something that does not evenly divide a side, and check for an exception
    // evenly divides no sides
    cl->setCellSize(4.2);
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->computeDimensions(); } );
    // evenly divides z
    cl->setCellSize(10.0);
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->computeDimensions(); } );
    //evenly divides y
    cl->setCellSize(8.0);
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->computeDimensions(); } );
    //evenly divides x
    cl->setCellSize(6.0);
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->computeDimensions(); } );
    }

//! Test for correct cell listing of a small system
template<class CL>
void celllist_small_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // place each particle in a different cell, doubling the first cell
    std::shared_ptr<mpcd::ParticleData> pdata_9;
        {
        auto mpcd_snap = std::make_shared<mpcd::ParticleDataSnapshot>(9);

        mpcd_snap->position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        mpcd_snap->position[1] = vec3<Scalar>( 0.5, -0.5, -0.5);
        mpcd_snap->position[2] = vec3<Scalar>(-0.5,  0.5, -0.5);
        mpcd_snap->position[3] = vec3<Scalar>( 0.5,  0.5, -0.5);
        mpcd_snap->position[4] = vec3<Scalar>(-0.5, -0.5,  0.5);
        mpcd_snap->position[5] = vec3<Scalar>( 0.5, -0.5,  0.5);
        mpcd_snap->position[6] = vec3<Scalar>(-0.5,  0.5,  0.5);
        mpcd_snap->position[7] = vec3<Scalar>( 0.5,  0.5,  0.5);

        mpcd_snap->position[8] = vec3<Scalar>(-0.5, -0.5, -0.5);
        pdata_9 = std::make_shared<mpcd::ParticleData>(mpcd_snap, snap->global_box, exec_conf);
        }

    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, pdata_9));
    cl->compute(0);

    // check that each particle is in the proper bin (cell list and velocity)
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(), access_location::host, access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 2 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 1 );

        // check the particle ids in each cell
        Index2D cli = cl->getCellListIndexer();
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,0))], 0 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(1, ci(0,0,0))], 8 );

        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,1))], 4 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,0))], 2 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,1))], 6 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,0,0))], 1 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,0,1))], 5 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,1,0))], 3 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,1,1))], 7 );

        ArrayHandle<Scalar4> h_vel(pdata_9->getVelocities(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[0].w), ci(0,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[1].w), ci(1,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[2].w), ci(0,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[3].w), ci(1,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[4].w), ci(0,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[5].w), ci(1,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[6].w), ci(0,1,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[7].w), ci(1,1,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[8].w), ci(0,0,0) );
        }

    // condense particles into two bins
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(-0.3, -0.3, -0.3, 0.0);
        h_pos.data[1] = make_scalar4( 0.3,  0.3,  0.3, 0.0);
        h_pos.data[2] = h_pos.data[0];
        h_pos.data[3] = h_pos.data[1];
        h_pos.data[4] = h_pos.data[0];
        h_pos.data[5] = h_pos.data[1];
        h_pos.data[6] = h_pos.data[0];
        h_pos.data[7] = h_pos.data[1];
        h_pos.data[8] = h_pos.data[0];
        }

    // check that each bin owns the right particles
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(), access_location::host, access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 5 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 4 );

        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 0);

        // check the particle ids in each cell
        Index2D cli = cl->getCellListIndexer();
            {
            std::vector<unsigned int> pids(5,0);
            for (unsigned int i=0; i < 5; ++i)
                {
                pids[i] = h_cell_list.data[cli(i, ci(0,0,0))];
                }
            sort(pids.begin(), pids.end());
            unsigned int check_pids[] = {0,2,4,6,8};
            UP_ASSERT_EQUAL(pids, check_pids);
            }
            {
            std::vector<unsigned int> pids(4,0);
            for (unsigned int i=0; i < 4; ++i)
                {
                pids[i] = h_cell_list.data[cli(i, ci(1,1,1))];
                }
            sort(pids.begin(), pids.end());
            unsigned int check_pids[] = {1,3,5,7};
            UP_ASSERT_EQUAL(pids, check_pids);
            }
        }

    // bring all particles into one box, which triggers a resize, and check that all particles are in this bin
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(0.9, -0.4, 0.0, 0.0);
        for (unsigned int i=1; i < 9; ++i)
            h_pos.data[i] = h_pos.data[0];
        }
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);

        // validate that this cell owns all the particles, and others own none
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 9);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 0);
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 0);
        }

    // send a particle out of bounds and check that an exception is raised
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(2.1, 2.1, 2.1, __int_as_scalar(0));
        }
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->compute(3); });
    // check the other side as well
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(-2.1, -2.1, -2.1, __int_as_scalar(0));
        }
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->compute(4); });
    }

//! Test that particles can be grid shifted correctly
template<class CL>
void celllist_grid_shift_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(6.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_1;
        {
        auto mpcd_snap = std::make_shared<mpcd::ParticleDataSnapshot>(1);
        mpcd_snap->position[0] = vec3<Scalar>(0.1, 0.1, 0.1);
        pdata_1 = std::make_shared<mpcd::ParticleData>(mpcd_snap, snap->global_box, exec_conf);
        }

    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, pdata_1));
    cl->compute(0);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(3,3,3)], 1);
        }

    // shift the grid and see that it falls from (3,3,3) to (2,2,2)
    cl->setGridShift(make_scalar3(0.5, 0.5, 0.5));
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(2,2,2)], 1);
        }

    // move to the other side and retry
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(-0.1, -0.1, -0.1, 0.0);
        }
    cl->setGridShift(make_scalar3(-0.5,-0.5,-0.5));
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(3,3,3)], 1);
        }

    // check for cell periodic wrapping by putting particles near the box boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(-2.9, -2.9, -2.9, 0.0);
        }
    cl->setGridShift(make_scalar3(0.5,0.5,0.5));
    cl->compute(3);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(5,5,5)], 1);
        }

    // and the other way
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(), access_location::host, access_mode::overwrite);
        h_pos.data[0] = make_scalar4(2.9, 2.9, 2.9, 0.0);
        }
    cl->setGridShift(make_scalar3(-0.5,-0.5,-0.5));
    cl->compute(4);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 1);
        }

    // check for error in grid shifting
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->setGridShift(make_scalar3(-0.51, -0.51, -0.51)); });
    UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ cl->setGridShift(make_scalar3( 0.51,  0.51,  0.51)); });
    }

//! Test that small systems can embed particles
template<class CL>
void celllist_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // setup a system where both MD and MPCD particles are in each of the cells
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.type_mapping.push_back("A");
        pdata_snap.type_mapping.push_back("B");
        pdata_snap.resize(8);
        pdata_snap.pos[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        pdata_snap.pos[1] = vec3<Scalar>( 0.5, -0.5, -0.5);
        pdata_snap.pos[2] = vec3<Scalar>(-0.5,  0.5, -0.5);
        pdata_snap.pos[3] = vec3<Scalar>( 0.5,  0.5, -0.5);
        pdata_snap.pos[4] = vec3<Scalar>(-0.5, -0.5,  0.5);
        pdata_snap.pos[5] = vec3<Scalar>( 0.5, -0.5,  0.5);
        pdata_snap.pos[6] = vec3<Scalar>(-0.5,  0.5,  0.5);
        pdata_snap.pos[7] = vec3<Scalar>( 0.5,  0.5,  0.5);
        pdata_snap.type[0] = 0;
        pdata_snap.type[1] = 1;
        pdata_snap.type[2] = 0;
        pdata_snap.type[3] = 1;
        pdata_snap.type[4] = 0;
        pdata_snap.type[5] = 1;
        pdata_snap.type[6] = 0;
        pdata_snap.type[7] = 1;
        }
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_8;
        {
        auto mpcd_snap = std::make_shared<mpcd::ParticleDataSnapshot>(8);
        mpcd_snap->position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        mpcd_snap->position[1] = vec3<Scalar>( 0.5, -0.5, -0.5);
        mpcd_snap->position[2] = vec3<Scalar>(-0.5,  0.5, -0.5);
        mpcd_snap->position[3] = vec3<Scalar>( 0.5,  0.5, -0.5);
        mpcd_snap->position[4] = vec3<Scalar>(-0.5, -0.5,  0.5);
        mpcd_snap->position[5] = vec3<Scalar>( 0.5, -0.5,  0.5);
        mpcd_snap->position[6] = vec3<Scalar>(-0.5,  0.5,  0.5);
        mpcd_snap->position[7] = vec3<Scalar>( 0.5,  0.5,  0.5);

        pdata_8 = std::make_shared<mpcd::ParticleData>(mpcd_snap, snap->global_box, exec_conf);
        }

    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef,pdata_8));
    cl->compute(0);

    // at first, there is no embedded particle, so everything should just look like the test before
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(), access_location::host, access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 1 );

        // check the particle ids in each cell
        Index2D cli = cl->getCellListIndexer();
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,0))], 0 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,1))], 4 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,0))], 2 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,1))], 6 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,0,0))], 1 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,0,1))], 5 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,1,0))], 3 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,1,1))], 7 );

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[0].w), ci(0,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[1].w), ci(1,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[2].w), ci(0,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[3].w), ci(1,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[4].w), ci(0,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[5].w), ci(1,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[6].w), ci(0,1,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[7].w), ci(1,1,1) );
        }

    // now we include the half embedded group
    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_B(new ParticleSelectorType(sysdef, 1,1));
    std::shared_ptr<ParticleGroup> group_B(new ParticleGroup(sysdef, selector_B));

    cl->setEmbeddedGroup(group_B);
    cl->compute(1);

    // now there should be a second particle in the cell
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(), access_location::host, access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 2 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 2 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 2 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 2 );

        // check the particle ids in each cell
        Index2D cli = cl->getCellListIndexer();
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,0))], 0 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,1))], 4 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,0))], 2 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,1))], 6 );
        // check two particles in cell (1,0,0)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,0,0))];
            result[1] = h_cell_list.data[cli(1,ci(1,0,0))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{1,8});
            }
        // check two particles in cell (1,0,1)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,0,1))];
            result[1] = h_cell_list.data[cli(1,ci(1,0,1))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{5,10});
            }
        // check two particles in cell (1,1,0)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,1,0))];
            result[1] = h_cell_list.data[cli(1,ci(1,1,0))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{3,9});
            }
        // check two particles in cell (1,1,1)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,1,1))];
            result[1] = h_cell_list.data[cli(1,ci(1,1,1))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{7,11});
            }

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[0].w), ci(0,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[1].w), ci(1,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[2].w), ci(0,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[3].w), ci(1,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[4].w), ci(0,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[5].w), ci(1,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[6].w), ci(0,1,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[7].w), ci(1,1,1) );

        ArrayHandle<unsigned int> h_embed_cell_ids(cl->getEmbeddedGroupCellIds(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[0], ci(1,0,0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[1], ci(1,1,0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[2], ci(1,0,1));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[3], ci(1,1,1));

        // all masses should stil be set to original values
        ArrayHandle<Scalar4> h_embed_vel(embed_pdata->getVelocities(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_embed_vel.data[0].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[1].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[2].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[3].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[4].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[5].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[6].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[7].w, 1.0, tol);
        }

    // pick a particle up and put it in a different cell, now there will be an extra embedded particle
        {
        ArrayHandle<Scalar4> h_embed_pos(embed_pdata->getPositions(), access_location::host, access_mode::overwrite);
        h_embed_pos.data[1] = make_scalar4(0.5, 0.5, -0.5, __int_as_scalar(1));
        }
    cl->compute(2);
    // now there should be a second particle in the cell
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(), access_location::host, access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,0,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(0,1,1)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,0)], 1 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,0,1)], 2 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,0)], 3 );
        CHECK_EQUAL_UINT( h_cell_np.data[ci(1,1,1)], 2 );

        // check the particle ids in each cell
        Index2D cli = cl->getCellListIndexer();
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,0))], 0 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,0,1))], 4 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,0))], 2 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(1,0,0))], 1 );
        CHECK_EQUAL_UINT( h_cell_list.data[cli(0, ci(0,1,1))], 6 );
        // check two particles in cell (1,0,1)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,0,1))];
            result[1] = h_cell_list.data[cli(1,ci(1,0,1))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{5,10});
            }
        // check two particles in cell (1,1,0)
            {
            std::vector<unsigned int> result(3);
            result[0] = h_cell_list.data[cli(0,ci(1,1,0))];
            result[1] = h_cell_list.data[cli(1,ci(1,1,0))];
            result[2] = h_cell_list.data[cli(2,ci(1,1,0))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{3,8,9});
            }
        // check two particles in cell (1,1,1)
            {
            std::vector<unsigned int> result(2);
            result[0] = h_cell_list.data[cli(0,ci(1,1,1))];
            result[1] = h_cell_list.data[cli(1,ci(1,1,1))];
            sort(result.begin(), result.end());
            UP_ASSERT_EQUAL(result, std::vector<unsigned int>{7,11});
            }

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[0].w), ci(0,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[1].w), ci(1,0,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[2].w), ci(0,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[3].w), ci(1,1,0) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[4].w), ci(0,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[5].w), ci(1,0,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[6].w), ci(0,1,1) );
        CHECK_EQUAL_UINT( __scalar_as_int(h_vel.data[7].w), ci(1,1,1) );

        ArrayHandle<unsigned int> h_embed_cell_ids(cl->getEmbeddedGroupCellIds(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[0], ci(1,1,0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[1], ci(1,1,0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[2], ci(1,0,1));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[3], ci(1,1,1));

        // all masses should stil be set to original values
        ArrayHandle<Scalar4> h_embed_vel(embed_pdata->getVelocities(), access_location::host, access_mode::read);
        CHECK_CLOSE(h_embed_vel.data[0].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[1].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[2].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[3].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[4].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[5].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[6].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[7].w, 1.0, tol);
        }
    }

//! dimension test case for MPCD CellList class
UP_TEST( mpcd_cell_list_dimensions )
    {
    celllist_dimension_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! small system test case for MPCD CellList class
UP_TEST( mpcd_cell_list_small_test )
    {
    celllist_small_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! small system test case for MPCD CellList class
UP_TEST( mpcd_cell_list_grid_shift_test )
    {
    celllist_grid_shift_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! embedded particle test case for MPCD CellList class
UP_TEST( mpcd_cell_list_embed_test )
    {
    celllist_embed_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! dimension test case for MPCD CellListGPU class
UP_TEST( mpcd_cell_list_gpu_dimensions )
    {
    celllist_dimension_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! small system test case for MPCD CellListGPU class
UP_TEST( mpcd_cell_list_gpu_small_test )
    {
    celllist_small_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! small system test case for MPCD CellListGPU class
UP_TEST( mpcd_cell_list_gpu_grid_shift_test )
    {
    celllist_grid_shift_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! embedded particle test case for MPCD CellListGPU class
UP_TEST( mpcd_cell_list_gpu_embed_test )
    {
    celllist_embed_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif // ENABLE_CUDA
