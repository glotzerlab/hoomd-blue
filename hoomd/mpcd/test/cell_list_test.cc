// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/CellList.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CellListGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/filter/ParticleFilterAll.h"
#include "hoomd/filter/ParticleFilterType.h"
#include "hoomd/test/upp11_config.h"

#include "utils.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of MPCD grid dimensions
template<class CL>
void celllist_dimension_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                             const Scalar3& L,
                             const Scalar3& tilt)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(L);
    snap->global_box->setTiltFactors(tilt.x, tilt.y, tilt.z);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    auto pdata_1 = sysdef->getMPCDParticleData();

    // define a system of different edge lengths
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(6, 8, 10), false));

    // compute the cell list dimensions
    cl->computeDimensions();

    // check the dimensions
    uint3 dim = cl->getDim();
    CHECK_EQUAL_UINT(dim.x, 6);
    CHECK_EQUAL_UINT(dim.y, 8);
    CHECK_EQUAL_UINT(dim.z, 10);

    // check the indexers
    Index3D cell_indexer = cl->getCellIndexer();
    CHECK_EQUAL_UINT(cell_indexer.getNumElements(), 6 * 8 * 10);
    UP_ASSERT(cl->getCellSizeArray().getNumElements() >= 6 * 8 * 10); // Each cell has one number

    /*******************/
    // Change the cell size, and ensure everything stays up to date
    cl->setGlobalDim(make_uint3(3, 4, 5));
    cl->computeDimensions();

    dim = cl->getDim();
    CHECK_EQUAL_UINT(dim.x, 3);
    CHECK_EQUAL_UINT(dim.y, 4);
    CHECK_EQUAL_UINT(dim.z, 5);

    // check the indexers
    cell_indexer = cl->getCellIndexer();
    CHECK_EQUAL_UINT(cell_indexer.getNumElements(), 3 * 4 * 5);
    UP_ASSERT(cl->getCellSizeArray().getNumElements() >= 3 * 4 * 5); // Each cell has one number
    }

//! Test for correct cell listing of a small system
template<class CL>
void celllist_small_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                         const Scalar3& L,
                         const Scalar3& tilt)
    {
    auto ref_box = std::make_shared<BoxDim>(2.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
    snap->particle_data.type_mapping.push_back("A");
    // place each particle in a different cell, doubling the first cell
    snap->mpcd_data.resize(9);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-0.5, -0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(0.5, -0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-0.5, 0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(0.5, 0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-0.5, -0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(0.5, -0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-0.5, 0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(0.5, 0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[8] = scale(vec3<Scalar>(-0.5, -0.5, -0.5), ref_box, box);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_9 = sysdef->getMPCDParticleData();

    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(2, 2, 2), false));
    cl->compute(0);

        // check that each particle is in the proper bin (cell list and velocity)
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 2);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 1);

        ArrayHandle<Scalar4> h_vel(pdata_9->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[1].w), ci(1, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[2].w), ci(0, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[3].w), ci(1, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[4].w), ci(0, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[5].w), ci(1, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[6].w), ci(0, 1, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[7].w), ci(1, 1, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[8].w), ci(0, 0, 0));
        }

        // condense particles into two bins
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(-0.3, -0.3, -0.3, __int_as_scalar(0)), ref_box, box);
        h_pos.data[1] = scale(make_scalar4(0.3, 0.3, 0.3, __int_as_scalar(0)), ref_box, box);
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
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 5);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 4);

        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 0);
        }

        // bring all particles into one box, which triggers a resize, and check that all particles
        // are in this bin
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(0.9, -0.4, 0.0, __int_as_scalar(0)), ref_box, box);
        for (unsigned int i = 1; i < 9; ++i)
            h_pos.data[i] = h_pos.data[0];
        }
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that this cell owns all the particles, and others own none
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 9);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 0);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 0);
        }

        // send a particle out of bounds and check that an exception is raised
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(2.1, 2.1, 2.1, __int_as_scalar(0)), ref_box, box);
        }
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { cl->compute(3); });
        // check the other side as well
        {
        ArrayHandle<Scalar4> h_pos(pdata_9->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(-2.1, -2.1, -2.1, __int_as_scalar(0)), ref_box, box);
        }
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { cl->compute(4); });
    }

//! Test that particles can be grid shifted correctly
template<class CL>
void celllist_grid_shift_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                              const Scalar3& L,
                              const Scalar3& tilt)
    {
    auto ref_box = std::make_shared<BoxDim>(6.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(0.1, 0.1, 0.1), ref_box, box);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_1 = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(6, 6, 6), false));
    cl->compute(0);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(3, 3, 3)], 1);
        }

    // shift the grid and see that it falls from (3,3,3) to (2,2,2)
    const Scalar3 shift = (Scalar(0.5) / 6) * make_scalar3(1, 1, 1);
    cl->setGridShift(shift);
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(2, 2, 2)], 1);
        }

        // move to the other side and retry
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(-0.1, -0.1, -0.1, __int_as_scalar(0)), ref_box, box);
        }
    cl->setGridShift(-shift);
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(3, 3, 3)], 1);
        }

        // check for cell periodic wrapping by putting particles near the box boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(-2.9, -2.9, -2.9, __int_as_scalar(0)), ref_box, box);
        }
    cl->setGridShift(shift);
    cl->compute(3);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(5, 5, 5)], 1);
        }

        // and the other way
        {
        ArrayHandle<Scalar4> h_pos(pdata_1->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        h_pos.data[0] = scale(make_scalar4(2.9, 2.9, 2.9, __int_as_scalar(0)), ref_box, box);
        }
    cl->setGridShift(-shift);
    cl->compute(4);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 1);
        }

    // check for error in grid shifting
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { cl->setGridShift(Scalar(-1.01) * shift); });
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { cl->setGridShift(Scalar(1.01) * shift); });
    }

//! Test that small systems can embed particles
template<class CL>
void celllist_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                         const Scalar3& L,
                         const Scalar3& tilt)
    {
    auto ref_box = std::make_shared<BoxDim>(2.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    // setup a system where both MD and MPCD particles are in each of the cells
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.type_mapping.push_back("A");
        pdata_snap.type_mapping.push_back("B");
        pdata_snap.resize(8);
        pdata_snap.pos[0] = scale(vec3<Scalar>(-0.5, -0.5, -0.5), ref_box, box);
        pdata_snap.pos[1] = scale(vec3<Scalar>(0.5, -0.5, -0.5), ref_box, box);
        pdata_snap.pos[2] = scale(vec3<Scalar>(-0.5, 0.5, -0.5), ref_box, box);
        pdata_snap.pos[3] = scale(vec3<Scalar>(0.5, 0.5, -0.5), ref_box, box);
        pdata_snap.pos[4] = scale(vec3<Scalar>(-0.5, -0.5, 0.5), ref_box, box);
        pdata_snap.pos[5] = scale(vec3<Scalar>(0.5, -0.5, 0.5), ref_box, box);
        pdata_snap.pos[6] = scale(vec3<Scalar>(-0.5, 0.5, 0.5), ref_box, box);
        pdata_snap.pos[7] = scale(vec3<Scalar>(0.5, 0.5, 0.5), ref_box, box);
        pdata_snap.type[0] = 0;
        pdata_snap.type[1] = 1;
        pdata_snap.type[2] = 0;
        pdata_snap.type[3] = 1;
        pdata_snap.type[4] = 0;
        pdata_snap.type[5] = 1;
        pdata_snap.type[6] = 0;
        pdata_snap.type[7] = 1;
        }
    snap->mpcd_data.resize(8);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-0.5, -0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(0.5, -0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-0.5, 0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(0.5, 0.5, -0.5), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-0.5, -0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(0.5, -0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-0.5, 0.5, 0.5), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(0.5, 0.5, 0.5), ref_box, box);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_8 = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(2, 2, 2), false));
    cl->compute(0);

        // at first, there is no embedded particle, so everything should just look like the test
        // before
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 1);

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[1].w), ci(1, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[2].w), ci(0, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[3].w), ci(1, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[4].w), ci(0, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[5].w), ci(1, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[6].w), ci(0, 1, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[7].w), ci(1, 1, 1));
        }

    // now we include the half embedded group
    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleFilter> selector_B(new ParticleFilterType({"B"}));
    std::shared_ptr<ParticleGroup> group_B(new ParticleGroup(sysdef, selector_B));

    cl->setEmbeddedGroup(group_B);
    cl->compute(1);

        // now there should be a second particle in the cell
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 2);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 2);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 2);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 2);

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[1].w), ci(1, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[2].w), ci(0, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[3].w), ci(1, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[4].w), ci(0, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[5].w), ci(1, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[6].w), ci(0, 1, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[7].w), ci(1, 1, 1));

        ArrayHandle<unsigned int> h_embed_cell_ids(cl->getEmbeddedGroupCellIds(),
                                                   access_location::host,
                                                   access_mode::read);
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[0], ci(1, 0, 0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[1], ci(1, 1, 0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[2], ci(1, 0, 1));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[3], ci(1, 1, 1));

        // all masses should stil be set to original values
        ArrayHandle<Scalar4> h_embed_vel(embed_pdata->getVelocities(),
                                         access_location::host,
                                         access_mode::read);
        CHECK_CLOSE(h_embed_vel.data[0].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[1].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[2].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[3].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[4].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[5].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[6].w, 1.0, tol);
        CHECK_CLOSE(h_embed_vel.data[7].w, 1.0, tol);
        }

        // pick a particle up and put it in a different cell, now there will be an extra embedded
        // particle
        {
        ArrayHandle<Scalar4> h_embed_pos(embed_pdata->getPositions(),
                                         access_location::host,
                                         access_mode::overwrite);
        h_embed_pos.data[1] = scale(make_scalar4(0.5, 0.5, -0.5, __int_as_scalar(1)), ref_box, box);
        }
    cl->compute(2);
        // now there should be a second particle in the cell
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);

        // validate that each cell has the right number
        Index3D ci = cl->getCellIndexer();
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 0, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(0, 1, 1)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 0)], 1);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 0, 1)], 2);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 0)], 3);
        CHECK_EQUAL_UINT(h_cell_np.data[ci(1, 1, 1)], 2);

        ArrayHandle<Scalar4> h_vel(pdata_8->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[1].w), ci(1, 0, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[2].w), ci(0, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[3].w), ci(1, 1, 0));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[4].w), ci(0, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[5].w), ci(1, 0, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[6].w), ci(0, 1, 1));
        CHECK_EQUAL_UINT(__scalar_as_int(h_vel.data[7].w), ci(1, 1, 1));

        ArrayHandle<unsigned int> h_embed_cell_ids(cl->getEmbeddedGroupCellIds(),
                                                   access_location::host,
                                                   access_mode::read);
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[0], ci(1, 1, 0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[1], ci(1, 1, 0));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[2], ci(1, 0, 1));
        CHECK_EQUAL_UINT(h_embed_cell_ids.data[3], ci(1, 1, 1));

        // all masses should stil be set to original values
        ArrayHandle<Scalar4> h_embed_vel(embed_pdata->getVelocities(),
                                         access_location::host,
                                         access_mode::read);
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

//! Test for correct calculation of cell thermo properties for MPCD particles
template<class CL>
void celllist_thermo_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                const Scalar3& L,
                                const Scalar3& tilt)
    {
    auto ref_box = std::make_shared<BoxDim>(2.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
    snap->particle_data.type_mapping.push_back("A");
    // place each particle in a different cell, doubling the first cell
    snap->mpcd_data.resize(5);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.5, 0.5, 0.5);
    snap->mpcd_data.position[4] = vec3<Scalar>(-0.5, 0.5, 0.5);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
    snap->mpcd_data.velocity[4] = vec3<Scalar>(1.0, -1.0, 4.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_5 = sysdef->getMPCDParticleData();

    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(2, 2, 2), false));
    AllThermoRequest thermo_req(cl);
    cl->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_avg_vel(cl->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);
        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 2.0, tol);
        // energy, temperature (relative to COM), np
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 2.5, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)], 2.0 * 0.5 * 0.5 / 3.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 2.0, tol);
        // energy, temperature (relative to COM), np
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)], 17.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(1, 1, 1)], 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)], 9.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 1, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 1)], 1);
        }

    // Check the net stats of the system
    CHECK_CLOSE(cl->getNetMomentum().x, 4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, -4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, -1.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 28.5, tol);
    CHECK_CLOSE(cl->getTemperature(),
                0.5 * (2.0 * 0.5 * 0.5 / 3.0 + 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0),
                tol);

    // increase the mass and make sure that energies depend on mass, but velocities don't
    pdata_5->setMass(4.0);
    cl->compute(1);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_avg_vel(cl->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);

        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 4.0 * 2.5, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)], 4.0 * 2.0 * 0.5 * 0.5 / 3.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)], 4.0 * 17.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(1, 1, 1)], 4.0 * 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 4.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)], 4.0 * 9.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 1, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 1)], 1);
        }

    // Check the net stats of the system
    CHECK_CLOSE(cl->getNetMomentum().x, 4.0 * 4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, 4.0 * -4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, 4.0 * -1.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 4.0 * 28.5, tol);
    CHECK_CLOSE(cl->getTemperature(),
                4.0 * 0.5 * (2.0 * 0.5 * 0.5 / 3.0 + 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0),
                tol);

    // switch a particle into a different cell, and make sure the DOF are reduced accordingly
    pdata_5->setMass(1.0);
        {
        ArrayHandle<Scalar4> h_pos(pdata_5->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        h_pos.data[2] = make_scalar4(-0.5, -0.5, -0.5, 0.0);
        }
    cl->compute(2);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_avg_vel(cl->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);

        // Three particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 7.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)],
                    (2 * 1.0 * 1.0 + 2 * 1.0 * 1.0 + 2.0 * 2.0) / 6.0,
                    tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 3);

        // One particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 1.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)], 12.5, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(1, 1, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 1);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)], 9.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 1, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 1)], 1);
        }

    // Check the net stats of the system, only average temperature should change now
    CHECK_CLOSE(cl->getNetMomentum().x, 4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, -4.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, -1.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 28.5, tol);
    CHECK_CLOSE(cl->getTemperature(), (2 * 1.0 * 1.0 + 2 * 1.0 * 1.0 + 2.0 * 2.0) / 6.0, tol);
    }

//! Test for correct calculation of cell thermo properties with embedded particles
template<class CL>
void celllist_thermo_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                const Scalar3& L,
                                const Scalar3& tilt)
    {
    auto ref_box = std::make_shared<BoxDim>(2.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.type_mapping.push_back("A");
        pdata_snap.resize(4);

        pdata_snap.pos[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        pdata_snap.pos[1] = vec3<Scalar>(-0.5, -0.5, 0.5);
        pdata_snap.pos[2] = vec3<Scalar>(0.5, -0.5, 0.5);
        pdata_snap.pos[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        pdata_snap.mass[0] = 3.0;
        pdata_snap.mass[1] = 2.0;
        pdata_snap.mass[2] = 4.0;
        pdata_snap.mass[3] = 5.0;

        pdata_snap.vel[0] = vec3<Scalar>(-2.0, 0.0, 0.0);
        pdata_snap.vel[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        pdata_snap.vel[2] = vec3<Scalar>(0.0, -3.0, 0.0);
        pdata_snap.vel[3] = vec3<Scalar>(0.0, 0.0, -5.0);
        }
    snap->mpcd_data.resize(4);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.5, -0.5, 0.5);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.5, -0.5, 0.5);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleFilter> selector(new ParticleFilterAll());
    std::shared_ptr<ParticleGroup> group(new ParticleGroup(sysdef, selector));

    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);
    cl->setEmbeddedGroup(group);
    AllThermoRequest thermo_req(cl);
    cl->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_avg_vel(cl->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);

        // Cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 4.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 8.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)], 4.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 2);

        // Cell (0,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 1)], 1.5, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 1)], 2);

        // Cell (1,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].y, -3.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].w, 5.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(1, 0, 1)], 22.5, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(1, 0, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 0, 1)], 2);

        // Cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 6.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)], 75.0, tol);
        CHECK_CLOSE(h_cell_temp.data[ci(1, 1, 1)], 0.0, tol);
        UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 2);
        }

    // Check the net stats of the system
    CHECK_CLOSE(cl->getNetMomentum().x, -1.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, -15.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, -30.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 107.0, tol);
    CHECK_CLOSE(cl->getTemperature(), (4.0 + 0.0 + 0.0 + 0.0) / 4., tol);
    }

//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_list_dimensions)
    {
    celllist_dimension_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 8.0, 10.0),
        make_scalar3(0, 0, 0));
    }

//! dimension test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_dimensions_noncubic)
    {
    celllist_dimension_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(8.5, 10.1, 5.9),
        make_scalar3(0, 0, 0));
    }

//! dimension test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_dimensions_triclinic)
    {
    celllist_dimension_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 8.0, 10.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! small system test case for MPCD CellList class
UP_TEST(mpcd_cell_list_small_test)
    {
    celllist_small_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }

//! small system test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_small_test_noncubic)
    {
    celllist_small_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(3.1, 4.2, 5.3),
        make_scalar3(0, 0, 0));
    }

//! small system test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_small_test_triclinic)
    {
    celllist_small_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! grid shift test case for MPCD CellList class
UP_TEST(mpcd_cell_list_grid_shift_test)
    {
    celllist_grid_shift_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! grid shift test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_grid_shift_test_noncubic)
    {
    celllist_grid_shift_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.1, 7.2, 8.3),
        make_scalar3(0, 0, 0));
    }

//! grid shift test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_grid_shift_test_triclinic)
    {
    celllist_grid_shift_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! embedded particle test case for MPCD CellList class
UP_TEST(mpcd_cell_list_embed_test)
    {
    celllist_embed_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }

//! embedded particle test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_embed_test_noncubic)
    {
    celllist_embed_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(3.1, 4.2, 5.3),
        make_scalar3(0, 0, 0));
    }

//! embedded particle test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_embed_test_triclinic)
    {
    celllist_embed_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! test case for cell list velocity and energy
UP_TEST(mpcd_cell_list_thermo_basic)
    {
    celllist_thermo_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }
UP_TEST(mpcd_cell_list_thermo_embed)
    {
    celllist_thermo_embed_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }

#ifdef ENABLE_HIP
//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions)
    {
    celllist_dimension_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 8.0, 10.0),
        make_scalar3(0, 0, 0));
    }

//! dimension test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_dimensions_noncubic)
    {
    celllist_dimension_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(8.5, 10.1, 5.9),
        make_scalar3(0, 0, 0));
    }

//! dimension test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_gpu_dimensions_triclinic)
    {
    celllist_dimension_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 8.0, 10.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! small system test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_small_test)
    {
    celllist_small_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }

//! small system test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_small_test_noncubic)
    {
    celllist_small_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(3.1, 4.2, 5.3),
        make_scalar3(0, 0, 0));
    }

//! small system test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_small_test_triclinic)
    {
    celllist_small_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! grid shift test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_grid_shift_test)
    {
    celllist_grid_shift_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! grid shift test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_grid_shift_test_noncubic)
    {
    celllist_grid_shift_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.1, 7.2, 8.3),
        make_scalar3(0, 0, 0));
    }

//! grid shift test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_grid_shift_test_triclinic)
    {
    celllist_grid_shift_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! embedded particle test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_embed_test)
    {
    celllist_embed_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }

//! embedded particle test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_embed_test_noncubic)
    {
    celllist_embed_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(3.1, 4.2, 5.3),
        make_scalar3(0, 0, 0));
    }

//! embedded particle test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_embed_test_triclinic)
    {
    celllist_embed_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! test case for cell list velocity and energy
UP_TEST(mpcd_cell_list_thermo_basic_gpu)
    {
    celllist_thermo_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }
UP_TEST(mpcd_cell_list_thermo_embed_gpu)
    {
    celllist_thermo_embed_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(2.0, 2.0, 2.0),
        make_scalar3(0, 0, 0));
    }
#endif // ENABLE_HIP
