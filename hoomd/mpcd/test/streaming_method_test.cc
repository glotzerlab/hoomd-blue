// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/ConfinedStreamingMethod.h"
#include "hoomd/mpcd/StreamingGeometry.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/ConfinedStreamingMethodGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for basic setup and functionality of the streaming method
template<class SM>
void streaming_method_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(10.0);
    snap->particle_data.type_mapping.push_back("A");

    // 2 particle system
    snap->mpcd_data.resize(2);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(1.0, 4.85, 3.0);
    snap->mpcd_data.position[1] = vec3<Scalar>(-3.0, -4.75, -1.0);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(1.0, 1.0, 1.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(-1.0, -1.0, -1.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // setup a streaming method at timestep 2 with period 2 and phase 1
    auto geom = std::make_shared<const mpcd::detail::BulkGeometry>();
    std::shared_ptr<mpcd::StreamingMethod> stream = std::make_shared<SM>(sysdef, 2, 2, 1, geom);
    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    stream->setCellList(cl);

    // set timestep to 0.05, so the MPCD step is 2 x 0.05 = 0.1
    stream->setDeltaT(0.05);
    CHECK_CLOSE(stream->getDeltaT(), 0.1, tol);

    // initially, we should not want to stream and so it shouldn't have any effect
    UP_ASSERT(!stream->peekStream(2));
    stream->stream(2);
    std::shared_ptr<mpcd::ParticleData> pdata_2 = sysdef->getMPCDParticleData();
        {
        ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, 1.0, tol);
        CHECK_CLOSE(h_pos.data[0].y, 4.85, tol);
        CHECK_CLOSE(h_pos.data[0].z, 3.0, tol);

        CHECK_CLOSE(h_pos.data[1].x, -3.0, tol);
        CHECK_CLOSE(h_pos.data[1].y, -4.75, tol);
        CHECK_CLOSE(h_pos.data[1].z, -1.0, tol);
        }

    // now if we peek, we should need to stream at step 3
    UP_ASSERT(stream->peekStream(3));
    stream->stream(3);
        {
        ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, 1.1, tol);
        CHECK_CLOSE(h_pos.data[0].y, 4.95, tol);
        CHECK_CLOSE(h_pos.data[0].z, 3.1, tol);

        CHECK_CLOSE(h_pos.data[1].x, -3.1, tol);
        CHECK_CLOSE(h_pos.data[1].y, -4.85, tol);
        CHECK_CLOSE(h_pos.data[1].z, -1.1, tol);
        }

    // next streaming step should now be off again
    UP_ASSERT(!stream->peekStream(4));
    // streaming on the 5th step should send one particle through the boundary
    UP_ASSERT(stream->peekStream(5));
    stream->stream(5);
        {
        ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, 1.2, tol);
        CHECK_CLOSE(h_pos.data[0].y, -4.95, tol);
        CHECK_CLOSE(h_pos.data[0].z, 3.2, tol);

        CHECK_CLOSE(h_pos.data[1].x, -3.2, tol);
        CHECK_CLOSE(h_pos.data[1].y, -4.95, tol);
        CHECK_CLOSE(h_pos.data[1].z, -1.2, tol);
        }

    // increase the timestep, which should increase distance travelled
    stream->setDeltaT(0.1);
    stream->stream(7);
        {
        ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        CHECK_CLOSE(h_pos.data[0].x, 1.4, tol);
        CHECK_CLOSE(h_pos.data[0].y, -4.75, tol);
        CHECK_CLOSE(h_pos.data[0].z, 3.4, tol);

        CHECK_CLOSE(h_pos.data[1].x, -3.4, tol);
        CHECK_CLOSE(h_pos.data[1].y, 4.85, tol);
        CHECK_CLOSE(h_pos.data[1].z, -1.4, tol);
        }
    }

//! basic test case for MPCD StreamingMethod class
UP_TEST(mpcd_streaming_method_basic)
    {
    typedef mpcd::ConfinedStreamingMethod<mpcd::detail::BulkGeometry> method;
    streaming_method_basic_test<method>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_HIP
//! basic test case for MPCD StreamingMethod class
UP_TEST(mpcd_streaming_method_setup)
    {
    typedef mpcd::ConfinedStreamingMethodGPU<mpcd::detail::BulkGeometry> method;
    streaming_method_basic_test<method>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_HIP
