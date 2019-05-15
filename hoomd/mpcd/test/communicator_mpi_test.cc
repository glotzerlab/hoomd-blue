// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifdef ENABLE_MPI

#include "hoomd/mpcd/Communicator.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/CommunicatorGPU.h"
#endif

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/SystemDefinition.h"

#include <functional>
using namespace std;
using namespace std::placeholders;

//! Typedef for function that creates the Communnicator on the CPU or GPU
typedef std::function<std::shared_ptr<mpcd::Communicator>(std::shared_ptr<mpcd::SystemData> mpcd_sys,
                                                          unsigned int nstages)> communicator_creator;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN()

// some convenience macros for casting triclinic boxes into a cubic reference frame
#define REF_TO_DEST(v) dest_box.makeCoordinates(ref_box.makeFraction(make_scalar3(v.x,v.y,v.z)))
#define DEST_TO_REF(v) ref_box.makeCoordinates(dest_box.makeFraction(make_scalar3(v.x,v.y,v.z)))

//! Test particle migration of Communicator
void test_communicator_migrate(communicator_creator comm_creator, std::shared_ptr<ExecutionConfiguration> exec_conf,
    BoxDim dest_box, unsigned int nstages)
    {
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(exec_conf->getHOOMDWorldMPICommunicator(), &size);
    UP_ASSERT_EQUAL(size,8);

    // default initialize an empty snapshot in the reference box
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = dest_box;
    snap->particle_data.type_mapping.push_back("A");
    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, snap->global_box.getL(),2,2,2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));

    // place eight mpcd particles
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
    BoxDim ref_box = BoxDim(2.0);
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
        mpcd_snap->position[0] = vec3<Scalar>(REF_TO_DEST(make_scalar3(-0.5,-0.5,-0.5)));
        mpcd_snap->position[1] = vec3<Scalar>(REF_TO_DEST(make_scalar3( 0.5,-0.5,-0.5)));
        mpcd_snap->position[2] = vec3<Scalar>(REF_TO_DEST(make_scalar3(-0.5, 0.5,-0.5)));
        mpcd_snap->position[3] = vec3<Scalar>(REF_TO_DEST(make_scalar3( 0.5, 0.5,-0.5)));
        mpcd_snap->position[4] = vec3<Scalar>(REF_TO_DEST(make_scalar3(-0.5,-0.5, 0.5)));
        mpcd_snap->position[5] = vec3<Scalar>(REF_TO_DEST(make_scalar3( 0.5,-0.5, 0.5)));
        mpcd_snap->position[6] = vec3<Scalar>(REF_TO_DEST(make_scalar3(-0.5, 0.5, 0.5)));
        mpcd_snap->position[7] = vec3<Scalar>(REF_TO_DEST(make_scalar3( 0.5, 0.5, 0.5)));

        mpcd_snap->velocity[0] = vec3<Scalar>(0., -0.5, 0.5);
        mpcd_snap->velocity[1] = vec3<Scalar>(1., -1.5, 1.5);
        mpcd_snap->velocity[2] = vec3<Scalar>(2., -2.5, 2.5);
        mpcd_snap->velocity[3] = vec3<Scalar>(3., -3.5, 3.5);
        mpcd_snap->velocity[4] = vec3<Scalar>(4., -4.5, 4.5);
        mpcd_snap->velocity[5] = vec3<Scalar>(5., -5.5, 5.5);
        mpcd_snap->velocity[6] = vec3<Scalar>(6., -6.5, 6.5);
        mpcd_snap->velocity[7] = vec3<Scalar>(7., -7.5, 7.5);

        mpcd_snap->type[0] = 0;
        mpcd_snap->type[1] = 1;
        mpcd_snap->type[2] = 2;
        mpcd_snap->type[3] = 3;
        mpcd_snap->type[4] = 4;
        mpcd_snap->type[5] = 5;
        mpcd_snap->type[6] = 6;
        mpcd_snap->type[7] = 7;
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    // set a small cell size so that nothing will lie in the diffusion layer
    mpcd_sys->getCellList()->setCellSize(0.05);

    // initialize the communicator
    std::shared_ptr<mpcd::Communicator> comm = comm_creator(mpcd_sys, nstages);

    // check that all particles were initialized onto their proper ranks
    std::shared_ptr<mpcd::ParticleData> pdata = mpcd_sys->getParticleData();

    // each rank should own one particle, in tag order, and should have everyone in the right place
    UP_ASSERT_EQUAL(pdata->getNGlobal(), 8);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    const unsigned int my_rank = exec_conf->getRank();

    // verify all particles were initialized correctly
        {
        const unsigned int tag = pdata->getTag(0);
        const Scalar3 pos = DEST_TO_REF(pdata->getPosition(0));
        const Scalar3 vel = pdata->getVelocity(0);
        const unsigned int type = pdata->getType(0);

        switch(my_rank)
            {
            case 0:
                UP_ASSERT_EQUAL(tag, 0);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 0., tol); CHECK_CLOSE(vel.y, -0.5, tol); CHECK_CLOSE(vel.z, 0.5, tol);
                UP_ASSERT_EQUAL(type, 0);
                break;
            case 1:
                UP_ASSERT_EQUAL(tag, 1);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 1., tol); CHECK_CLOSE(vel.y, -1.5, tol); CHECK_CLOSE(vel.z, 1.5, tol);
                UP_ASSERT_EQUAL(type, 1);
                break;
            case 2:
                UP_ASSERT_EQUAL(tag, 2);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 2., tol); CHECK_CLOSE(vel.y, -2.5, tol); CHECK_CLOSE(vel.z, 2.5, tol);
                UP_ASSERT_EQUAL(type, 2);
                break;
            case 3:
                UP_ASSERT_EQUAL(tag, 3);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 3., tol); CHECK_CLOSE(vel.y, -3.5, tol); CHECK_CLOSE(vel.z, 3.5, tol);
                UP_ASSERT_EQUAL(type, 3);
                break;
            case 4:
                UP_ASSERT_EQUAL(tag, 4);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 4., tol); CHECK_CLOSE(vel.y, -4.5, tol); CHECK_CLOSE(vel.z, 4.5, tol);
                UP_ASSERT_EQUAL(type, 4);
                break;
            case 5:
                UP_ASSERT_EQUAL(tag, 5);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 5., tol); CHECK_CLOSE(vel.y, -5.5, tol); CHECK_CLOSE(vel.z, 5.5, tol);
                UP_ASSERT_EQUAL(type, 5);
                break;
            case 6:
                UP_ASSERT_EQUAL(tag, 6);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 6., tol); CHECK_CLOSE(vel.y, -6.5, tol); CHECK_CLOSE(vel.z, 6.5, tol);
                UP_ASSERT_EQUAL(type, 6);
                break;
            case 7:
                UP_ASSERT_EQUAL(tag, 7);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 7., tol); CHECK_CLOSE(vel.y, -7.5, tol); CHECK_CLOSE(vel.z, 7.5, tol);
                UP_ASSERT_EQUAL(type, 7);
                break;
            };
        }

    // attempt a migration, everyone should stay in place
    comm->migrateParticles(0);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
        {
        const unsigned int tag = pdata->getTag(0);
        const Scalar3 pos = DEST_TO_REF(pdata->getPosition(0));
        const Scalar3 vel = pdata->getVelocity(0);
        const unsigned int type = pdata->getType(0);

        switch(my_rank)
            {
            case 0:
                UP_ASSERT_EQUAL(tag, 0);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 0., tol); CHECK_CLOSE(vel.y, -0.5, tol); CHECK_CLOSE(vel.z, 0.5, tol);
                UP_ASSERT_EQUAL(type, 0);
                break;
            case 1:
                UP_ASSERT_EQUAL(tag, 1);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 1., tol); CHECK_CLOSE(vel.y, -1.5, tol); CHECK_CLOSE(vel.z, 1.5, tol);
                UP_ASSERT_EQUAL(type, 1);
                break;
            case 2:
                UP_ASSERT_EQUAL(tag, 2);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 2., tol); CHECK_CLOSE(vel.y, -2.5, tol); CHECK_CLOSE(vel.z, 2.5, tol);
                UP_ASSERT_EQUAL(type, 2);
                break;
            case 3:
                UP_ASSERT_EQUAL(tag, 3);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 3., tol); CHECK_CLOSE(vel.y, -3.5, tol); CHECK_CLOSE(vel.z, 3.5, tol);
                UP_ASSERT_EQUAL(type, 3);
                break;
            case 4:
                UP_ASSERT_EQUAL(tag, 4);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 4., tol); CHECK_CLOSE(vel.y, -4.5, tol); CHECK_CLOSE(vel.z, 4.5, tol);
                UP_ASSERT_EQUAL(type, 4);
                break;
            case 5:
                UP_ASSERT_EQUAL(tag, 5);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 5., tol); CHECK_CLOSE(vel.y, -5.5, tol); CHECK_CLOSE(vel.z, 5.5, tol);
                UP_ASSERT_EQUAL(type, 5);
                break;
            case 6:
                UP_ASSERT_EQUAL(tag, 6);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 6., tol); CHECK_CLOSE(vel.y, -6.5, tol); CHECK_CLOSE(vel.z, 6.5, tol);
                UP_ASSERT_EQUAL(type, 6);
                break;
            case 7:
                UP_ASSERT_EQUAL(tag, 7);
                CHECK_CLOSE(pos.x, 0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 7., tol); CHECK_CLOSE(vel.y, -7.5, tol); CHECK_CLOSE(vel.z, 7.5, tol);
                UP_ASSERT_EQUAL(type, 7);
                break;
            };
        }

    // move particles to new ranks
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        Scalar3 new_pos;
        switch(my_rank)
            {
            case 0:
                // move particle 0 into domain 1
                new_pos = REF_TO_DEST(make_scalar3(0.1,-0.5,-0.5));
                break;
            case 1:
                // move particle 1 into domain 2
                new_pos = REF_TO_DEST(make_scalar3(-0.2, 0.5, -0.5));
                break;
            case 2:
                // move particle 2 into domain 3
                new_pos = REF_TO_DEST(make_scalar3(0.2, 0.3, -0.5));
                break;
            case 3:
                // move particle 3 into domain 4
                new_pos = REF_TO_DEST(make_scalar3(-0.5, -0.3, 0.2));
                break;
            case 4:
                // move particle 4 into domain 5
                new_pos = REF_TO_DEST(make_scalar3(0.1, -0.3, 0.2));
                break;
            case 5:
                // move particle 5 into domain 6
                new_pos = REF_TO_DEST(make_scalar3(-0.2, 0.4, 0.2));
                break;
            case 6:
                // move particle 6 into domain 7
                new_pos = REF_TO_DEST(make_scalar3(0.6, 0.1, 0.2));
                break;
            case 7:
                // move particle 7 into domain 0
                new_pos = REF_TO_DEST(make_scalar3(-0.6, -0.1, -0.2));
                break;
            };
        h_pos.data[0].x = new_pos.x;
        h_pos.data[0].y = new_pos.y;
        h_pos.data[0].z = new_pos.z;
        }

    // migrate to new domains
    comm->migrateParticles(1);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
        {
        const unsigned int tag = pdata->getTag(0);
        const Scalar3 pos = DEST_TO_REF(pdata->getPosition(0));
        const Scalar3 vel = pdata->getVelocity(0);
        const unsigned int type = pdata->getType(0);

        switch(my_rank)
            {
            case 0:
                UP_ASSERT_EQUAL(tag, 7);
                CHECK_CLOSE(pos.x, -0.6, tol); CHECK_CLOSE(pos.y, -0.1, tol); CHECK_CLOSE(pos.z, -0.2, tol);
                CHECK_CLOSE(vel.x, 7., tol); CHECK_CLOSE(vel.y, -7.5, tol); CHECK_CLOSE(vel.z, 7.5, tol);
                UP_ASSERT_EQUAL(type, 7);
                break;
            case 1:
                UP_ASSERT_EQUAL(tag, 0);
                CHECK_CLOSE(pos.x, 0.1, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 0., tol); CHECK_CLOSE(vel.y, -0.5, tol); CHECK_CLOSE(vel.z, 0.5, tol);
                UP_ASSERT_EQUAL(type, 0);
                break;
            case 2:
                UP_ASSERT_EQUAL(tag, 1);
                CHECK_CLOSE(pos.x, -0.2, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 1., tol); CHECK_CLOSE(vel.y, -1.5, tol); CHECK_CLOSE(vel.z, 1.5, tol);
                UP_ASSERT_EQUAL(type, 1);
                break;
            case 3:
                UP_ASSERT_EQUAL(tag, 2);
                CHECK_CLOSE(pos.x, 0.2, tol); CHECK_CLOSE(pos.y, 0.3, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 2., tol); CHECK_CLOSE(vel.y, -2.5, tol); CHECK_CLOSE(vel.z, 2.5, tol);
                UP_ASSERT_EQUAL(type, 2);
                break;
            case 4:
                UP_ASSERT_EQUAL(tag, 3);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, -0.3, tol); CHECK_CLOSE(pos.z, 0.2, tol);
                CHECK_CLOSE(vel.x, 3., tol); CHECK_CLOSE(vel.y, -3.5, tol); CHECK_CLOSE(vel.z, 3.5, tol);
                UP_ASSERT_EQUAL(type, 3);
                break;
            case 5:
                UP_ASSERT_EQUAL(tag, 4);
                CHECK_CLOSE(pos.x, 0.1, tol); CHECK_CLOSE(pos.y, -0.3, tol); CHECK_CLOSE(pos.z, 0.2, tol);
                CHECK_CLOSE(vel.x, 4., tol); CHECK_CLOSE(vel.y, -4.5, tol); CHECK_CLOSE(vel.z, 4.5, tol);
                UP_ASSERT_EQUAL(type, 4);
                break;
            case 6:
                UP_ASSERT_EQUAL(tag, 5);
                CHECK_CLOSE(pos.x, -0.2, tol); CHECK_CLOSE(pos.y, 0.4, tol); CHECK_CLOSE(pos.z, 0.2, tol);
                CHECK_CLOSE(vel.x, 5., tol); CHECK_CLOSE(vel.y, -5.5, tol); CHECK_CLOSE(vel.z, 5.5, tol);
                UP_ASSERT_EQUAL(type, 5);
                break;
            case 7:
                UP_ASSERT_EQUAL(tag, 6);
                CHECK_CLOSE(pos.x, 0.6, tol); CHECK_CLOSE(pos.y, 0.1, tol); CHECK_CLOSE(pos.z, 0.2, tol);
                CHECK_CLOSE(vel.x, 6., tol); CHECK_CLOSE(vel.y, -6.5, tol); CHECK_CLOSE(vel.z, 6.5, tol);
                UP_ASSERT_EQUAL(type, 6);
                break;
            };
        }

    // move particles through the global boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        Scalar3 new_pos;
        switch(my_rank)
            {
            case 0:
                // particle 7 crosses the global boundary in the -z direction
                new_pos = REF_TO_DEST(make_scalar3(-0.6, -0.1,- 1.5));
                break;
            case 1:
                // particle 0 crosses the global boundary in +x direction
                new_pos = REF_TO_DEST(make_scalar3(1.1,-0.5,-0.5));
                break;
            case 2:
                // particle 1 crosses the global bounadry in the -x direction
                new_pos = REF_TO_DEST(make_scalar3(-1.1, 0.5, -0.5));
                break;
            case 3:
                // particle 2 crosses the global boundary in the +y direction
                new_pos = REF_TO_DEST(make_scalar3(0.2, 1.3, -0.5));
                break;
            case 4:
                // particle 3 crosses the global boundary in the -y direction
                new_pos = REF_TO_DEST(make_scalar3(-0.5, -1.5, 0.2));
                break;
            case 5:
                // particle 4 crosses the global boundary in the +z direction
                new_pos = REF_TO_DEST(make_scalar3(0.1, -0.3, 1.6));
                break;
            case 6:
                // particle 5 crosses the global boundary in the +z direction and in the -x direction
                new_pos = REF_TO_DEST(make_scalar3(-1.1, 0.4, 1.25));
                break;
            case 7:
                // particle 6 crosses the global boundary in the +z direction and in the +x direction
                new_pos = REF_TO_DEST(make_scalar3(1.3, 0.1, 1.05));
                break;
            };
        h_pos.data[0].x = new_pos.x;
        h_pos.data[0].y = new_pos.y;
        h_pos.data[0].z = new_pos.z;
        }

    // some domains have different numbers of particles after migration
    comm->migrateParticles(2);
        {
        unsigned int tag(0xffffffff), type(0xffffffff);
        Scalar3 pos, vel;

        if (pdata->getN())
            {
            tag = pdata->getTag(0);
            pos = DEST_TO_REF(pdata->getPosition(0));
            vel = pdata->getVelocity(0);
            type = pdata->getType(0);
            }

        switch (my_rank)
            {
            case 0:
                UP_ASSERT_EQUAL(pdata->getN(), 1); // particle 0

                UP_ASSERT_EQUAL(tag, 0);
                CHECK_CLOSE(pos.x,  -0.9, tol); CHECK_CLOSE(pos.y, -0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                CHECK_CLOSE(vel.x, 0., tol); CHECK_CLOSE(vel.y, -0.5, tol); CHECK_CLOSE(vel.z, 0.5, tol);
                UP_ASSERT_EQUAL(type, 0);
                break;
            case 1:
                UP_ASSERT_EQUAL(pdata->getN(), 2); // particles 2 and 4

                // we don't care about particle sorting and we only have 2 particles
                // so we will compare sets using ifs
                UP_ASSERT(tag == 2 || tag == 4);

                if (tag == 2)
                    {
                    UP_ASSERT_EQUAL(tag, 2);
                    CHECK_CLOSE(pos.x, 0.2, tol); CHECK_CLOSE(pos.y, -0.7, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                    CHECK_CLOSE(vel.x, 2., tol); CHECK_CLOSE(vel.y, -2.5, tol); CHECK_CLOSE(vel.z, 2.5, tol);
                    UP_ASSERT_EQUAL(type, 2);

                    // load up the other particle, which must be particle 4
                    tag = pdata->getTag(1);
                    pos = DEST_TO_REF(pdata->getPosition(1));
                    vel = pdata->getVelocity(1);
                    type = pdata->getType(1);

                    UP_ASSERT_EQUAL(tag, 4);
                    CHECK_CLOSE(pos.x,  0.1, tol); CHECK_CLOSE(pos.y, -0.3, tol); CHECK_CLOSE(pos.z, -0.4, tol);
                    CHECK_CLOSE(vel.x, 4., tol); CHECK_CLOSE(vel.y, -4.5, tol); CHECK_CLOSE(vel.z, 4.5, tol);
                    UP_ASSERT_EQUAL(type, 4);
                    }
                else
                    {
                    UP_ASSERT_EQUAL(tag, 4);
                    CHECK_CLOSE(pos.x,  0.1, tol); CHECK_CLOSE(pos.y, -0.3, tol); CHECK_CLOSE(pos.z, -0.4, tol);
                    CHECK_CLOSE(vel.x, 4., tol); CHECK_CLOSE(vel.y, -4.5, tol); CHECK_CLOSE(vel.z, 4.5, tol);
                    UP_ASSERT_EQUAL(type, 4);

                    // load up the other particle, which must be particle 2
                    tag = pdata->getTag(1);
                    pos = DEST_TO_REF(pdata->getPosition(1));
                    vel = pdata->getVelocity(1);
                    type = pdata->getType(1);

                    UP_ASSERT_EQUAL(tag, 2);
                    CHECK_CLOSE(pos.x, 0.2, tol); CHECK_CLOSE(pos.y, -0.7, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                    CHECK_CLOSE(vel.x, 2., tol); CHECK_CLOSE(vel.y, -2.5, tol); CHECK_CLOSE(vel.z, 2.5, tol);
                    UP_ASSERT_EQUAL(type, 2);
                    }
                break;
            case 2:
                UP_ASSERT_EQUAL(pdata->getN(), 1); // particle 6

                UP_ASSERT_EQUAL(tag, 6);
                CHECK_CLOSE(pos.x, -0.7, tol); CHECK_CLOSE(pos.y, 0.1, tol); CHECK_CLOSE(pos.z, -0.95, tol);
                CHECK_CLOSE(vel.x, 6., tol); CHECK_CLOSE(vel.y, -6.5, tol); CHECK_CLOSE(vel.z, 6.5, tol);
                UP_ASSERT_EQUAL(type, 6);
                break;
            case 3:
                UP_ASSERT_EQUAL(pdata->getN(), 2); // particles 1 and 5

                UP_ASSERT(tag == 1 || tag == 5);

                if (tag == 1)
                    {
                    UP_ASSERT_EQUAL(tag, 1);
                    CHECK_CLOSE(pos.x, 0.9, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                    CHECK_CLOSE(vel.x, 1., tol); CHECK_CLOSE(vel.y, -1.5, tol); CHECK_CLOSE(vel.z, 1.5, tol);
                    UP_ASSERT_EQUAL(type, 1);

                    // load up the other particle, which must be particle 5
                    tag = pdata->getTag(1);
                    pos = DEST_TO_REF(pdata->getPosition(1));
                    vel = pdata->getVelocity(1);
                    type = pdata->getType(1);

                    UP_ASSERT_EQUAL(tag, 5);
                    CHECK_CLOSE(pos.x, 0.9, tol); CHECK_CLOSE(pos.y, 0.4, tol); CHECK_CLOSE(pos.z, -0.75, tol);
                    CHECK_CLOSE(vel.x, 5., tol); CHECK_CLOSE(vel.y, -5.5, tol); CHECK_CLOSE(vel.z, 5.5, tol);
                    UP_ASSERT_EQUAL(type, 5);
                    }
                else
                    {
                    UP_ASSERT_EQUAL(tag, 5);
                    CHECK_CLOSE(pos.x, 0.9, tol); CHECK_CLOSE(pos.y, 0.4, tol); CHECK_CLOSE(pos.z, -0.75, tol);
                    CHECK_CLOSE(vel.x, 5., tol); CHECK_CLOSE(vel.y, -5.5, tol); CHECK_CLOSE(vel.z, 5.5, tol);
                    UP_ASSERT_EQUAL(type, 5);

                    // load up the other particle, which must be particle 1
                    tag = pdata->getTag(1);
                    pos = DEST_TO_REF(pdata->getPosition(1));
                    vel = pdata->getVelocity(1);
                    type = pdata->getType(1);

                    UP_ASSERT_EQUAL(tag, 1);
                    CHECK_CLOSE(pos.x, 0.9, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, -0.5, tol);
                    CHECK_CLOSE(vel.x, 1., tol); CHECK_CLOSE(vel.y, -1.5, tol); CHECK_CLOSE(vel.z, 1.5, tol);
                    UP_ASSERT_EQUAL(type, 1);
                    }
                break;
            case 4:
                UP_ASSERT_EQUAL(pdata->getN(), 1); // particle 7

                UP_ASSERT_EQUAL(tag, 7);
                CHECK_CLOSE(pos.x, -0.6, tol); CHECK_CLOSE(pos.y, -0.1, tol); CHECK_CLOSE(pos.z, 0.5, tol);
                CHECK_CLOSE(vel.x, 7., tol); CHECK_CLOSE(vel.y, -7.5, tol); CHECK_CLOSE(vel.z, 7.5, tol);
                UP_ASSERT_EQUAL(type, 7);
                break;
            case 5:
                UP_ASSERT_EQUAL(pdata->getN(), 0);
                break;
            case 6:
                UP_ASSERT_EQUAL(pdata->getN(), 1); // particle 3

                UP_ASSERT_EQUAL(tag, 3);
                CHECK_CLOSE(pos.x, -0.5, tol); CHECK_CLOSE(pos.y, 0.5, tol); CHECK_CLOSE(pos.z, 0.2, tol);
                CHECK_CLOSE(vel.x, 3., tol); CHECK_CLOSE(vel.y, -3.5, tol); CHECK_CLOSE(vel.z, 3.5, tol);
                UP_ASSERT_EQUAL(type, 3);
                break;
            case 7:
                UP_ASSERT_EQUAL(pdata->getN(), 0);
                break;
            }
        }
    }

class MigrateSelectOp
    {
    public:
        MigrateSelectOp(std::shared_ptr<mpcd::Communicator> comm)
            : m_comm(comm)
            {
            if (m_comm)
                m_comm->getMigrateRequestSignal().connect<MigrateSelectOp, &MigrateSelectOp::operator()>(this);
            }

        ~MigrateSelectOp()
            {
            if (m_comm)
                m_comm->getMigrateRequestSignal().disconnect<MigrateSelectOp, &MigrateSelectOp::operator()>(this);
            }

        bool operator()(unsigned int timestep) const
            {
            return !(timestep % 2);
            }

    private:
        std::shared_ptr<mpcd::Communicator> m_comm;
    };

//! Test particle migration of Communicator in orthorhombic box where decomposition is not cubic
void test_communicator_migrate_ortho(communicator_creator comm_creator, std::shared_ptr<ExecutionConfiguration> exec_conf, unsigned int nstages)
    {
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(exec_conf->getHOOMDWorldMPICommunicator(), &size);
    UP_ASSERT_EQUAL(size,8);

    // default initialize an empty snapshot in the reference box
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(4.0, 2.0, 1.0);
    snap->particle_data.type_mapping.push_back("A");
    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, snap->global_box.getL(),4,2,1));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));

    // place eight mpcd particles
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;

        mpcd_snap->resize(8);
        mpcd_snap->position[0] = vec3<Scalar>(-1.5,-0.5,0.0);
        mpcd_snap->position[1] = vec3<Scalar>(-0.5,-0.5,0.0);
        mpcd_snap->position[2] = vec3<Scalar>( 0.5,-0.5,0.0);
        mpcd_snap->position[3] = vec3<Scalar>( 1.5,-0.5,0.0);
        mpcd_snap->position[4] = vec3<Scalar>(-1.5, 0.5,0.0);
        mpcd_snap->position[5] = vec3<Scalar>(-0.5, 0.5,0.0);
        mpcd_snap->position[6] = vec3<Scalar>( 0.5, 0.5,0.0);
        mpcd_snap->position[7] = vec3<Scalar>( 1.5, 0.5,0.0);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    // set a small cell size so that nothing will lie in the diffusion layer
    mpcd_sys->getCellList()->setCellSize(0.05);

    // initialize the communicator
    std::shared_ptr<mpcd::Communicator> comm = comm_creator(mpcd_sys, nstages);
    MigrateSelectOp migrate_op(comm);

    // check that all particles were initialized onto their proper ranks
    std::shared_ptr<mpcd::ParticleData> pdata = mpcd_sys->getParticleData();
    UP_ASSERT_EQUAL(pdata->getNGlobal(), 8);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    UP_ASSERT_EQUAL(pdata->getTag(0), exec_conf->getRank());

    // move particles to new ranks
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        Scalar3 new_pos;
        switch(exec_conf->getRank())
            {
            case 0:
                // move particle 0 into domain 1
                new_pos = make_scalar3(-0.5,-0.5, 0.0);
                break;
            case 1:
                // move particle 1 into domain 2
                new_pos = make_scalar3( 0.5,-0.5, 0.0);
                break;
            case 2:
                // move particle 2 into domain 3
                new_pos = make_scalar3( 1.5,-0.5, 0.0);
                break;
            case 3:
                // move particle 3 into domain 4
                new_pos = make_scalar3(2.1, 0.5, 0.0);
                break;
            case 4:
                // move particle 4 into domain 5
                new_pos = make_scalar3(-0.5, 0.5,0.0);
                break;
            case 5:
                // move particle 5 into domain 6
                new_pos = make_scalar3( 0.5, 0.5,0.0);
                break;
            case 6:
                // move particle 6 into domain 7
                new_pos = make_scalar3( 1.5, 0.5,0.0);
                break;
            case 7:
                // move particle 7 into domain 0
                new_pos = make_scalar3(2.1,-0.5,0.0);
                break;
            };
        h_pos.data[0].x = new_pos.x;
        h_pos.data[0].y = new_pos.y;
        h_pos.data[0].z = new_pos.z;
        }

    comm->communicate(0);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    int ref_tag = ((int)exec_conf->getRank() - 1) % 8;
    if (ref_tag < 0) ref_tag += 8;
    UP_ASSERT_EQUAL(pdata->getTag(0), ref_tag);

    // move all particles onto domains 5 and 6
    const unsigned int rank = exec_conf->getRank();
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        // just get them all in the same place
        // this first set will put tags 7, 0, 3, and 4 on rank 5
        Scalar3 new_pos;
        if (rank == 0 || rank == 1 || rank == 4 || rank == 5)
            {
            new_pos = make_scalar3(-0.5,0.5,0.0);
            }
        else
            {
            new_pos = make_scalar3( 0.5,0.5,0.0);
            }
        h_pos.data[0].x = new_pos.x; h_pos.data[0].y = new_pos.y; h_pos.data[0].z = new_pos.z;
        }
    // first call to communicate should fail since only migrating on even steps
    comm->communicate(1);
    UP_ASSERT_EQUAL(pdata->getN(), 1);
    // but forcing a migration should proceed
    comm->forceMigrate(); comm->communicate(1);
    if (rank == 5 || rank == 6)
        {
        UP_ASSERT_EQUAL(pdata->getN(), 4);
        }
    else
        {
        UP_ASSERT_EQUAL(pdata->getN(), 0);
        }

    // now send multiple particles out from each rank in different directions
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
        if (rank == 5)
            {
            // send one particle to rank 6, rank 4, and rank 0
            h_pos.data[0].x = Scalar(0.5); h_pos.data[0].y = Scalar(0.5); h_pos.data[0].z = Scalar(0.0);
            h_pos.data[1].x = Scalar(-1.5); h_pos.data[1].y = Scalar(0.5); h_pos.data[1].z = Scalar(0.0);
            h_pos.data[3].x = Scalar(-1.5); h_pos.data[3].y = Scalar(-0.5); h_pos.data[3].z = Scalar(0.0);
            }
        else if (rank == 6)
            {
            // send two particles to rank 5, one to rank 7, and to rank 3
            h_pos.data[0].x = Scalar(1.5); h_pos.data[0].y = Scalar(0.5); h_pos.data[0].z = Scalar(0.0);
            h_pos.data[1].x = Scalar(-0.5); h_pos.data[1].y = Scalar(0.5); h_pos.data[1].z = Scalar(0.0);
            h_pos.data[2].x = Scalar(1.5); h_pos.data[2].y = Scalar(-0.5); h_pos.data[2].z = Scalar(0.0);
            h_pos.data[3].x = Scalar(-0.5); h_pos.data[3].y = Scalar(0.5); h_pos.data[3].z = Scalar(0.0);
            }
        }
    comm->communicate(2);
    if (rank == 5)
        {
        UP_ASSERT_EQUAL(pdata->getN(), 3);
        }
    else if (rank == 0 || rank == 3 || rank == 4 || rank == 6 || rank == 7)
        {
        UP_ASSERT_EQUAL(pdata->getN(), 1);
        }
    else
        {
        UP_ASSERT_EQUAL(pdata->getN(), 0);
        }

    // finally, call again and just make sure nobody moved
    comm->forceMigrate(); comm->communicate(3);
    if (rank == 5)
        {
        UP_ASSERT_EQUAL(pdata->getN(), 3);
        }
    else if (rank == 0 || rank == 3 || rank == 4 || rank == 6 || rank == 7)
        {
        UP_ASSERT_EQUAL(pdata->getN(), 1);
        }
    else
        {
        UP_ASSERT_EQUAL(pdata->getN(), 0);
        }
    }

//! Test particle migration of Communicator
void test_communicator_overdecompose(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                     unsigned int nx,
                                     unsigned int ny,
                                     unsigned int nz,
                                     bool should_fail)
    {
    // only run tests on first partition
    if (exec_conf->getPartition() != 0) return;
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), nx*ny*nz);

    // default initialize an empty snapshot in the reference box
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(4.0);
    snap->particle_data.type_mapping.push_back("A");

    auto decomposition = std::make_shared<DomainDecomposition>(exec_conf, snap->global_box.getL(), nx, ny, nz);
    auto sysdef = std::make_shared<SystemDefinition>(snap, exec_conf, decomposition);

    // empty MPCD system
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);

    // initialize the communicator
    auto comm = std::make_shared<mpcd::Communicator>(mpcd_sys);
    if (should_fail)
        {
        UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ comm->communicate(0); });
        }
    else
        {
        comm->communicate(0);
        }

    // make sure test gets run again
    mpcd_sys->getCellList()->setNExtraCells(1);
    if (should_fail)
        {
        UP_ASSERT_EXCEPTION(std::runtime_error, [&]{ comm->communicate(1); });
        }
    else
        {
        comm->communicate(1);
        }
    }


//! Communicator creator for unit tests
/*!
 * \a nstages is ignored because it is meaningless for the CPU base class
 */
std::shared_ptr<mpcd::Communicator> base_class_communicator_creator(std::shared_ptr<mpcd::SystemData> mpcd_sys,
                                                                    unsigned int nstages)
    {
    return std::shared_ptr<mpcd::Communicator>(new mpcd::Communicator(mpcd_sys));
    }

UP_TEST( mpcd_communicator_migrate_test )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU));

    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    // cubic box
    test_communicator_migrate(communicator_creator_base, exec_conf, BoxDim(2.0),3);
    // orthorhombic box
    test_communicator_migrate(communicator_creator_base, exec_conf, BoxDim(1.0,2.0,3.0),3);
    }

UP_TEST( mpcd_communicator_migrate_ortho_test )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU));

    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_communicator_migrate_ortho(communicator_creator_base, exec_conf, 3);
    }

UP_TEST( mpcd_communicator_overdecompose_test )
    {
    // two ranks in any direction
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>(),
                                                                  false,
                                                                  false);
        exec_conf->getMPIConfig()->splitPartitions(2);
        test_communicator_overdecompose(exec_conf, 2, 1, 1, false);
        test_communicator_overdecompose(exec_conf, 1, 2, 1, false);
        test_communicator_overdecompose(exec_conf, 1, 1, 2, false);
        }
    // four ranks in any direction
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>(),
                                                                  false,
                                                                  false);
        exec_conf->getMPIConfig()->splitPartitions(4);
        test_communicator_overdecompose(exec_conf, 4, 1, 1, false);
        test_communicator_overdecompose(exec_conf, 1, 4, 1, false);
        test_communicator_overdecompose(exec_conf, 1, 1, 4, false);
        }
    // eight ranks in any direction
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>(),
                                                                  false,
                                                                  false);
        exec_conf->getMPIConfig()->splitPartitions(8);
        test_communicator_overdecompose(exec_conf, 8, 1, 1, true);
        test_communicator_overdecompose(exec_conf, 1, 8, 1, true);
        test_communicator_overdecompose(exec_conf, 1, 1, 8, true);
        }
    }
#ifdef ENABLE_CUDA
std::shared_ptr<mpcd::Communicator> gpu_communicator_creator(std::shared_ptr<mpcd::SystemData> mpcd_sys,
                                                             unsigned int nstages)
    {
    std::shared_ptr<mpcd::CommunicatorGPU> comm = std::shared_ptr<mpcd::CommunicatorGPU>(new mpcd::CommunicatorGPU(mpcd_sys));
    comm->setMaxStages(nstages);
    return comm;
    }

UP_TEST( mpcd_communicator_migrate_test_GPU_one_stage )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));

    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    // cubic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(2.0),1);
    // orthorhombic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(1.0,2.0,3.0),1);
    }

UP_TEST( mpcd_communicator_migrate_test_GPU_two_stage )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));

    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    // cubic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(2.0),2);
    // orthorhombic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(1.0,2.0,3.0),2);
    }

UP_TEST( mpcd_communicator_migrate_test_GPU_three_stage )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));

    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    // cubic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(2.0),3);
    // orthorhombic box
    test_communicator_migrate(communicator_creator_gpu, exec_conf,BoxDim(1.0,2.0,3.0),3);
    }

UP_TEST( mpcd_communicator_migrate_ortho_test_GPU_one_stage )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));

    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    test_communicator_migrate_ortho(communicator_creator_gpu, exec_conf, 1);
    }

UP_TEST( mpcd_communicator_migrate_ortho_test_GPU_two_stage )
    {
    auto exec_conf = std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));

    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    test_communicator_migrate_ortho(communicator_creator_gpu, exec_conf, 2);
    }
#endif // ENABLE_CUDA
#endif // ENABLE_MPI
