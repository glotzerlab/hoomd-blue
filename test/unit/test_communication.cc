#ifdef ENABLE_MPI

//! name the boost unit test module
#define BOOST_TEST_MODULE CommunicationTests
#include "boost_utf_configure.h"

#include "ExecutionConfiguration.h"
#include "System.h"

#include <boost/mpi.hpp>
#include <boost/shared_ptr.hpp>

#include "Communicator.h"
#include "MPIInitializer.h"

#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#endif

#include <algorithm>

using namespace boost;

//! MPI environment
boost::mpi::environment *env;

//! MPI communicator
boost::shared_ptr<boost::mpi::communicator> world;

//! Typedef for function that creates the Communnicator on the CPU or GPU
typedef boost::function<shared_ptr<Communicator> (shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<boost::mpi::communicator> mpi_comm,
                                                  shared_ptr<MPIInitializer> mpi_init)> communicator_creator;

shared_ptr<Communicator> base_class_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<boost::mpi::communicator> mpi_comm,
                                                         shared_ptr<MPIInitializer> mpi_init);

shared_ptr<Communicator> gpu_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<boost::mpi::communicator> mpi_comm,
                                                  shared_ptr<MPIInitializer> mpi_init);


#ifdef ENABLE_CUDA
//! Excution Configuration for GPU
/* For MPI libraries that directly support CUDA, it is required that
   CUDA be initialized before setting up the MPI environmnet. This
   global variable stores the ExecutionConfiguration for GPU, which is
   initialized once
*/
boost::shared_ptr<ExecutionConfiguration> exec_conf_gpu;
#endif

//! Execution configuration on the CPU
boost::shared_ptr<ExecutionConfiguration> exec_conf_cpu;

void test_mpi_initializer(boost::shared_ptr<ExecutionConfiguration> exec_conf)
{
    // this test needs to be run on eight processors
    if (world->size() != 8)
        {
        std::cerr << "***Error! This test needs to be run on 8 processors.\n" << endl << endl;
        throw std::runtime_error("Error setting up unit test");
        }

    // create a system with eight particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    // initialize a 2x2x2 domain decomposition on processor with rank 0
    boost::shared_ptr<MPIInitializer> mpi_init(new MPIInitializer(sysdef,
                                                                  world,
                                                                  0));

    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());
    pdata->setMPICommunicator(world);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        // set up eight particles, one in every domain
        h_pos.data[0].x = -0.5;
        h_pos.data[0].y = -0.5;
        h_pos.data[0].z = -0.5;

        h_pos.data[1].x = 0.5;
        h_pos.data[1].y = -0.5;
        h_pos.data[1].z = -0.5;

        h_pos.data[2].x = -0.5;
        h_pos.data[2].y = 0.5;
        h_pos.data[2].z = -0.5;

        h_pos.data[3].x = 0.5;
        h_pos.data[3].y = 0.5;
        h_pos.data[3].z = -0.5;

        h_pos.data[4].x = -0.5;
        h_pos.data[4].y = -0.5;
        h_pos.data[4].z = 0.5;

        h_pos.data[5].x = 0.5;
        h_pos.data[5].y = -0.5;
        h_pos.data[5].z = 0.5;

        h_pos.data[6].x = -0.5;
        h_pos.data[6].y = 0.5;
        h_pos.data[6].z = 0.5;

        h_pos.data[7].x = 0.5;
        h_pos.data[7].y = 0.5;
        h_pos.data[7].z = 0.5;
        }

    // distribute particle data on processors
    mpi_init->scatter(0);

    // check that every domain has exactly one particle
    BOOST_CHECK_EQUAL(pdata->getN(), 1);

    // check that every particle ended up in the domain to where it belongs
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), 0);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), 2);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), 3);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), 4);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), 5);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), 6);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), 7);

    // check that the positions have been transferred correctly
    Scalar3 pos = pdata->getPosition(0);
    BOOST_CHECK_CLOSE(pos.x, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(1);
    BOOST_CHECK_CLOSE(pos.x, 0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(2);
    BOOST_CHECK_CLOSE(pos.x, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(3);
    BOOST_CHECK_CLOSE(pos.x,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(4);
    BOOST_CHECK_CLOSE(pos.x, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.5, tol_small);

    pos = pdata->getPosition(5);
    BOOST_CHECK_CLOSE(pos.x,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.5, tol_small);

    pos = pdata->getPosition(6);
    BOOST_CHECK_CLOSE(pos.x, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.5, tol_small);

    pos = pdata->getPosition(7);
    BOOST_CHECK_CLOSE(pos.x,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.5, tol_small);
    }

//! Test particle migration of Communicator
void test_communicator_migrate(communicator_creator comm_creator, shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this test needs to be run on eight processors
    if (world->size() != 8)
        {
        std::cerr << "***Error! This test needs to be run on 8 processors.\n" << endl << endl;
        throw std::runtime_error("Error setting up unit test");
        }

    // create a system with eight particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    // initialize a 2x2x2 domain decomposition on processor with rank 0
    boost::shared_ptr<MPIInitializer> mpi_init(new MPIInitializer(sysdef,
                                                                  world,
                                                                  0));

    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());
    pdata->setMPICommunicator(world);

    boost::shared_ptr<Communicator> comm = comm_creator(sysdef, world, mpi_init);

        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        // set up eight particles, one in every domain
        h_pos.data[0].x = -0.5;
        h_pos.data[0].y = -0.5;
        h_pos.data[0].z = -0.5;

        h_pos.data[1].x = 0.5;
        h_pos.data[1].y = -0.5;
        h_pos.data[1].z = -0.5;

        h_pos.data[2].x = -0.5;
        h_pos.data[2].y = 0.5;
        h_pos.data[2].z = -0.5;

        h_pos.data[3].x = 0.5;
        h_pos.data[3].y = 0.5;
        h_pos.data[3].z = -0.5;

        h_pos.data[4].x = -0.5;
        h_pos.data[4].y = -0.5;
        h_pos.data[4].z = 0.5;

        h_pos.data[5].x = 0.5;
        h_pos.data[5].y = -0.5;
        h_pos.data[5].z = 0.5;

        h_pos.data[6].x = -0.5;
        h_pos.data[6].y = 0.5;
        h_pos.data[6].z = 0.5;

        h_pos.data[7].x = 0.5;
        h_pos.data[7].y = 0.5;
        h_pos.data[7].z = 0.5;
        }

    // distribute particle data on processors
    mpi_init->scatter(0);

    // migrate atoms
    comm->migrateAtoms();

    // check that every domain has exactly one particle
    BOOST_CHECK_EQUAL(pdata->getN(), 1);

    // check that every particle stayed where it was
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), 0);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), 2);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), 3);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), 4);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), 5);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), 6);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), 7);

    // Now move particle 0 into domain 1
    pdata->setPosition(0, make_scalar3(0.1,-0.5,-0.5));
    // move particle 1 into domain 2
    pdata->setPosition(1, make_scalar3(-0.2, 0.5, -0.5));
    // move particle 2 into domain 3
    pdata->setPosition(2, make_scalar3(0.2, 0.3, -0.5));
    // move particle 3 into domain 4
    pdata->setPosition(3, make_scalar3(-0.5, -0.3, 0.2));
    // move particle 4 into domain 5
    pdata->setPosition(4, make_scalar3(0.1, -0.3, 0.2));
    // move particle 5 into domain 6
    pdata->setPosition(5, make_scalar3(-0.2, 0.4, 0.2));
    // move particle 6 into domain 7
    pdata->setPosition(6, make_scalar3(0.6, 0.1, 0.2));
    // move particle 7 into domain 0
    pdata->setPosition(7, make_scalar3(-0.6, -0.1,- 0.2));

    // migrate atoms
    comm->migrateAtoms();

    // check that every particle has ended up in the right domain
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), 2);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), 3);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), 4);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), 5);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), 6);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), 7);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), 0);

    // check positions
    Scalar3 pos = pdata->getPosition(0);
    BOOST_CHECK_CLOSE(pos.x,  0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(1);
    BOOST_CHECK_CLOSE(pos.x, -0.2, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(2);
    BOOST_CHECK_CLOSE(pos.x,  0.2, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.3, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(3);
    BOOST_CHECK_CLOSE(pos.x,  -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  -0.3, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.2, tol_small);

    pos = pdata->getPosition(4);
    BOOST_CHECK_CLOSE(pos.x,  0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.3, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.2, tol_small);

    pos = pdata->getPosition(5);
    BOOST_CHECK_CLOSE(pos.x, -0.2, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.4, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.2, tol_small);

    pos = pdata->getPosition(6);
    BOOST_CHECK_CLOSE(pos.x,  0.6, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.2, tol_small);

    pos = pdata->getPosition(7);
    BOOST_CHECK_CLOSE(pos.x, -0.6, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.2, tol_small);

    //
    // check that that particles are correctly wrapped across the boundary
    //

    // particle 0 crosses the global boundary in +x direction
    pdata->setPosition(0, make_scalar3(1.1,-0.5,-0.5));
    //  particle 1 crosses the global bounadry in the -x direction
    pdata->setPosition(1, make_scalar3(-1.1, 0.5, -0.5));
    // particle 2 crosses the global boundary in the + y direction
    pdata->setPosition(2, make_scalar3(0.2, 1.3, -0.5));
    // particle 3 crosses the global boundary in the - y direction
    pdata->setPosition(3, make_scalar3(-0.5, -1.5, 0.2));
    // particle 4 crosses the global boundary in the + z direction
    pdata->setPosition(4, make_scalar3(0.1, -0.3, 1.6));
    // particle 5 crosses the global boundary in the + z direction and in the -x direction
    pdata->setPosition(5, make_scalar3(-1.1, 0.4, 1.25));
    // particle 6 crosses the global boundary in the + z direction and in the +x direction
    pdata->setPosition(6, make_scalar3(1.3, 0.1, 1.05));
    // particle 7 crosses the global boundary in the - z direction
    pdata->setPosition(7, make_scalar3(-0.6, -0.1,- 1.5));

    // migrate particles
    comm->migrateAtoms();

    // check number of particles
    switch (world->rank())
        {
        case 0:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 1:
            BOOST_CHECK_EQUAL(pdata->getN(), 2);
            break;
        case 2:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 3:
            BOOST_CHECK_EQUAL(pdata->getN(), 2);
            break;
        case 4:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 5:
            BOOST_CHECK_EQUAL(pdata->getN(), 0);
            break;
        case 6:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 7:
            BOOST_CHECK_EQUAL(pdata->getN(), 0);
            break;
        }

    // check that every particle has ended up in the right domain
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), 0);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), 3);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), 6);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), 3);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), 2);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), 4);

    // check positions (taking into account that particles should have been wrapped)
    pos = pdata->getPosition(0);
    BOOST_CHECK_CLOSE(pos.x,  -0.9, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(1);
    BOOST_CHECK_CLOSE(pos.x, 0.9, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(2);
    BOOST_CHECK_CLOSE(pos.x,  0.2, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  -0.7, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.5, tol_small);

    pos = pdata->getPosition(3);
    BOOST_CHECK_CLOSE(pos.x,  -0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.y,   0.5, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.2, tol_small);

    pos = pdata->getPosition(4);
    BOOST_CHECK_CLOSE(pos.x,  0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.3, tol_small);
    BOOST_CHECK_CLOSE(pos.z, -0.4, tol_small);

    pos = pdata->getPosition(5);
    BOOST_CHECK_CLOSE(pos.x,  0.9, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.4, tol_small);
    BOOST_CHECK_CLOSE(pos.z,-0.75, tol_small);

    pos = pdata->getPosition(6);
    BOOST_CHECK_CLOSE(pos.x, -0.7, tol_small);
    BOOST_CHECK_CLOSE(pos.y,  0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.z,-0.95, tol_small);

    pos = pdata->getPosition(7);
    BOOST_CHECK_CLOSE(pos.x, -0.6, tol_small);
    BOOST_CHECK_CLOSE(pos.y, -0.1, tol_small);
    BOOST_CHECK_CLOSE(pos.z,  0.5, tol_small);
    }

//! Test ghost particle communication
void test_communicator_ghosts(communicator_creator comm_creator, shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this test needs to be run on eight processors
    if (world->size() != 8)
        {
        std::cerr << "***Error! This test needs to be run on 8 processors.\n" << endl << endl;
        throw std::runtime_error("Error setting up unit test");
        }

    // create a system with eight particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(16,          // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    // initialize a 2x2x2 domain decomposition on processor with rank 0
    boost::shared_ptr<MPIInitializer> mpi_init(new MPIInitializer(sysdef,
                                                                  world,
                                                                  0));

    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    boost::shared_ptr<Communicator> comm = comm_creator(sysdef, world, mpi_init);

    // width of ghost layer
    Scalar ghost_layer_width = Scalar(0.1);
    comm->setGhostLayerWidth(ghost_layer_width);

    // Set initial atom positions
    // place one particle in the middle of every box (outside the ghost layer)
    pdata->setPosition(0, make_scalar3(-0.5,-0.5,-0.5));
    pdata->setPosition(1, make_scalar3( 0.5,-0.5,-0.5));
    pdata->setPosition(2, make_scalar3(-0.5, 0.5,-0.5));
    pdata->setPosition(3, make_scalar3( 0.5, 0.5,-0.5));
    pdata->setPosition(4, make_scalar3(-0.5,-0.5, 0.5));
    pdata->setPosition(5, make_scalar3( 0.5,-0.5, 0.5));
    pdata->setPosition(6, make_scalar3(-0.5, 0.5, 0.5));
    pdata->setPosition(7, make_scalar3( 0.5, 0.5, 0.5));

    // place particle 8 in the same box as particle 0 and in the ghost layer of its +x neighbor
    pdata->setPosition(8, make_scalar3(-0.02,-0.5,-0.5));
    // place particle 9 in the same box as particle 0 and in the ghost layer of its +y neighbor
    pdata->setPosition(9, make_scalar3(-0.5,-0.05,-0.5));
    // place particle 10 in the same box as particle 0 and in the ghost layer of its +z neighbor
    pdata->setPosition(10, make_scalar3(-0.5, -0.01,-0.05));
    // place particle 11 in the same box as particle 0 and in the ghost layer of its +x and +y neighbor
    pdata->setPosition(11, make_scalar3(-0.05, -0.03,-0.5));
    // place particle 12 in the same box as particle 0 and in the ghost layer of its +x , +y and +z neighbor
    pdata->setPosition(12, make_scalar3(-0.05, -0.03,-0.001));
    // place particle 13 in the same box as particle 1 and in the ghost layer of its -x neighbor
    pdata->setPosition(13, make_scalar3( 0.05, -0.5, -0.5));
    // place particle 14 in the same box as particle 1 and in the ghost layer of its -x neighbor and its +y neighbor
    pdata->setPosition(14, make_scalar3( 0.01, -0.0123, -0.5));
    // place particle 15 in the same box as particle 1 and in the ghost layer of its -x neighbor, of its +y neighbor, and of its +z neighbor
    pdata->setPosition(15, make_scalar3( 0.01, -0.0123, -0.09));

    // distribute particle data on processors
    mpi_init->scatter(0);

    pdata->setMPICommunicator(world);

    // Check number of particles
    switch (world->rank())
        {
        case 0:
            BOOST_CHECK_EQUAL(pdata->getN(), 6);
            break;
        case 1:
            BOOST_CHECK_EQUAL(pdata->getN(), 4);
            break;
        case 2:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 3:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 4:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 5:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 6:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 7:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        }

    // we should have zero ghosts before the exchange
    BOOST_CHECK_EQUAL(pdata->getNGhosts(),0);

    // exchange ghosts
    comm->exchangeGhosts();

   // check ghost atom numbers and positions
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_global_rtag(pdata->getGlobalRTags(), access_location::host, access_mode::read);
        unsigned int rtag;
        switch (world->rank())
            {
            case 0:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 3);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.05,tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.5,tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5,tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01,tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5,tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);
                break;
            case 1:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 3);

                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                break;
            case 2:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 6);

                rtag = h_global_rtag.data[9];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.05, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,  0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);

                break;
            case 3:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 4);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01,tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5,tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);

                break;
            case 4:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 3);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.05, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);
                break;

            case 5:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 2);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);
                break;

            case 6:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 3);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.05, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);
                break;

            case 7:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 2);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.001, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.0123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.09, tol_small);
                break;
            }
        }

    // migrate atoms
    // this should reset the number of ghost particles
    comm->migrateAtoms();

    BOOST_CHECK_EQUAL(pdata->getNGhosts(),0);

    //
    // check handling of periodic boundary conditions
    //

    // place some atoms in a ghost layer at a global boundary

    // place particle 8 in the same box as particle 0 and in the ghost layer of its -x neighbor and -y neighbor
    pdata->setPosition(8, make_scalar3(-0.02,-0.95,-0.5));
    // place particle 9 in the same box as particle 0 and in the ghost layer of its -y neighbor
    pdata->setPosition(9, make_scalar3(-0.5,-0.96,-0.5));
    // place particle 10 in the same box as particle 0 and in the ghost layer of its +y neighbor and -z neighbor
    pdata->setPosition(10, make_scalar3(-0.5, -0.01,-0.97));
    // place particle 11 in the same box as particle 0 and in the ghost layer of its -x and -y neighbor
    pdata->setPosition(11, make_scalar3(-0.97, -0.99,-0.5));
    // place particle 12 in the same box as particle 0 and in the ghost layer of its -x , -y and -z neighbor
    pdata->setPosition(12, make_scalar3(-0.997, -0.998,-0.999));
    // place particle 13 in the same box as particle 0 and in the ghost layer of its -x neighbor and +y neighbor
    pdata->setPosition(13, make_scalar3( -0.96, -0.005, -0.50));
    // place particle 14 in the same box as particle 7 and in the ghost layer of its +x neighbor and its +y neighbor
    pdata->setPosition(14, make_scalar3( 0.901, .98, 0.50));
    // place particle 15 in the same box as particle 7 and in the ghost layer of its +x neighbor, of its +y neighbor, and of its +z neighbor
    pdata->setPosition(15, make_scalar3( 0.99, 0.999, 0.9999));

    // migrate atoms in their respective boxes
    comm->migrateAtoms();

    // Check number of particles
    switch (world->rank())
        {
        case 0:
            BOOST_CHECK_EQUAL(pdata->getN(), 7);
            break;
        case 1:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 2:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 3:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 4:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 5:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 6:
            BOOST_CHECK_EQUAL(pdata->getN(), 1);
            break;
        case 7:
            BOOST_CHECK_EQUAL(pdata->getN(), 3);
            break;
        }

   // exchange ghosts
   comm->exchangeGhosts();

   // check ghost atom numbers and positions, taking into account that the particles should have been
   // wrapped across the boundaries
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_global_rtag(pdata->getGlobalRTags(), access_location::host, access_mode::read);
        unsigned int rtag;
        switch (world->rank())
            {
            case 0:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 1);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -1.0001, tol_small);
                break;

            case 1:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 5);

                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,  1.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.99, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.003, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.998, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.999, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.04, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.50, tol_small);


                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.99, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.0001, tol_small);
                break;

            case 2:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 7);
                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[9];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.04, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  -0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.97, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.97, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.997, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.002, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.999, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.96, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.50, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.0001, tol_small);
               break;

            case 3:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 5);

                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,  1.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.5, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.003, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 1.002, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.999, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.04, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.50, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.99, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.0001, tol_small);
                break;

            case 4:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 4);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,  1.03, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.997, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.998, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, 1.001, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.099, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -1.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,  0.50, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.9999, tol_small);
                break;

            case 5:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 3);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,1.003, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.998, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.001, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,0.901, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.50, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,0.99, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.9999, tol_small);
                break;

            case 6:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 4);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.5, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.03, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.997, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,1.002, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.001, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.099, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,0.98, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.50, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.9999, tol_small);
                break;

            case 7:
                BOOST_CHECK_EQUAL(pdata->getNGhosts(), 1);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,1.003, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,1.002, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.001, tol_small);
                break;
            }
        }

   //
   // Test ghost updating
   //

   // set some new positions for the ghost particles
   // the ghost particles could have moved anywhere even outside the ghost layers or boxes they were in originally
   //(but usually they should not move further than half the skin length),

   pdata->setPosition(8, make_scalar3(-0.12,-1.05,-0.6));
   pdata->setPosition(9, make_scalar3(-0.03,-1.09,-0.3));
   pdata->setPosition(10, make_scalar3(-0.11,  0.01,-1.02));
   pdata->setPosition(11, make_scalar3(-0.80, -0.92,-0.2));
   pdata->setPosition(12, make_scalar3(-1.02, -1.05,-1.100));
   pdata->setPosition(13, make_scalar3(-0.89,  0.005, -0.99));
   pdata->setPosition(14, make_scalar3( 1.123, 1.321, 0.9));
   pdata->setPosition(15, make_scalar3( 0.6, 1.001, 1.012));

   // update ghosts
   comm->copyGhosts();

   // check ghost positions, taking into account that the particles should have been wrapped across the boundaries
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_global_rtag(pdata->getGlobalRTags(), access_location::host, access_mode::read);
        unsigned int rtag;
        switch (world->rank())
            {
            case 0:
                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -1.4, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.988, tol_small);
                break;

            case 1:
                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.12, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.6, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,  1.20, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, -0.92, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.2, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.98,  tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.100, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.11, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.99, tol_small);


                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.6, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.988, tol_small);
                break;

            case 2:
                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.12, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.6, tol_small);

                rtag = h_global_rtag.data[9];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.03, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.91, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.3, tol_small);

                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.11, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.02, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.80, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.08, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.2, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -1.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -1.100, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.89, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.99, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -1.40, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.988, tol_small);
               break;

            case 3:
                rtag = h_global_rtag.data[8];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -.12, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.6, tol_small);

                rtag = h_global_rtag.data[11];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,  1.20, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  1.08, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, -0.2, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.98, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-1.100, tol_small);

                rtag = h_global_rtag.data[13];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 1.11, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.005, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.99, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, 0.6, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,-0.988, tol_small);
                break;

            case 4:
                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x, -0.11, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,  0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,  0.98, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, 0.90, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.877, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.679, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,  0.90, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.40, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, 1.012, tol_small);
                break;

            case 5:
                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,0.98, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-1.05, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.900, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,1.123, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.679, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.90, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,0.6, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,-0.999, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.012, tol_small);
                break;

            case 6:
                rtag = h_global_rtag.data[10];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.11, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.01, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.98, tol_small);

                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.02, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y, 0.95, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z, 0.90, tol_small);

                rtag = h_global_rtag.data[14];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-0.877, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,1.321, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.90, tol_small);

                rtag = h_global_rtag.data[15];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,-1.40, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,1.001, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,1.012, tol_small);
                break;

            case 7:
                rtag = h_global_rtag.data[12];
                BOOST_CHECK(rtag >= pdata->getN() && rtag < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_CLOSE(h_pos.data[rtag].x,0.980, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].y,0.950, tol_small);
                BOOST_CHECK_CLOSE(h_pos.data[rtag].z,0.900, tol_small);
                break;
            }
        }

   }

//! Test particle communication for covalently bonded ghosts
void test_communicator_bonded_ghosts(communicator_creator comm_creator, shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this test needs to be run on eight processors
    if (world->size() != 8)
        {
        std::cerr << "***Error! This test needs to be run on 8 processors.\n" << endl << endl;
        throw std::runtime_error("Error setting up unit test");
        }

    // create a system with eight particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             1,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    // initialize a 2x2x2 domain decomposition on processor with rank 0
    boost::shared_ptr<MPIInitializer> mpi_init(new MPIInitializer(sysdef,
                                                                  world,
                                                                  0));

    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    boost::shared_ptr<Communicator> comm = comm_creator(sysdef, world, mpi_init);

    // width of ghost layer
    Scalar ghost_layer_width = Scalar(0.1);
    comm->setGhostLayerWidth(ghost_layer_width);

    // Set initial atom positions
    // place one particle in the middle of every box (outside the ghost layer)
    pdata->setPosition(0, make_scalar3(-0.5,-0.5,-0.5));
    pdata->setPosition(1, make_scalar3( 0.5,-0.5,-0.5));
    pdata->setPosition(2, make_scalar3(-0.5, 0.5,-0.5));
    pdata->setPosition(3, make_scalar3( 0.5, 0.5,-0.5));
    pdata->setPosition(4, make_scalar3(-0.5,-0.5, 0.5));
    pdata->setPosition(5, make_scalar3( 0.5,-0.5, 0.5));
    pdata->setPosition(6, make_scalar3(-0.5, 0.5, 0.5));
    pdata->setPosition(7, make_scalar3( 0.5, 0.5, 0.5));

    // now bond these particles together, forming a cube

    boost::shared_ptr<BondData> bdata(sysdef->getBondData());

    bdata->addBond(Bond(0,0,1));  // bond type, tag a, tag b
    bdata->addBond(Bond(0,0,2));
    bdata->addBond(Bond(0,0,4));
    bdata->addBond(Bond(0,1,3));
    bdata->addBond(Bond(0,1,5));
    bdata->addBond(Bond(0,2,3));
    bdata->addBond(Bond(0,2,6));
    bdata->addBond(Bond(0,3,7));
    bdata->addBond(Bond(0,4,5));
    bdata->addBond(Bond(0,4,6));
    bdata->addBond(Bond(0,5,7));
    bdata->addBond(Bond(0,6,7));

    // distribute particle data on processors
    mpi_init->scatter(0);

    pdata->setMPICommunicator(world);

    // we should have zero ghost particles
    BOOST_CHECK_EQUAL(pdata->getNGhosts(),  0);

    // exchange ghost particles
    comm->exchangeGhosts();

    // rebuild bond list
    pdata->notifyParticleSort();

        {
        // all bonds should be complete, every processor should have three bonds
        ArrayHandle<uint2> h_gpu_bondlist(bdata->getGPUBondList(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_bonds(bdata->getNBondsArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(pdata->getGlobalTags(), access_location::host, access_mode::read);

        BOOST_CHECK_EQUAL(h_n_bonds.data[0],3);
        unsigned int pitch = bdata->getGPUBondList().getPitch();

        unsigned int sorted_tags[3];
        sorted_tags[0] = h_tag.data[h_gpu_bondlist.data[0].x];
        sorted_tags[1] = h_tag.data[h_gpu_bondlist.data[pitch].x];
        sorted_tags[2] = h_tag.data[h_gpu_bondlist.data[2*pitch].x];

        std::sort(sorted_tags, sorted_tags + 3);

        // check bond partners
        switch (world->rank())
            {
            case 0:
                BOOST_CHECK_EQUAL(sorted_tags[0], 1);
                BOOST_CHECK_EQUAL(sorted_tags[1], 2);
                BOOST_CHECK_EQUAL(sorted_tags[2], 4);
                break;
            case 1:
                BOOST_CHECK_EQUAL(sorted_tags[0], 0);
                BOOST_CHECK_EQUAL(sorted_tags[1], 3);
                BOOST_CHECK_EQUAL(sorted_tags[2], 5);
                break;
            case 2:
                BOOST_CHECK_EQUAL(sorted_tags[0], 0);
                BOOST_CHECK_EQUAL(sorted_tags[1], 3);
                BOOST_CHECK_EQUAL(sorted_tags[2], 6);
                break;
            case 3:
                BOOST_CHECK_EQUAL(sorted_tags[0], 1);
                BOOST_CHECK_EQUAL(sorted_tags[1], 2);
                BOOST_CHECK_EQUAL(sorted_tags[2], 7);
                break;
            case 4:
                BOOST_CHECK_EQUAL(sorted_tags[0], 0);
                BOOST_CHECK_EQUAL(sorted_tags[1], 5);
                BOOST_CHECK_EQUAL(sorted_tags[2], 6);
                break;
            case 5:
                BOOST_CHECK_EQUAL(sorted_tags[0], 1);
                BOOST_CHECK_EQUAL(sorted_tags[1], 4);
                BOOST_CHECK_EQUAL(sorted_tags[2], 7);
                break;
            case 6:
                BOOST_CHECK_EQUAL(sorted_tags[0], 2);
                BOOST_CHECK_EQUAL(sorted_tags[1], 4);
                BOOST_CHECK_EQUAL(sorted_tags[2], 7);
                break;
            case 7:
                BOOST_CHECK_EQUAL(sorted_tags[0], 3);
                BOOST_CHECK_EQUAL(sorted_tags[1], 5);
                BOOST_CHECK_EQUAL(sorted_tags[2], 6);
                break;
            }
        }
    }

//! Communicator creator for unit tests
shared_ptr<Communicator> base_class_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<boost::mpi::communicator> mpi_comm,
                                                         shared_ptr<MPIInitializer> mpi_init)
    {
    std::vector<unsigned int> neighbor_rank;
    std::vector<bool> is_at_boundary;
    for (unsigned int i = 0; i < 6; i++)
        {
        neighbor_rank.push_back(mpi_init->getNeighborRank(i));
        is_at_boundary.push_back(mpi_init->isAtBoundary(i));
        }

    uint3 dim = make_uint3(mpi_init->getDimension(0),
                         mpi_init->getDimension(1),
                         mpi_init->getDimension(2));

    return shared_ptr<Communicator>(new Communicator(sysdef,mpi_comm, neighbor_rank, is_at_boundary, dim));
    }

shared_ptr<Communicator> gpu_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<boost::mpi::communicator> mpi_comm,
                                                  shared_ptr<MPIInitializer> mpi_init)
    {
    std::vector<unsigned int> neighbor_rank;
    std::vector<bool> is_at_boundary;
    for (unsigned int i = 0; i < 6; i++)
        {
        neighbor_rank.push_back(mpi_init->getNeighborRank(i));
        is_at_boundary.push_back(mpi_init->isAtBoundary(i));
        }

    uint3 dim = make_uint3(mpi_init->getDimension(0),
                         mpi_init->getDimension(1),
                         mpi_init->getDimension(2));

    return shared_ptr<Communicator>(new CommunicatorGPU(sysdef,mpi_comm, neighbor_rank, is_at_boundary, dim) );
    }


//! Fixture to setup and tear down MPI
struct MPISetup
    {
    //! Setup
    MPISetup()
        {
        int argc = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

#ifdef ENABLE_CUDA
        exec_conf_gpu = boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));
#endif
        exec_conf_cpu = boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU));
        env = new boost::mpi::environment(argc,argv);
        world = boost::shared_ptr<boost::mpi::communicator>(new boost::mpi::communicator());
        }

    //! Cleanup
    ~MPISetup()
        {
        delete env;
        }

    };

BOOST_GLOBAL_FIXTURE( MPISetup )

//! Tests MPIInitializer
BOOST_AUTO_TEST_CASE( MPIInitializer_test )
    {
    test_mpi_initializer(exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( communicator_migrate_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2,_3);
    test_communicator_migrate(communicator_creator_base, exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( communicator_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2,_3);
    test_communicator_ghosts(communicator_creator_base, exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( communicator_bonded_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2,_3);
    test_communicator_bonded_ghosts(communicator_creator_base, exec_conf_cpu);
    }



#ifdef ENABLE_CUDA
//! Tests MPIInitializer on GPU
BOOST_AUTO_TEST_CASE( MPIInitializer_test_GPU )
    {
    test_mpi_initializer(exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( communicator_migrate_test_GPU )
    {
    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2,_3);
    test_communicator_migrate(communicator_creator_gpu, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( communicator_ghosts_test_GPU )
    {
    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2,_3);
    test_communicator_ghosts(communicator_creator_gpu, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( communicator_bonded_ghosts_test_GPU )
    {
    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2,_3);
    test_communicator_bonded_ghosts(communicator_creator_gpu, exec_conf_gpu);
    }
#endif

#endif //ENABLE_MPI
