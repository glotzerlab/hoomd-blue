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

using namespace boost;


//! MPI environment
boost::mpi::environment *env;

//! MPI communicator
boost::shared_ptr<boost::mpi::communicator> world;

//! Excution Configuration for GPU
/* For MPI libraries that directly support CUDA, it is required that
   CUDA be initialized before setting up the MPI environmnet. This
   global variable stores the ExecutionConfiguration for GPU, which is
   initialized once
*/
boost::shared_ptr<ExecutionConfiguration> exec_conf_gpu;

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
    }

//! Fixture to setup and tear down MPI
struct MPISetup
    {
    //! Setup
    MPISetup()
        {
        int argc = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        exec_conf_gpu = boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU));
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

#ifdef ENABLE_CUDA
//! Tests MPIInitializer on GPU
BOOST_AUTO_TEST_CASE( MPIInitializer_test_GPU )
    {
    test_mpi_initializer(exec_conf_gpu);
    }
#endif

#endif //ENABLE_MPI
