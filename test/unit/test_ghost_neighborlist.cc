#ifdef ENABLE_MPI

//! name the boost unit test module
#define BOOST_TEST_MODULE NeighborListGhostTests
#include "boost_utf_configure.h"

#include "ExecutionConfiguration.h"
#include "System.h"

#include <boost/shared_ptr.hpp>

#include "Communicator.h"
#include "DomainDecomposition.h"
#include "CellList.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"

#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#include "CellListGPU.h"
#include "NeighborListGPUBinned.h"
#endif

#include <algorithm>

using namespace boost;

char env_str[] = "MV2_USE_CUDA=1";

//! Typedef for function that creates the Communnicator on the CPU or GPU
typedef boost::function<shared_ptr<Communicator> (shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<DomainDecomposition> decomposition)> communicator_creator;

shared_ptr<Communicator> base_class_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<DomainDecomposition> decomposition);

#ifdef ENABLE_CUDA
shared_ptr<Communicator> gpu_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<DomainDecomposition> decomposition);

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

//! Test ghost particle communication
void test_neighborlist_ghosts(communicator_creator comm_creator, shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this test needs to be run on two processors
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BOOST_REQUIRE_EQUAL(size,2);

    // create a system with eight particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(6,          // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



   boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    // Set initial atom positions
    // place one particle in the middle of each box (outside the ghost layer)
    pdata->setPosition(0, make_scalar3(-0.5, 0.0,0.0));
    pdata->setPosition(1, make_scalar3( 0.5, 0.0,0.0));

    // place ptl 2 in same box as ptl 0 but in -x ghost layer of other box
    pdata->setPosition(2, make_scalar3(-0.95, 0.0,0.0));
    // place ptl 3 in same box as ptl 0 but in -x ghost layer of other box
    pdata->setPosition(3, make_scalar3(-0.05, 0.0,0.0));
    // place ptl 4 in same box as ptl 1 but in +x ghost layer of other box
    pdata->setPosition(4, make_scalar3(0.95, 0.0,0.0));
    // place ptl 5 in same box as ptl 1 but in +x ghost layer of other box
    pdata->setPosition(5, make_scalar3(.05, 0.0,0.0));

    // distribute particle data on processors
    SnapshotParticleData snap(6);
    pdata->takeSnapshot(snap);

    // initialize a 2x1x1 domain decomposition on processor with rank 0
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf,  pdata->getBox().getL(),2,1,1));
    boost::shared_ptr<Communicator> comm = comm_creator(sysdef, decomposition);

    pdata->setDomainDecomposition(decomposition);

    pdata->initializeFromSnapshot(snap);
   
    // every processor should have three particles
    BOOST_CHECK_EQUAL(pdata->getN(), 3);

    // width of ghost layer
    Scalar ghost_layer_width = Scalar(0.1);
    comm->setGhostLayerWidth(ghost_layer_width);

    // replicate ghosts
    comm->exchangeGhosts();

    // each box should have two ghosts
    BOOST_CHECK_EQUAL(pdata->getNGhosts(), 2);

    shared_ptr<CellList> cell_list;
    if (exec_conf->isCUDAEnabled())
        cell_list = shared_ptr<CellList>(new CellListGPU(sysdef));
    else
        cell_list = shared_ptr<CellList>(new CellList(sysdef));

    Scalar r_cut=Scalar(0.25);
    Scalar r_buff=Scalar(0.05);

    shared_ptr<NeighborList> nlist;
    if (exec_conf->isCUDAEnabled())
        nlist = shared_ptr<NeighborList>(new NeighborListGPUBinned(sysdef,r_cut,r_buff,cell_list));
    else
        nlist = shared_ptr<NeighborList>(new NeighborListBinned(sysdef,r_cut,r_buff,cell_list));

    // compute neighbor list
    nlist->compute(0);

    ArrayHandle<unsigned int> nlist_array(nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> nneigh_array(nlist->getNNeighArray(), access_location::host, access_mode::read);

    // no particle should have non-ghost neighbors
    BOOST_CHECK(nneigh_array.data[0] == 0);
    BOOST_CHECK(nneigh_array.data[1] == 0);
    BOOST_CHECK(nneigh_array.data[2] == 0);

    ArrayHandle<unsigned int> ghost_nlist_array(nlist->getGhostNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> ghost_nneigh_array(nlist->getNGhostNeighArray(), access_location::host, access_mode::read);

    const Index2D& nli = nlist->getNListIndexer();

    for (unsigned int tag = 0;  tag < 6; ++tag)
        {
        if (pdata->getOwnerRank(tag) == exec_conf->getRank())
            {
            unsigned int idx = pdata->getRTag(tag);
            // ptl 4 should be ptl 2's ghost neighbor and vice versa
            if (tag == 2)
                {
                BOOST_CHECK_EQUAL(ghost_nneigh_array.data[idx], 1);

                unsigned int idx2=pdata->getRTag(4);
                BOOST_CHECK(idx2 >= pdata->getN() && idx2 < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_EQUAL(ghost_nlist_array.data[nli(idx,0)], idx2);
                }
            if (tag == 4)
                {
                BOOST_CHECK_EQUAL(ghost_nneigh_array.data[idx], 1);

                unsigned int idx2=pdata->getRTag(2);
                BOOST_CHECK(idx2 >= pdata->getN() && idx2 < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_EQUAL(ghost_nlist_array.data[nli(idx,0)], idx2);
                }

            // ptl 5 should be ptl 3's neighbor and vice versa
            if (tag == 3)
                {
                BOOST_CHECK_EQUAL(ghost_nneigh_array.data[idx], 1);

                unsigned int idx2=pdata->getRTag(5);
                BOOST_CHECK(idx2 >= pdata->getN() && idx2 < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_EQUAL(ghost_nlist_array.data[nli(idx,0)], idx2);
                }
            if (tag == 5)
                {
                BOOST_CHECK_EQUAL(ghost_nneigh_array.data[idx], 1);

                unsigned int idx2=pdata->getRTag(3);
                BOOST_CHECK(idx2 >= pdata->getN() && idx2 < pdata->getN()+pdata->getNGhosts());
                BOOST_CHECK_EQUAL(ghost_nlist_array.data[nli(idx,0)], idx2);
                } 
            }
        }
    }

//! Communicator creator for unit tests
shared_ptr<Communicator> base_class_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<DomainDecomposition> decomposition)
    {
    return shared_ptr<Communicator>(new Communicator(sysdef, decomposition) );
    }

#ifdef ENABLE_CUDA
shared_ptr<Communicator> gpu_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<DomainDecomposition> decomposition)
    {
    return shared_ptr<Communicator>(new CommunicatorGPU(sysdef, decomposition) );
    }
#endif

//! Fixture to setup and tear down MPI
struct MPISetup
    {
    //! Setup
    MPISetup()
        {
        int argc = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

#ifdef ENABLE_CUDA
        exec_conf_gpu = boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU, -1, false, false, boost::shared_ptr<Messenger>(), false));
#endif
        exec_conf_cpu = boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU, -1, false, false, boost::shared_ptr<Messenger>(), false));

        int provided;
        #ifdef ENABLE_MPI_CUDA
        putenv(env_str);
        #endif
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

#ifdef ENABLE_CUDA
        exec_conf_gpu->setMPICommunicator(MPI_COMM_WORLD);
#endif
        exec_conf_cpu->setMPICommunicator(MPI_COMM_WORLD);
        }

    //! Cleanup
    ~MPISetup()
        {
        MPI_Finalize();
        }

    };

BOOST_GLOBAL_FIXTURE( MPISetup )

#if 0
//! Tests particle distribution
BOOST_AUTO_TEST_CASE( neighborlist_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts(communicator_creator_base, exec_conf_cpu);
    }
#endif

#ifdef ENABLE_CUDA
//! Tests particle distribution on GPU
BOOST_AUTO_TEST_CASE( neighborlist_hosts_test_GPU )
    {
    communicator_creator communicator_creator_gpu = bind(gpu_communicator_creator, _1, _2);
    test_neighborlist_ghosts(communicator_creator_gpu, exec_conf_gpu);
    }
#endif

#endif //ENABLE_MPI
