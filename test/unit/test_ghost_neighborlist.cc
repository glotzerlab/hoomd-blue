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

//! Test that ghost particles are correctly included in the neighborlist
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
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        cell_list = shared_ptr<CellList>(new CellListGPU(sysdef));
    else
#endif
        cell_list = shared_ptr<CellList>(new CellList(sysdef));

    Scalar r_cut=Scalar(0.25);
    Scalar r_buff=Scalar(0.05);

    shared_ptr<NeighborList> nlist;
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        nlist = shared_ptr<NeighborList>(new NeighborListGPUBinned(sysdef,r_cut,r_buff,cell_list));
    else
#endif
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

void test_neighborlist_compare(communicator_creator comm_creator, shared_ptr<ExecutionConfiguration> exec_conf)
    {
    unsigned int n = 1000;

    // create two identical systems
    shared_ptr<SystemDefinition> sysdef_1(new SystemDefinition(n,          // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));

    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(n,          // number of particles
                                                             BoxDim(2.0), // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));


    boost::shared_ptr<ParticleData> pdata_1(sysdef_1->getParticleData());
    boost::shared_ptr<ParticleData> pdata_2(sysdef_2->getParticleData());

    // initialize domain decomposition on system 1, but not on the other
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf,  pdata_1->getBox().getL()));
    boost::shared_ptr<Communicator> comm = comm_creator(sysdef_1, decomposition);

    pdata_1->setDomainDecomposition(decomposition);

    // initialize both system with same random configuration
    Scalar3 L = pdata_1->getGlobalBox().getL();
    Scalar3 lo = pdata_1->getGlobalBox().getLo();

    SnapshotParticleData snap(n);
    for (unsigned int i = 0; i < n; ++i)
        {
        snap.pos[i] = make_scalar3(lo.x + (Scalar)rand()/(Scalar)RAND_MAX*L.x,
                                   lo.y + (Scalar)rand()/(Scalar)RAND_MAX*L.y,
                                   lo.z + (Scalar)rand()/(Scalar)RAND_MAX*L.z);
        }


    pdata_1->initializeFromSnapshot(snap);
    pdata_2->initializeFromSnapshot(snap);

    // Check that numbers of particles are identical
    BOOST_CHECK_EQUAL(pdata_1->getNGlobal(),pdata_2->getN());

    // width of ghost layer
    Scalar ghost_layer_width = Scalar(0.25);
    comm->setGhostLayerWidth(ghost_layer_width);

    // set up ghost particles in system1
    comm->migrateAtoms();
    comm->exchangeGhosts();

    // Set up cell & neighbor lists for both systems
    shared_ptr<CellList> cell_list_1, cell_list_2;
    cell_list_1 = shared_ptr<CellList>(new CellList(sysdef_1));
    cell_list_2 = shared_ptr<CellList>(new CellList(sysdef_2));

    Scalar r_cut=Scalar(0.2);
    Scalar r_buff=Scalar(0.05);

    shared_ptr<NeighborList> nlist_1, nlist_2;
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        nlist_1 = shared_ptr<NeighborList>(new NeighborListGPUBinned(sysdef_1,r_cut,r_buff,cell_list_1));
        nlist_2 = shared_ptr<NeighborList>(new NeighborListGPUBinned(sysdef_2,r_cut,r_buff,cell_list_2));
        } 
    else
#endif
        {
        nlist_1 = shared_ptr<NeighborList>(new NeighborListBinned(sysdef_1,r_cut,r_buff,cell_list_1));
        nlist_2 = shared_ptr<NeighborList>(new NeighborListBinned(sysdef_2,r_cut,r_buff,cell_list_2));

        // for this test we need NeighborList::StorageMode full
        nlist_1->setStorageMode(NeighborList::full);
        nlist_2->setStorageMode(NeighborList::full);
        }

    // compute neighbor lists
    nlist_1->compute(0);
    nlist_2->compute(0);

    ArrayHandle<unsigned int> nlist_array_1(nlist_1->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> nneigh_array_1(nlist_1->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> ghost_nlist_array_1(nlist_1->getGhostNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> ghost_nneigh_array_1(nlist_1->getNGhostNeighArray(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> nlist_array_2(nlist_2->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> nneigh_array_2(nlist_2->getNNeighArray(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_tag_1(pdata_1->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag_2(pdata_2->getTags(), access_location::host, access_mode::read);

    const Index2D& nli_1 = nlist_1->getNListIndexer();
    const Index2D& nli_2 = nlist_2->getNListIndexer();

    // compare neighbor lists
    for (unsigned int tag = 0; tag < n; ++tag)
        {
        if (pdata_1->getOwnerRank(tag) == exec_conf->getRank())
            {
            // we own that particle
            unsigned int idx_1 = pdata_1->getRTag(tag);
            unsigned idx_2 = pdata_2->getRTag(tag);

            // check that this particle has the same number of neighbors (ghost + regular) in both systems
            unsigned int nneigh_1 = nneigh_array_1.data[idx_1] + ghost_nneigh_array_1.data[idx_1];
            unsigned int nneigh_2 = nneigh_array_2.data[idx_2];
            BOOST_CHECK_EQUAL(nneigh_1, nneigh_2);

            // for all neighbors of particle idx_2 in system 2
            for (unsigned int j = 0; j < nneigh_2; ++j)
                {
                unsigned int neigh_idx_2 = nlist_array_2.data[nli_2(idx_2,j)];
                unsigned int neigh_tag = h_tag_2.data[neigh_idx_2];
                unsigned int neigh_idx_1 = pdata_1->getRTag(neigh_tag);

                // check that it occurs either as a ghost or regular neighbor of idx_1 in system 1
                BOOST_CHECK(neigh_idx_1 < pdata_1->getN() + pdata_1->getNGhosts());

                bool found = false;
                for (unsigned int k = 0; k < nneigh_array_1.data[idx_1]; ++k)
                    if (nlist_array_1.data[nli_1(idx_1, k)] == neigh_idx_1)
                        {
                        found = true;
                        break;
                        }

                for (unsigned int k = 0; k < ghost_nneigh_array_1.data[idx_1]; ++k)
                    if (ghost_nlist_array_1.data[nli_1(idx_1, k)] == neigh_idx_1)
                        {
                        found = true;
                        break;
                        }

                BOOST_CHECK(found);
                }

            // for all neighbors of particle idx_1 in system 1
            for (unsigned int j = 0; j < nneigh_array_1.data[idx_1]; ++j)
                {
                unsigned int neigh_idx_1 = nlist_array_1.data[nli_1(idx_1, j)];
                unsigned int neigh_tag = h_tag_1.data[neigh_idx_1];
                unsigned int neigh_idx_2 = pdata_2->getRTag(neigh_tag);

                bool found = false;
                //check that it is also a neighbor of ptl idx_2 in system 2
                for (unsigned int k = 0; k < nneigh_array_2.data[idx_2]; ++k)
                    if (nlist_array_2.data[nli_2(idx_2, k)] == neigh_idx_2)
                        {
                        found = true;
                        break;
                        }

                BOOST_CHECK(found);
                }

            // for all ghost neighbors of particle idx_1 in system 1
            for (unsigned int j = 0; j < ghost_nneigh_array_1.data[idx_1]; ++j)
                {
                unsigned int neigh_idx_1 = ghost_nlist_array_1.data[nli_1(idx_1, j)];
                unsigned int neigh_tag = h_tag_1.data[neigh_idx_1];
                unsigned int neigh_idx_2 = pdata_2->getRTag(neigh_tag);

                bool found = false;
                //check that it is also a neighbor of ptl idx_2 in system 2
                for (unsigned int k = 0; k < nneigh_array_2.data[idx_2]; ++k)
                    if (nlist_array_2.data[nli_2(idx_2, k)] == neigh_idx_2)
                        {
                        found = true;
                        break;
                        }

                BOOST_CHECK(found);
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
        exec_conf_gpu->setCUDAErrorChecking(true);
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
//! Tests particle distribution
BOOST_AUTO_TEST_CASE( neighborlist_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts(communicator_creator_base, exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( neighborlist_compare_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare(communicator_creator_base, exec_conf_cpu);
    } 

#ifdef ENABLE_CUDA
//! Tests particle distribution on GPU
BOOST_AUTO_TEST_CASE( neighborlist_ghosts_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts(communicator_creator_base, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( neighborlist_compare_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare(communicator_creator_base, exec_conf_gpu);
    } 
#endif

#endif //ENABLE_MPI
