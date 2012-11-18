/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef ENABLE_MPI

//! name the boost unit test module
#define BOOST_TEST_MODULE NeighborListGhostTests

// this has to be included after naming the test module
#include "MPITestSetup.h"

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

//! Typedef for function that creates the Communnicator on the CPU or GPU
typedef boost::function<shared_ptr<Communicator> (shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<DomainDecomposition> decomposition)> communicator_creator;

shared_ptr<Communicator> base_class_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<DomainDecomposition> decomposition);

#ifdef ENABLE_CUDA
shared_ptr<Communicator> gpu_communicator_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<DomainDecomposition> decomposition);
#endif

//! Test that ghost particles are correctly included in the neighborlist
template<class nlist_class>
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

    Scalar r_cut=Scalar(0.25);
    Scalar r_buff=Scalar(0.05);

    shared_ptr<NeighborList> nlist;

    nlist = shared_ptr<NeighborList>(new nlist_class(sysdef,r_cut,r_buff));

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

template<class nlist_class>
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
    Scalar r_cut=Scalar(0.2);
    Scalar r_buff=Scalar(0.05);

    shared_ptr<NeighborList> nlist_1, nlist_2;
        {
        nlist_1 = shared_ptr<NeighborList>(new nlist_class(sysdef_1,r_cut,r_buff));
        nlist_2 = shared_ptr<NeighborList>(new nlist_class(sysdef_2,r_cut,r_buff));

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

BOOST_AUTO_TEST_CASE( nsq_neighborlist_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts<NeighborList>(communicator_creator_base, exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( nsq_neighborlist_compare_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare<NeighborList>(communicator_creator_base, exec_conf_cpu);
    } 


BOOST_AUTO_TEST_CASE( neighborlist_ghosts_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts<NeighborListBinned>(communicator_creator_base, exec_conf_cpu);
    }

BOOST_AUTO_TEST_CASE( neighborlist_compare_test )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare<NeighborListBinned>(communicator_creator_base, exec_conf_cpu);
    } 

#ifdef ENABLE_CUDA
BOOST_AUTO_TEST_CASE( nsq_neighborlist_ghosts_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts<NeighborListGPU>(communicator_creator_base, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( nsq_neighborlist_compare_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare<NeighborListGPU>(communicator_creator_base, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( neighborlist_ghosts_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_ghosts<NeighborListGPUBinned>(communicator_creator_base, exec_conf_gpu);
    }

BOOST_AUTO_TEST_CASE( neighborlist_compare_test_GPU )
    {
    communicator_creator communicator_creator_base = bind(base_class_communicator_creator, _1, _2);
    test_neighborlist_compare<NeighborListGPUBinned>(communicator_creator_base, exec_conf_gpu);
    } 
#endif

#endif //ENABLE_MPI
