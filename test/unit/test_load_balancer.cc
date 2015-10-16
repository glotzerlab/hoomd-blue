/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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
#define BOOST_TEST_MODULE LoadBalancerTests

// this has to be included after naming the test module
#include "boost_utf_configure.h"

#include "System.h"

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include "ExecutionConfiguration.h"
#include "Communicator.h"
#include "LoadBalancer.h"
#ifdef ENABLE_CUDA
#include "LoadBalancerGPU.h"
#endif

#define TO_TRICLINIC(v) dest_box.makeCoordinates(ref_box.makeFraction(make_scalar3(v.x,v.y,v.z)))
#define TO_POS4(v) make_scalar4(v.x,v.y,v.z,h_pos.data[rtag].w)
#define FROM_TRICLINIC(v) ref_box.makeCoordinates(dest_box.makeFraction(make_scalar3(v.x,v.y,v.z)))


using namespace std;

template<class LB>
void test_load_balancer_basic(boost::shared_ptr<ExecutionConfiguration> exec_conf, const BoxDim& dest_box)
{
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BOOST_REQUIRE_EQUAL(size,8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             dest_box,        // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.25,-0.25,0.25)),false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.25,-0.25,0.75)),false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.25,-0.75,0.25)),false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.25,-0.75,0.75)),false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.75,-0.25,0.25)),false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.75,-0.25,0.75)),false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.75,-0.75,0.25)),false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.75,-0.75,0.75)),false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs(1), fys(1), fzs(1);
    fxs[0] = Scalar(0.5);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.5);
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    boost::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);

    pdata->initializeFromSnapshot(snap);

    boost::shared_ptr<LoadBalancer> lb(new LB(sysdef,decomposition));
    lb->setCommunicator(comm);
    lb->setMaxIterations(2);
    
    // migrate atoms
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();   
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(1,0,1));

    // adjust the domain boundaries
    for (unsigned int t=0; t < 10; ++t)
        {
        lb->update(t);
        }

    // each rank should own one particle
    BOOST_CHECK_EQUAL(pdata->getN(), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(0,1,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(0,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(0,0,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(1,1,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(1,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(1,0,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(1,0,1));
    
    // flip the particle signs and see if the domains can realign correctly
    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(-0.25,0.25,-0.25)),false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(-0.25,0.25,-0.75)),false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(-0.25,0.75,-0.25)),false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(-0.25,0.75,-0.75)),false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(-0.75,0.25,-0.25)),false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(-0.75,0.25,-0.75)),false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(-0.75,0.75,-0.25)),false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(-0.75,0.75,-0.75)),false);
    comm->migrateParticles();

    for (unsigned int t=10; t < 20; ++t)
        {
        lb->update(t);
        }
    // each rank should own one particle
    BOOST_CHECK_EQUAL(pdata->getN(), 1);
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(1,0,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(1,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(1,1,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(0,0,0));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(0,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(0,1,0));
    }

template<class LB>
void test_load_balancer_multi(boost::shared_ptr<ExecutionConfiguration> exec_conf, const BoxDim& dest_box)
{
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BOOST_REQUIRE_EQUAL(size,8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             dest_box,        // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.1,-0.1,-0.4)),false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.1,-0.2,-0.4)),false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.1,-0.1,0.2)),false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.1,-0.2,0.2)),false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.2,-0.1,0.55)),false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.2,-0.2,0.55)),false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.2,-0.1,0.9)),false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.2,-0.2,0.9)),false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 1x2x4 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs, fys(1), fzs(3);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.25); fzs[1] = Scalar(0.25); fzs[2] = Scalar(0.25);
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    boost::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);

    pdata->initializeFromSnapshot(snap);

    boost::shared_ptr<LoadBalancer> lb(new LB(sysdef,decomposition));
    lb->setCommunicator(comm);
    lb->enableDimension(1, false);
    lb->setMaxIterations(100);

    // migrate atoms and check placement
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(0,0,2));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(0,0,2));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(0,0,3));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(0,0,3));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(0,0,3));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(0,0,3));

    // balance particles along z only
    lb->update(0);
        {
        uint3 grid_pos = decomposition->getGridPos();
        if (grid_pos.y == 0)
            {
            BOOST_CHECK_EQUAL(pdata->getN(), 2);
            }
        else
            {
            BOOST_CHECK_EQUAL(pdata->getN(), 0);
            }

        // check that fractional cuts lie in the right ranges
        vector<Scalar> frac_y = decomposition->getCumulativeFractions(1);
        MY_BOOST_CHECK_CLOSE(frac_y[1], 0.5, tol);
        vector<Scalar> frac_z = decomposition->getCumulativeFractions(2);
        BOOST_CHECK(frac_z[1] > 0.3 && frac_z[1] <= 0.6);
        BOOST_CHECK(frac_z[2] > 0.6 && frac_z[2] <= 0.775);
        BOOST_CHECK(frac_z[3] > 0.775 && frac_z[3] <= 0.95);

        BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(0,0,0));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(0,0,0));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(0,0,1));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(0,0,1));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(0,0,2));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(0,0,2));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(0,0,3));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(0,0,3));
        }

    // turn on balancing along y and check that this balances now
    lb->enableDimension(1, true);
    lb->update(10);
        {
        BOOST_CHECK_EQUAL(pdata->getN(), 1);

        // check that fractional cuts lie in the right ranges
        vector<Scalar> frac_y = decomposition->getCumulativeFractions(1);
        BOOST_CHECK(frac_y[1] > 0.4 && frac_y[1] <= 0.45);

        BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(0,1,0));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(0,0,0));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(0,1,1));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(0,0,1));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(0,1,2));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(0,0,2));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(0,1,3));
        BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(0,0,3));
        }
    }

//! Ghost layer subscriber
struct ghost_layer_width_request
    {
    //! Constructor
    /*!
     * \param r_ghost Ghost layer width
     */
    ghost_layer_width_request(Scalar r_ghost) : m_r_ghost(r_ghost) {}

    //! Get the ghost width layey
    /*!
     * \param type Type index
     * \returns Constant ghost layer width for all types
     */
    Scalar get(unsigned int type)
        {
        return m_r_ghost;
        }
    Scalar m_r_ghost;   //!< Ghost layer width
    };

template<class LB>
void test_load_balancer_ghost(boost::shared_ptr<ExecutionConfiguration> exec_conf, const BoxDim& dest_box)
{
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BOOST_REQUIRE_EQUAL(size,8);

    // create a system with eight particles
    BoxDim ref_box = BoxDim(2.0);
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             dest_box,        // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());

    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.25,-0.25,0.9)),false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.25,-0.25,0.99)),false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.25,-0.75,0.9)),false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.25,-0.75,0.99)),false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.75,-0.25,0.9)),false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.75,-0.25,0.99)),false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.75,-0.75,0.9)),false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.75,-0.75,0.99)),false);

    SnapshotParticleData<Scalar> snap(8);
    pdata->takeSnapshot(snap);

    // initialize a 2x2x2 domain decomposition on processor with rank 0
    std::vector<Scalar> fxs(1), fys(1), fzs(1);
    fxs[0] = Scalar(0.5);
    fys[0] = Scalar(0.5);
    fzs[0] = Scalar(0.5);
    boost::shared_ptr<DomainDecomposition> decomposition(new DomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    boost::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);

    pdata->initializeFromSnapshot(snap);

    boost::shared_ptr<LoadBalancer> lb(new LB(sysdef,decomposition));
    lb->setCommunicator(comm);

    // migrate atoms and check placement
    comm->migrateParticles();
    const Index3D& di = decomposition->getDomainIndexer();
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(1,0,1));

    // add a ghost layer subscriber and exchange ghosts
    ghost_layer_width_request g(Scalar(0.05));
    comm->addGhostLayerWidthRequest(bind(&ghost_layer_width_request::get,g,_1));
    comm->exchangeGhosts();

    for (unsigned int t=0; t < 20; ++t)
        {
        lb->update(t);
        }

    // because of the ghost layer width, you shouldn't be able to get to a domain this small
    uint3 grid_pos = decomposition->getGridPos();
    if (grid_pos.z == 1) // top layer has 2 each because (x,y) balanced out
        {
        BOOST_CHECK_EQUAL(pdata->getN(), 2);
        }
    else // bottom layer has none
        {
        BOOST_CHECK_EQUAL(pdata->getN(), 0);
        }
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(0), di(0,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(1), di(0,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(2), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(3), di(0,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(4), di(1,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(5), di(1,1,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(6), di(1,0,1));
    BOOST_CHECK_EQUAL(pdata->getOwnerRank(7), di(1,0,1));
    }

//! Tests basic particle redistribution
BOOST_AUTO_TEST_CASE( LoadBalancer_test_basic )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_basic<LoadBalancer>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }

//! Tests particle redistribution with multiple domains and specific directions
BOOST_AUTO_TEST_CASE( LoadBalancer_test_multi )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_multi<LoadBalancer>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }

//! Tests particle redistribution with ghost layer width minimum
BOOST_AUTO_TEST_CASE( LoadBalancer_test_ghost )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    // cubic box
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_ghost<LoadBalancer>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }

#ifdef ENABLE_CUDA
//! Tests basic particle redistribution on the GPU
BOOST_AUTO_TEST_CASE( LoadBalancerGPU_test_basic )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_basic<LoadBalancerGPU>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }

//! Tests particle redistribution with multiple domains and specific directions on the GPU
BOOST_AUTO_TEST_CASE( LoadBalancerGPU_test_multi )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_multi<LoadBalancerGPU>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }

//! Tests particle redistribution with ghost layer width minimum
BOOST_AUTO_TEST_CASE( LoadBalancerGPU_test_ghost )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    // cubic box
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(2.0));
    // triclinic box 1
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(1.0,.1,.2,.3));
    // triclinic box 2
    test_load_balancer_ghost<LoadBalancerGPU>(exec_conf, BoxDim(1.0,-.6,.7,.5));
    }
#endif // ENABLE_CUDA

#endif // ENABLE_MPI
