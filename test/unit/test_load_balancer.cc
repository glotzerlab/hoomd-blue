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
#include "BalancedDomainDecomposition.h"
#include "LoadBalancer.h"

#define TO_TRICLINIC(v) dest_box.makeCoordinates(ref_box.makeFraction(make_scalar3(v.x,v.y,v.z)))
#define TO_POS4(v) make_scalar4(v.x,v.y,v.z,h_pos.data[rtag].w)
#define FROM_TRICLINIC(v) ref_box.makeCoordinates(dest_box.makeFraction(make_scalar3(v.x,v.y,v.z)))

void test_load_balancer(boost::shared_ptr<ExecutionConfiguration> exec_conf, const BoxDim& dest_box)
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
    std::vector<Scalar> fxs(2), fys(2), fzs(2);
    fxs[0] = Scalar(0.5); fxs[1] = Scalar(0.5);
    fys[0] = Scalar(0.5); fys[1] = Scalar(0.5);
    fzs[0] = Scalar(0.5); fzs[1] = Scalar(0.5);
    boost::shared_ptr<BalancedDomainDecomposition> decomposition(new BalancedDomainDecomposition(exec_conf, pdata->getBox().getL(), fxs, fys, fzs));
    boost::shared_ptr<Communicator> comm(new Communicator(sysdef, decomposition));
    pdata->setDomainDecomposition(decomposition);

    pdata->initializeFromSnapshot(snap);

    boost::shared_ptr<LoadBalancer> lb(new LoadBalancer(sysdef,decomposition));
    lb->setCommunicator(comm);
    
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

    for (unsigned int t=0; t < 10; ++t)
        {
        lb->update(t);
        }
    BOOST_CHECK_EQUAL(pdata->getN(), 1);
    
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

    for (unsigned int t=10; t < 30; ++t)
        {
        lb->update(t);
        }
    BOOST_CHECK_EQUAL(pdata->getN(), 1);
    
    // pathological case, everything is collapsed to a point (what happens??)
    pdata->setPosition(0, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(1, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(2, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(3, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(4, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(5, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(6, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    pdata->setPosition(7, TO_TRICLINIC(make_scalar3(0.0,0.0,0.0)),false);
    comm->migrateParticles();
    
    for (unsigned int t=30; t < 50; ++t)
        {
        lb->update(t);
        }
    }


//! Tests particle distribution
BOOST_AUTO_TEST_CASE( LoadBalancer_test )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    test_load_balancer(exec_conf, BoxDim(2.0));
    }

#endif // ENABLE_MPI
