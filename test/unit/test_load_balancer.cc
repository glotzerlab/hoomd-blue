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

void test_load_balancer(boost::shared_ptr<ExecutionConfiguration> exec_conf)
{
    // this test needs to be run on eight processors
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BOOST_REQUIRE_EQUAL(size,8);

    // create a system with eight particles
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8,           // number of particles
                                                             BoxDim(2.0),        // box dimensions
                                                             1,           // number of particle types
                                                             0,           // number of bond types
                                                             0,           // number of angle types
                                                             0,           // number of dihedral types
                                                             0,           // number of dihedral types
                                                             exec_conf));



    boost::shared_ptr<ParticleData> pdata(sysdef->getParticleData());
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        // set up eight particles, one in every domain
        h_pos.data[0].x = 0.25;
        h_pos.data[0].y = -0.25;
        h_pos.data[0].z = 0.25;

        h_pos.data[1].x = 0.25;
        h_pos.data[1].y = -0.25;
        h_pos.data[1].z = 0.75;

        h_pos.data[2].x = 0.25;
        h_pos.data[2].y = -0.75;
        h_pos.data[2].z = 0.25;

        h_pos.data[3].x = 0.25;
        h_pos.data[3].y = -0.75;
        h_pos.data[3].z = 0.75;

        h_pos.data[4].x = 0.75;
        h_pos.data[4].y = -0.25;
        h_pos.data[4].z = 0.25;

        h_pos.data[5].x = 0.75;
        h_pos.data[5].y = -0.25;
        h_pos.data[5].z = 0.75;

        h_pos.data[6].x = 0.75;
        h_pos.data[6].y = -0.75;
        h_pos.data[6].z = 0.25;

        h_pos.data[7].x = 0.75;
        h_pos.data[7].y = -0.75;
        h_pos.data[7].z = 0.75;
        }

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

    for (unsigned int t=0; t < 6; ++t)
        {
        lb->update(t);
        }
    lb->update(7);
    BOOST_CHECK_EQUAL(pdata->getN(), 1);
    }


//! Tests particle distribution
BOOST_AUTO_TEST_CASE( LoadBalancer_test )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    test_load_balancer(exec_conf);
    }

#endif // ENABLE_MPI
