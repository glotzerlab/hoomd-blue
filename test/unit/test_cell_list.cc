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


#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "CellList.h"
#include "Initializers.h"

#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#endif

#include <math.h>

using namespace std;
using namespace boost;

/*! \file test_cell_list.cc
    \brief Implements unit tests for CellList and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE CellListTests
#include "boost_utf_configure.h"

//! Test the ability of CellList to initialize dimensions
template <class CL>
void celllist_dimension_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with a simple simulation box size 10
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(10.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(1.0); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    h_pos.data[2].x = Scalar(2.0); h_pos.data[2].y = h_pos.data[2].z = 0.0;
    }

    // ********* initialize a cell list *********
    shared_ptr<CellList> cl(new CL(sysdef_3));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->compute(0);

    // verify the dimensions
    uint3 dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 10);
    BOOST_CHECK_EQUAL_UINT(dim.y, 10);
    BOOST_CHECK_EQUAL_UINT(dim.z, 10);

    // verify the indexers
    Index3D ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 10*10*10);
    BOOST_CHECK_EQUAL_UINT(cl->getCellSizeArray().getNumElements(), 10*10*10);

    Index2D cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 10*10*10*cl->getNmax());
    BOOST_CHECK_EQUAL_UINT(cl->getXYZFArray().getNumElements(), 10*10*10*cl->getNmax());

    Index2D adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 10*10*10*27);
    BOOST_CHECK_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 10*10*10*27);

    // ********* change the box size and verify the results *********
    pdata_3->setGlobalBoxL(make_scalar3(5.5f, 5.5f, 5.5f));
    cl->compute(0);

    // verify the dimensions
    dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 5);
    BOOST_CHECK_EQUAL_UINT(dim.y, 5);
    BOOST_CHECK_EQUAL_UINT(dim.z, 5);

    // verify the indexers
    ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 5*5*5);
    BOOST_CHECK_EQUAL_UINT(cl->getCellSizeArray().getNumElements(), 5*5*5);

    cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 5*5*5*cl->getNmax());
    BOOST_CHECK_EQUAL_UINT(cl->getXYZFArray().getNumElements(), 5*5*5*cl->getNmax());

    adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 5*5*5*27);
    BOOST_CHECK_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 5*5*5*27);

    // ********* change the nominal width and verify the reusults *********
    cl->setNominalWidth(Scalar(0.5));
    cl->compute(0);

    // verify the dimensions
    dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 11);
    BOOST_CHECK_EQUAL_UINT(dim.y, 11);
    BOOST_CHECK_EQUAL_UINT(dim.z, 11);

    // verify the indexers
    ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 11*11*11);
    BOOST_CHECK_EQUAL_UINT(cl->getCellSizeArray().getNumElements(), 11*11*11);

    cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 11*11*11*cl->getNmax());
    BOOST_CHECK_EQUAL_UINT(cl->getXYZFArray().getNumElements(), 11*11*11*cl->getNmax());

    adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 11*11*11*27);
    BOOST_CHECK_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 11*11*11*27);

    // ********* change the box size to a non cube and verify the results *********
    cl->setNominalWidth(Scalar(1.0));
    pdata_3->setGlobalBoxL(make_scalar3(5.5f, 3.0f, 10.5f));
    cl->compute(0);

    // verify the dimensions
    dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 5);
    BOOST_CHECK_EQUAL_UINT(dim.y, 3);
    BOOST_CHECK_EQUAL_UINT(dim.z, 10);

    // verify the indexers
    ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 5*3*10);
    BOOST_CHECK_EQUAL_UINT(cl->getCellSizeArray().getNumElements(), 5*3*10);

    cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 5*3*10*cl->getNmax());
    BOOST_CHECK_EQUAL_UINT(cl->getXYZFArray().getNumElements(), 5*3*10*cl->getNmax());

    adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 5*3*10*27);
    BOOST_CHECK_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 5*3*10*27);
    }

//! boost test case for cell list dimension test on the CPU
BOOST_AUTO_TEST_CASE( CellList_dimension )
    {
    celllist_dimension_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Test the ability of CellList to initialize the adj array
template <class CL>
void celllist_adj_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with a simple simulation box size 3
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(3.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(1.0); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    h_pos.data[2].x = Scalar(1.25); h_pos.data[2].y = h_pos.data[2].z = 0.0;
    }

    // ********* initialize a basic cell list *********
    shared_ptr<CellList> cl(new CL(sysdef_3));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->compute(0);

    // verify the indexer
    Index3D ci = cl->getCellIndexer();
    BOOST_REQUIRE_EQUAL_UINT(ci.getNumElements(), 3*3*3);

    Index2D adji = cl->getCellAdjIndexer();
    BOOST_REQUIRE_EQUAL_UINT(adji.getNumElements(), 3*3*3*27);
    BOOST_REQUIRE_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 3*3*3*27);

    // verify all the cell adj values
    // note that in a 3x3x3 box, ALL cells should have adj from 0-26
        {
        ArrayHandle<unsigned int> h_cell_adj(cl->getCellAdjArray(), access_location::host, access_mode::read);

        for (unsigned int cidx = 0; cidx < ci.getNumElements(); cidx++)
            {
            for (unsigned int offset = 0; offset < 27; offset++)
                {
                unsigned int adjidx = h_cell_adj.data[adji(offset, cidx)];
                BOOST_REQUIRE_EQUAL(adjidx, offset);
                }
            }
        }

    // ********** Test adj array with a radius ***********
    // use a 5x5x5 box with radius 2
    cl->setRadius(2);
    pdata_3->setGlobalBoxL(make_scalar3(5.0f, 5.0f, 5.0f));

    cl->compute(0);

    // verify the indexer
    ci = cl->getCellIndexer();
    BOOST_REQUIRE_EQUAL_UINT(ci.getNumElements(), 5*5*5);

    adji = cl->getCellAdjIndexer();
    BOOST_REQUIRE_EQUAL_UINT(adji.getNumElements(), 5*5*5*125);
    BOOST_REQUIRE_EQUAL_UINT(cl->getCellAdjArray().getNumElements(), 5*5*5*125);

    // verify all the cell adj values
    // note that in a 5x5x5 box, ALL cells should have adj from 0-124
        {
        ArrayHandle<unsigned int> h_cell_adj(cl->getCellAdjArray(), access_location::host, access_mode::read);

        for (unsigned int cidx = 0; cidx < ci.getNumElements(); cidx++)
            {
            for (unsigned int offset = 0; offset < 125; offset++)
                {
                unsigned int adjidx = h_cell_adj.data[adji(offset, cidx)];
                BOOST_REQUIRE_EQUAL(adjidx, offset);
                }
            }
        }
    }

//! boost test case for cell list adj test on the CPU
BOOST_AUTO_TEST_CASE( CellList_adj )
    {
    celllist_adj_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Validate that the cell list itself is computed properly
template <class CL>
void celllist_small_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with a simple simulation a non-cubic box
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8, BoxDim(3, 5, 7), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    pdata->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata->setType(0,1);
    pdata->setCharge(0,1.0);
    pdata->setDiameter(0,0.5f);
    pdata->setBody(0,2);

    pdata->setPosition(1,make_scalar3(1.0,0.0,0.0));
    pdata->setType(1,2);
    pdata->setCharge(1,2.0f);
    pdata->setDiameter(1,1.0f);
    pdata->setBody(1,3);

    pdata->setPosition(2, make_scalar3(-1.0,0.0,0.0));
    pdata->setType(2, 3);
    pdata->setCharge(2,3.0f);
    pdata->setDiameter(2,1.5f);
    pdata->setBody(2,0);

    pdata->setPosition(3,make_scalar3(1.0,2.0,0.0));
    pdata->setType(3,0);
    pdata->setCharge(3,4.0f);
    pdata->setDiameter(3,2.0f);
    pdata->setBody(3,1);

    pdata->setPosition(4,make_scalar3(0.25,0.25,0.0));
    pdata->setType(4,1);
    pdata->setCharge(4,5.0f);
    pdata->setDiameter(4,2.5f);
    pdata->setBody(4,2);

    pdata->setPosition(5, make_scalar3(1.25,2.25,0.0));
    pdata->setType(5,2);
    pdata->setCharge(5,6.0f);
    pdata->setDiameter(5,3.0f);
    pdata->setBody(5,3);

    pdata->setPosition(6,make_scalar3(0.25,-2.0,3.0));
    pdata->setType(6,3);
    pdata->setCharge(6,7.0f);
    pdata->setDiameter(6,3.5f);
    pdata->setBody(6,0);

    pdata->setPosition(7,make_scalar3(-0.25,-2.0,-3.0));
    pdata->setType(7,0);
    pdata->setCharge(7,8.0f);
    pdata->setDiameter(7,4.0f);
    pdata->setBody(7,1);

    // ********* initialize a cell list *********
    shared_ptr<CellList> cl(new CL(sysdef));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->setFlagIndex();
    cl->compute(0);

    // verify the indexers
    Index3D ci = cl->getCellIndexer();
    BOOST_REQUIRE_EQUAL_UINT(ci.getNumElements(), 3*5*7);
    BOOST_REQUIRE_EQUAL_UINT(cl->getCellSizeArray().getNumElements(), 3*5*7);

    Index2D cli = cl->getCellListIndexer();
    BOOST_REQUIRE_EQUAL_UINT(cli.getNumElements(), 3*5*7*cl->getNmax());
    BOOST_REQUIRE_EQUAL_UINT(cl->getXYZFArray().getNumElements(), 3*5*7*cl->getNmax());
    BOOST_REQUIRE_EQUAL_UINT(cl->getTDBArray().getNumElements(), 0);

    // verify the cell contents
        {
        Scalar4 val;

        ArrayHandle<unsigned int> h_cell_size(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_xyzf(cl->getXYZFArray(), access_location::host, access_mode::read);

        // verify cell 2,2,3
        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(2,2,3)], 1);
        val = h_xyzf.data[cli(0, ci(2,2,3))];
        MY_BOOST_CHECK_CLOSE(val.x, 1.0f, tol);
        MY_BOOST_CHECK_SMALL(val.y, tol_small);
        MY_BOOST_CHECK_SMALL(val.z, tol_small);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.w), 1);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(0,2,3)], 1);
        val = h_xyzf.data[cli(0, ci(0,2,3))];
        MY_BOOST_CHECK_CLOSE(val.x, -1.0f, tol);
        MY_BOOST_CHECK_SMALL(val.y, tol_small);
        MY_BOOST_CHECK_SMALL(val.z, tol_small);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.w), 2);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,0,6)], 1);
        val = h_xyzf.data[cli(0, ci(1,0,6))];
        MY_BOOST_CHECK_CLOSE(val.x, 0.25f, tol);
        MY_BOOST_CHECK_CLOSE(val.y, -2.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.z, 3.0f, tol_small);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.w), 6);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,0,0)], 1);
        val = h_xyzf.data[cli(0, ci(1,0,0))];
        MY_BOOST_CHECK_CLOSE(val.x, -0.25f, tol);
        MY_BOOST_CHECK_CLOSE(val.y, -2.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.z, -3.0f, tol_small);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.w), 7);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,2,3)], 2);
        for (unsigned int i = 0; i < 2; i++)
            {
            val = h_xyzf.data[cli(i, ci(1,2,3))];

            // particles can be in any order in the cell list
            bool ok = false;
            if (__scalar_as_int(val.w) == 0)
                {
                ok = (fabs(val.x - 0.0f) < tol &&
                      fabs(val.y - 0.0f) < tol &&
                      fabs(val.z - 0.0f) < tol);
                }
            else if (__scalar_as_int(val.w) == 4)
                {
                ok = (fabs(val.x - 0.25f) < tol &&
                      fabs(val.y - 0.25f) < tol &&
                      fabs(val.z - 0.0f) < tol);
                }
            BOOST_CHECK(ok);
            }

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(2,4,3)], 2);
        for (unsigned int i = 0; i < 2; i++)
            {
            val = h_xyzf.data[cli(i, ci(2,4,3))];

            // particles can be in any order in the cell list
            bool ok = false;
            if (__scalar_as_int(val.w) == 3)
                {
                ok = (fabs(val.x - 1.0f) < tol &&
                      fabs(val.y - 2.0f) < tol &&
                      fabs(val.z - 0.0f) < tol);
                }
            else if (__scalar_as_int(val.w) == 5)
                {
                ok = (fabs(val.x - 1.25f) < tol &&
                      fabs(val.y - 2.25f) < tol &&
                      fabs(val.z - 0.0f) < tol);
                }
            BOOST_CHECK(ok);
            }
        }

    // enable charge and TDB options and test that they work properly
    cl->setFlagCharge();
    cl->setComputeTDB(true);
    cl->compute(0);

    // update the indexers
    ci = cl->getCellIndexer();
    cli = cl->getCellListIndexer();

    BOOST_REQUIRE_EQUAL_UINT(cl->getTDBArray().getNumElements(), 3*5*7*cl->getNmax());

        {
        ArrayHandle<unsigned int> h_cell_size(cl->getCellSizeArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_xyzf(cl->getXYZFArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_tdb(cl->getTDBArray(), access_location::host, access_mode::read);

        Scalar4 val;

        // verify cell 2,2,3
        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(2,2,3)], 1);
        val = h_xyzf.data[cli(0, ci(2,2,3))];
        MY_BOOST_CHECK_CLOSE(val.x, 1.0f, tol);
        MY_BOOST_CHECK_SMALL(val.y, tol_small);
        MY_BOOST_CHECK_SMALL(val.z, tol_small);
        MY_BOOST_CHECK_CLOSE(val.w, 2.0f, tol);
        val = h_tdb.data[cli(0, ci(2,2,3))];
        BOOST_CHECK_EQUAL(__scalar_as_int(val.x), 2);
        MY_BOOST_CHECK_CLOSE(val.y, 1.0f, tol);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.z), 3);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(0,2,3)], 1);
        val = h_xyzf.data[cli(0, ci(0,2,3))];
        MY_BOOST_CHECK_CLOSE(val.x, -1.0f, tol);
        MY_BOOST_CHECK_SMALL(val.y, tol_small);
        MY_BOOST_CHECK_SMALL(val.z, tol_small);
        MY_BOOST_CHECK_CLOSE(val.w, 3.0f, tol);
        val = h_tdb.data[cli(0, ci(0,2,3))];
        BOOST_CHECK_EQUAL(__scalar_as_int(val.x), 3);
        MY_BOOST_CHECK_CLOSE(val.y, 1.5f, tol);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.z), 0);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,0,6)], 1);
        val = h_xyzf.data[cli(0, ci(1,0,6))];
        MY_BOOST_CHECK_CLOSE(val.x, 0.25f, tol);
        MY_BOOST_CHECK_CLOSE(val.y, -2.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.z, 3.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.w, 7.0f, tol);
        val = h_tdb.data[cli(0, ci(1,0,6))];
        BOOST_CHECK_EQUAL(__scalar_as_int(val.x), 3);
        MY_BOOST_CHECK_CLOSE(val.y, 3.5f, tol);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.z), 0);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,0,0)], 1);
        val = h_xyzf.data[cli(0, ci(1,0,0))];
        MY_BOOST_CHECK_CLOSE(val.x, -0.25f, tol);
        MY_BOOST_CHECK_CLOSE(val.y, -2.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.z, -3.0f, tol_small);
        MY_BOOST_CHECK_CLOSE(val.w, 8.0f, tol);
        val = h_tdb.data[cli(0, ci(1,0,0))];
        BOOST_CHECK_EQUAL(__scalar_as_int(val.x), 0);
        MY_BOOST_CHECK_CLOSE(val.y, 4.0f, tol);
        BOOST_CHECK_EQUAL(__scalar_as_int(val.z), 1);

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(1,2,3)], 2);
        for (unsigned int i = 0; i < 2; i++)
            {
            val = h_xyzf.data[cli(i, ci(1,2,3))];
            Scalar4 val_tdb = h_tdb.data[cli(i, ci(1,2,3))];

            // particles can be in any order in the cell list
            bool ok = false;
            if (fabs(val.w - 1.0f) < tol)
                {
                ok = (fabs(val.x - 0.0f) < tol &&
                      fabs(val.y - 0.0f) < tol &&
                      fabs(val.z - 0.0f) < tol &&
                      __scalar_as_int(val_tdb.x) == 1 &&
                      fabs(val_tdb.y - 0.5f) < tol &&
                      __scalar_as_int(val_tdb.z) == 2);
                }
            else if (fabs(val.w - 5.0f) < tol)
                {
                ok = (fabs(val.x - 0.25f) < tol &&
                      fabs(val.y - 0.25f) < tol &&
                      fabs(val.z - 0.0f) < tol &&
                      __scalar_as_int(val_tdb.x) == 1 &&
                      fabs(val_tdb.y - 2.5f) < tol &&
                      __scalar_as_int(val_tdb.z) == 2);

                }
            BOOST_CHECK(ok);
            }

        BOOST_REQUIRE_EQUAL_UINT(h_cell_size.data[ci(2,4,3)], 2);
        for (unsigned int i = 0; i < 2; i++)
            {
            val = h_xyzf.data[cli(i, ci(2,4,3))];
            Scalar4 val_tdb = h_tdb.data[cli(i, ci(2,4,3))];

            // particles can be in any order in the cell list
            bool ok = false;
            if (fabs(val.w - 4.0f) < tol)
                {
                ok = (fabs(val.x - 1.0f) < tol &&
                      fabs(val.y - 2.0f) < tol &&
                      fabs(val.z - 0.0f) < tol &&
                      __scalar_as_int(val_tdb.x) == 0 &&
                      fabs(val_tdb.y - 2.0f) < tol &&
                      __scalar_as_int(val_tdb.z) == 1);
                }
            else if (fabs(val.w - 6.0f) < tol)
                {
                ok = (fabs(val.x - 1.25f) < tol &&
                      fabs(val.y - 2.25f) < tol &&
                      fabs(val.z - 0.0f) < tol &&
                      __scalar_as_int(val_tdb.x) == 2 &&
                      fabs(val_tdb.y - 3.0f) < tol &&
                      __scalar_as_int(val_tdb.z) == 3);
                }
            BOOST_CHECK(ok);
            }
        }
    }

//! boost test case for celllist_small_test
BOOST_AUTO_TEST_CASE( CellList_small )
    {
    celllist_small_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for celllist_small_test on the GPU
BOOST_AUTO_TEST_CASE( CellListGPU_small )
    {
    celllist_small_test<CellListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

//! Validate that the cell list itself can be computed for a large system of particles
template <class CL>
void celllist_large_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    unsigned int N = 10000;
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    boost::shared_ptr<SnapshotSystemData> snap;
    snap = rand_init.getSnapshot();
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // ********* initialize a cell list *********
    shared_ptr<CellList> cl(new CL(sysdef));
    cl->setNominalWidth(Scalar(3.0));
    cl->setRadius(1);
    cl->setFlagIndex();
    cl->compute(0);

    // verify that the sum of the cell sizes adds up to N
    ArrayHandle<unsigned int> h_cell_size(cl->getCellSizeArray(), access_location::host, access_mode::read);
    unsigned int total = 0;
    unsigned int ncell = cl->getCellIndexer().getNumElements();
    for (unsigned int cell = 0; cell < ncell; cell++)
        {
        total += h_cell_size.data[cell];
        }

    BOOST_CHECK_EQUAL_UINT(total, N);

    // verify that every particle appears once in the cell list
    vector<bool> present(N);
    for (unsigned int p = 0; p < N; p++)
        present[p] = false;

    Index2D cli = cl->getCellListIndexer();
    ArrayHandle<Scalar4> h_xyzf(cl->getXYZFArray(), access_location::host, access_mode::read);

    for (unsigned int cell = 0; cell < ncell; cell++)
        {
        for (unsigned int offset = 0; offset < h_cell_size.data[cell]; offset++)
            {
            unsigned int p = __scalar_as_int(h_xyzf.data[cli(offset, cell)].w);
            present[p] = true;
            }
        }

    for (unsigned int p = 0; p < N; p++)
        BOOST_CHECK(present[p]);
    }

//! boost test case for celllist_large_test
BOOST_AUTO_TEST_CASE( CellList_large )
    {
    celllist_large_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for celllist_large_test on the GPU
BOOST_AUTO_TEST_CASE( CellListGPU_large )
    {
    celllist_large_test<CellListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
