/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

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
    
    ParticleDataArrays arrays = pdata_3->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(1.0); arrays.y[1] = arrays.z[1] = 0.0;
    arrays.x[2] = Scalar(2.0); arrays.y[2] = arrays.z[2] = 0.0;
    pdata_3->release();
    
    // initialize a cell list
    shared_ptr<CellList> cl(new CL(sysdef_3));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->compute(0);
    
    // verify the dimensions
    Scalar3 width = cl->getWidth();
    MY_BOOST_CHECK_CLOSE(width.x, 1.0f, tol);
    MY_BOOST_CHECK_CLOSE(width.y, 1.0f, tol);
    MY_BOOST_CHECK_CLOSE(width.z, 1.0f, tol);
    
    uint3 dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 10);
    BOOST_CHECK_EQUAL_UINT(dim.y, 10);
    BOOST_CHECK_EQUAL_UINT(dim.z, 10);
    
    // verify the indexers
    Index3D ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 10*10*10);
    
    Index2D cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 10*10*10*cl->getNmax());
    
    Index2D adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 10*10*10*27);
    
    // change the box size and verify the results
    pdata_3->setBox(BoxDim(5.5f));
    cl->compute(0);
    
    // verify the dimensions
    width = cl->getWidth();
    MY_BOOST_CHECK_CLOSE(width.x, 1.1f, tol);
    MY_BOOST_CHECK_CLOSE(width.y, 1.1f, tol);
    MY_BOOST_CHECK_CLOSE(width.z, 1.1f, tol);
    
    dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 5);
    BOOST_CHECK_EQUAL_UINT(dim.y, 5);
    BOOST_CHECK_EQUAL_UINT(dim.z, 5);
    
    // verify the indexers
    ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 5*5*5);
    
    cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 5*5*5*cl->getNmax());
    
    adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 5*5*5*27);
    
    // change the nominal width and verify the reusults
    cl->setNominalWidth(Scalar(0.5));
    cl->compute(0);
    
    // verify the dimensions
    width = cl->getWidth();
    MY_BOOST_CHECK_CLOSE(width.x, 0.5f, tol);
    MY_BOOST_CHECK_CLOSE(width.y, 0.5f, tol);
    MY_BOOST_CHECK_CLOSE(width.z, 0.5f, tol);
    
    dim = cl->getDim();
    BOOST_CHECK_EQUAL_UINT(dim.x, 11);
    BOOST_CHECK_EQUAL_UINT(dim.y, 11);
    BOOST_CHECK_EQUAL_UINT(dim.z, 11);
    
    // verify the indexers
    ci = cl->getCellIndexer();
    BOOST_CHECK_EQUAL_UINT(ci.getNumElements(), 11*11*11);
    
    cli = cl->getCellListIndexer();
    BOOST_CHECK_EQUAL_UINT(cli.getNumElements(), 11*11*11*cl->getNmax());
    
    adji = cl->getCellAdjIndexer();
    BOOST_CHECK_EQUAL_UINT(adji.getNumElements(), 11*11*11*27);
    }

//! boost test case for cell list test on the CPU
BOOST_AUTO_TEST_CASE( CellList_dimension )
    {
    celllist_dimension_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
