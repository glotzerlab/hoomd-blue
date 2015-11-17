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

#include "CellList.h"
#include "CellListStencil.h"
#include "Initializers.h"

#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#endif

#include <vector>
using namespace std;

#include <boost/shared_ptr.hpp>
using namespace boost;

/*! \file test_cell_list_stencil.cc
    \brief Implements unit tests for CellListStencil
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE CellListStencilTests
#include "boost_utf_configure.h"

//! Test the cell list stencil as cell list, stencil radius, and box sizes change
template <class CL>
void celllist_stencil_basic_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with a simple simulation box size 3
    boost::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(1, BoxDim(3.0), 2, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata = sysdef_3->getParticleData();

    // initialize a cell list and stencil
    boost::shared_ptr<CellList> cl(new CL(sysdef_3));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->compute(0);
    boost::shared_ptr<CellListStencil> cls(new CellListStencil(sysdef_3, cl));
    cls->compute(0);

    // default initialization should be no stencils
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 0);
        }

    vector<Scalar> rstencil(2,1.0);
    cls->setRStencil(rstencil);
    cls->compute(1);
    // stencils should cover the box (27 cells)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 27);
        }

    // update the nominal cell width so that there are only two cells in the box
    cl->setNominalWidth(Scalar(1.5));
    cl->compute(10);
    cls->requestCompute(); // trigger an update due to the cell list resize (neighbor list ordinarily handles this)
    cls->compute(10);
    // stencils should cover the box but not duplicate it (8 cells)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 8);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 8);
        }

    // grow the box size
    pdata->setGlobalBox(BoxDim(5.0));
    cl->setNominalWidth(Scalar(1.0));
    cl->compute(20);
    cls->compute(20);
    // we should still only have 27 cells right now (and the updated box should have triggered the recompute)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 27);
        }

    // update the rstencil and make sure this gets handled correctly
    rstencil[0] = Scalar(0.5);
    rstencil[1] = Scalar(2.0);
    cls->setRStencil(rstencil);
    cls->compute(21);
    // the first type still has 27, the second type has all 125 cells
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
        }

    // deactivate one of the search radiuses
    rstencil[0] = Scalar(-1.0);
    cls->setRStencil(rstencil);
    cls->compute(22);
    // the first type has 0, the second type has all 125 cells
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
        }

    // check with a smaller cell width
    cl->setNominalWidth(Scalar(0.5));
    cl->compute(30);
    rstencil[0] = Scalar(1.5);
    rstencil[1] = Scalar(1.0);
    cls->setRStencil(rstencil);
    cls->compute(30);
    // the first type has fewer than 344 cells (some corners get skipped), the second type has exactly 125 cells
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        BOOST_CHECK(h_nstencil.data[0] < 344);
        BOOST_CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
        }

    // finally, for a small cutoff, verify the distances to the nearby cells
    rstencil[1] = Scalar(0.5);
    cls->setRStencil(rstencil);
    cls->compute(31);
        {
        ArrayHandle<Scalar4> h_stencil(cls->getStencils(), access_location::host, access_mode::read);
        const Index2D& stencil_idx = cls->getStencilIndexer();

        for (unsigned int cur_stencil=0; cur_stencil < 27; ++cur_stencil)
            {
            Scalar4 stencil = h_stencil.data[stencil_idx(cur_stencil,1)];
            int i = __scalar_as_int(stencil.x);
            int j = __scalar_as_int(stencil.y);
            int k = __scalar_as_int(stencil.z);
            Scalar min_dist2 = stencil.w;

            Scalar cell_dist2 = 0.0;
            if (i != 0) cell_dist2 += Scalar(0.25)*(abs(i)-1)*(abs(i)-1);
            if (j != 0) cell_dist2 += Scalar(0.25)*(abs(j)-1)*(abs(j)-1);
            if (k != 0) cell_dist2 += Scalar(0.25)*(abs(k)-1)*(abs(k)-1);

            if (cell_dist2 > 0.0)
                {
                MY_BOOST_CHECK_CLOSE(min_dist2, cell_dist2, tol_small);
                }
            else
                {
                MY_BOOST_CHECK_SMALL(min_dist2, tol_small);
                }
            }
        }
    }

//! test case for cell list stencil on the CPU
BOOST_AUTO_TEST_CASE( CellListStencil_cpu )
    {
    celllist_stencil_basic_test<CellList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for cell list stencil on the GPU
BOOST_AUTO_TEST_CASE( CellListStencil_gpu )
    {
    celllist_stencil_basic_test<CellListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif