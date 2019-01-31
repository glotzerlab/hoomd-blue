// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#include "hoomd/CellList.h"
#include "hoomd/CellListStencil.h"
#include "hoomd/Initializers.h"

#ifdef ENABLE_CUDA
#include "hoomd/CellListGPU.h"
#endif

#include <vector>
using namespace std;

#include <memory>

#include "upp11_config.h"

/*! \file test_cell_list_stencil.cc
    \brief Implements unit tests for CellListStencil
    \ingroup unit_tests
*/

HOOMD_UP_MAIN();

//! Test the cell list stencil as cell list, stencil radius, and box sizes change
template <class CL>
void celllist_stencil_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with a simple simulation box size 3
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(1, BoxDim(3.0), 2, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef_3->getParticleData();

    // initialize a cell list and stencil
    std::shared_ptr<CellList> cl(new CL(sysdef_3));
    cl->setNominalWidth(Scalar(1.0));
    cl->setRadius(1);
    cl->compute(0);
    std::shared_ptr<CellListStencil> cls(new CellListStencil(sysdef_3, cl));
    cls->compute(0);

    // default initialization should be no stencils
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 0);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 0);
        }

    vector<Scalar> rstencil(2,1.0);
    cls->setRStencil(rstencil);
    cls->compute(1);
    // stencils should cover the box (27 cells)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 27);
        }

    // update the nominal cell width so that there are only two cells in the box
    cl->setNominalWidth(Scalar(1.5));
    cl->compute(10);
    cls->compute(10);
    // stencils should cover the box but not duplicate it (8 cells)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 8);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 8);
        }

    // grow the box size
    pdata->setGlobalBox(BoxDim(5.0));
    cl->setNominalWidth(Scalar(1.0));
    cl->compute(20);
    cls->compute(20);
    // we should still only have 27 cells right now (and the updated box should have triggered the recompute)
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 27);
        }

    // update the rstencil and make sure this gets handled correctly
    rstencil[0] = Scalar(0.5);
    rstencil[1] = Scalar(2.0);
    cls->setRStencil(rstencil);
    cls->compute(21);
    // the first type still has 27, the second type has all 125 cells
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 27);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
        }

    // deactivate one of the search radiuses
    rstencil[0] = Scalar(-1.0);
    cls->setRStencil(rstencil);
    cls->compute(22);
    // the first type has 0, the second type has all 125 cells
        {
        ArrayHandle<unsigned int> h_nstencil(cls->getStencilSizes(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nstencil.data[0], 0);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
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
        UP_ASSERT(h_nstencil.data[0] < 344);
        CHECK_EQUAL_UINT(h_nstencil.data[1], 125);
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
                MY_CHECK_CLOSE(min_dist2, cell_dist2, tol_small);
                }
            else
                {
                MY_CHECK_SMALL(min_dist2, tol_small);
                }
            }
        }
    }

//! test case for cell list stencil on the CPU
UP_TEST( CellListStencil_cpu )
    {
    celllist_stencil_basic_test<CellList>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for cell list stencil on the GPU
UP_TEST( CellListStencil_gpu )
    {
    celllist_stencil_basic_test<CellListGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
