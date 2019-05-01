// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file CellListStencil.cc
    \brief Defines CellListStencil
*/

#include "CellListStencil.h"

namespace py = pybind11;
#include <algorithm>

using namespace std;

/*!
 * \param sysdef System definition
 * \param cl Cell list to pair the stencil with
 */
CellListStencil::CellListStencil(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<CellList> cl)
    : Compute(sysdef), m_cl(cl), m_compute_stencil(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing CellListStencil" << endl;

    m_pdata->getNumTypesChangeSignal().connect<CellListStencil, &CellListStencil::slotTypeChange>(this);
    m_pdata->getBoxChangeSignal().connect<CellListStencil, &CellListStencil::requestCompute>(this);
    m_cl->getCellWidthChangeSignal().connect<CellListStencil, &CellListStencil::requestCompute>(this);

    // Default initialization is no stencil for any type
    m_rstencil = std::vector<Scalar>(m_pdata->getNTypes(), -1.0);

    // allocate initial stencil memory
    GPUArray<unsigned int> n_stencil(m_pdata->getNTypes(), m_exec_conf);
    m_n_stencil.swap(n_stencil);

    GPUArray<Scalar4> stencil(m_pdata->getNTypes(), m_exec_conf);
    m_stencil.swap(stencil);
    }

CellListStencil::~CellListStencil()
    {
    m_exec_conf->msg->notice(5) << "Destroying CellListStencil" << endl;

    m_pdata->getNumTypesChangeSignal().disconnect<CellListStencil, &CellListStencil::slotTypeChange>(this);
    m_pdata->getBoxChangeSignal().disconnect<CellListStencil, &CellListStencil::requestCompute>(this);
    m_cl->getCellWidthChangeSignal().disconnect<CellListStencil, &CellListStencil::requestCompute>(this);
    }

void CellListStencil::compute(unsigned int timestep)
    {
    // guard against unnecessary calls
    if (!shouldCompute(timestep)) return;

    // sanity check that rstencil is correctly sized
    assert(m_rstencil.size() >= m_pdata->getNTypes());

    if (m_prof)
        m_prof->push("Stencil");

    // compute the size of the bins in each dimension so that we know how big each is
    const uint3 dim = m_cl->getDim();
    const Scalar3 cell_size = m_cl->getCellWidth();

    const BoxDim& box = m_pdata->getBox();
    const uchar3 periodic = box.getPeriodic();

    Scalar rstencil_max = *std::max_element(m_rstencil.begin(), m_rstencil.end());
    int3 max_stencil_size = make_int3(static_cast<int>(ceil(rstencil_max / cell_size.x)),
                                      static_cast<int>(ceil(rstencil_max / cell_size.y)),
                                      static_cast<int>(ceil(rstencil_max / cell_size.z)));
    if (m_sysdef->getNDimensions() == 2) max_stencil_size.z = 0;

    // extremely rare: zero interactions, quit without generating stencils
    if (rstencil_max < Scalar(0.0))
        {
        ArrayHandle<unsigned int> h_n_stencil(m_n_stencil, access_location::host, access_mode::overwrite);
        memset((void*)h_n_stencil.data, 0, sizeof(unsigned int)*m_pdata->getNTypes());
        return;
        }

    // compute the maximum number of bins in the stencil
    unsigned int max_n_stencil = (2*max_stencil_size.x+1)*(2*max_stencil_size.y+1)*(2*max_stencil_size.z+1);

    // reallocate the stencil memory if needed
    if (max_n_stencil*m_pdata->getNTypes() > m_stencil.getNumElements())
        {
        m_stencil_idx = Index2D(max_n_stencil, m_pdata->getNTypes());
        GPUArray<Scalar4> stencil(max_n_stencil*m_pdata->getNTypes(), m_exec_conf);
        m_stencil.swap(stencil);
        }

    // the cell in the "middle" of the box (will be used to guard against running over ends or double counting)
    int3 origin = make_int3((dim.x-1)/2, (dim.y-1)/2, (dim.z-1)/2);

    ArrayHandle<Scalar4> h_stencil(m_stencil, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_stencil(m_n_stencil, access_location::host, access_mode::overwrite);
    for (unsigned int cur_type=0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        // compute the maximum distance for inclusion in the neighbor list for this type
        Scalar r_list_max = m_rstencil[cur_type];
        // rare: ignore this totally non-interacting particle
        if (r_list_max <= Scalar(0.0))
            {
            h_n_stencil.data[cur_type] = 0;
            continue;
            }

        Scalar r_listsq_max = r_list_max*r_list_max;

        // get the stencil size
        int3 stencil_size = make_int3(static_cast<int>(ceil(r_list_max / cell_size.x)),
                                      static_cast<int>(ceil(r_list_max / cell_size.y)),
                                      static_cast<int>(ceil(r_list_max / cell_size.z)));
        if (m_sysdef->getNDimensions() == 2) stencil_size.z = 0;

        // loop through the possible stencils
        // all active stencils must have at least one member -- the current cell
        h_stencil.data[m_stencil_idx(0, cur_type)] = make_scalar4(__int_as_scalar(0), __int_as_scalar(0), __int_as_scalar(0), 0.0);
        unsigned int n_stencil_i = 1;
        for (int k=-stencil_size.z; k <= stencil_size.z; ++k)
            {
            // skip this stencil site if it could take the representative "origin" stencil out of the grid
            // by symmetry of the stencil and because of periodic wrapping. this stops us from double counting
            // cases that would cover the entire grid
            if (periodic.z && ((origin.z + k) < 0 || (origin.z + k) >= (int)dim.z) ) continue;

            for (int j=-stencil_size.y; j <= stencil_size.y; ++j)
                {
                if (periodic.y && ((origin.y + j) < 0 || (origin.y + j) >= (int)dim.y) ) continue;

                for (int i=-stencil_size.x; i <= stencil_size.x; ++i)
                    {
                    if (periodic.z && ((origin.x + i) < 0 || (origin.x + i) >= (int)dim.x) ) continue;

                    // (0,0,0) is always added first
                    if (i == 0 && j == 0 && k == 0) continue;

                    // compute the distance to the closest point in the bin
                    Scalar3 dr = make_scalar3(0.0,0.0,0.0);
                    if (i > 0) dr.x = (i-1) * cell_size.x;
                    else if (i < 0) dr.x = (i+1) * cell_size.x;

                    if (j > 0) dr.y = (j-1) * cell_size.y;
                    else if (j < 0) dr.y = (j+1) * cell_size.y;

                    if (k > 0) dr.z = (k-1) * cell_size.z;
                    else if (k < 0) dr.z = (k+1) * cell_size.z;

                    Scalar dr2 = dot(dr, dr);

                    if (dr2 < r_listsq_max)
                        {
                        h_stencil.data[m_stencil_idx(n_stencil_i, cur_type)] = make_scalar4(__int_as_scalar(i),
                                                                                            __int_as_scalar(j),
                                                                                            __int_as_scalar(k),
                                                                                            dr2);
                        ++n_stencil_i;
                        }
                    }
                }
            }

        assert(n_stencil_i <= max_n_stencil);
        h_n_stencil.data[cur_type] = n_stencil_i;
        }

    if (m_prof)
        m_prof->pop();
    }

bool CellListStencil::shouldCompute(unsigned int timestep)
    {
    if (m_compute_stencil)
        {
        m_compute_stencil = false;
        return true;
        }

    return false;
    }

void export_CellListStencil(py::module& m)
    {
    py::class_<CellListStencil, std::shared_ptr<CellListStencil> >(m,"CellListStencil", py::base<Compute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<CellList> >());
    }
