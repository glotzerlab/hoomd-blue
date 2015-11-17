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

// Maintainer: mphoward

/*! \file CellListStencil.cc
    \brief Defines CellListStencil
*/

#include "CellListStencil.h"

#include <boost/python.hpp>
#include <boost/bind.hpp>
#include <algorithm>

using namespace std;
using namespace boost::python;

/*!
 * \param sysdef System definition
 * \param cl Cell list to pair the stencil with
 */
CellListStencil::CellListStencil(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<CellList> cl)
    : Compute(sysdef), m_cl(cl), m_compute_stencil(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing CellListStencil" << endl;

    m_num_type_change_conn = m_pdata->connectNumTypesChange(boost::bind(&CellListStencil::slotTypeChange, this));
    m_box_change_conn = m_pdata->connectBoxChange(boost::bind(&CellListStencil::requestCompute, this));
    m_width_change_conn = m_cl->connectCellWidthChange(boost::bind(&CellListStencil::requestCompute, this));

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

    m_num_type_change_conn.disconnect();
    m_box_change_conn.disconnect();
    m_width_change_conn.disconnect();
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

void export_CellListStencil()
    {
    class_<CellListStencil, boost::shared_ptr<CellListStencil>, bases<Compute>, boost::noncopyable >
        ("CellListStencil", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<CellList> >());
    }
