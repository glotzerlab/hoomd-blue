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

#include <limits.h>
#include <boost/bind.hpp>

#include "CellList.h"

using namespace boost;

/*! \param sysdef system to compute the cell list of
*/
CellList::CellList(boost::shared_ptr<SystemDefinition> sysdef)
    : Compute(sysdef),  m_nominal_width(Scalar(1.0f)), m_radius(1), m_max_cells(UINT_MAX), m_compute_tdb(false),
      m_flag_charge(false)
    {
    // allocation is deferred until the first compute() call - initialize values to dummy variables
    m_width = make_scalar3(0.0, 0.0, 0.0);
    m_dim = make_uint3(0,0,0);
    m_Nmax = 32;
    m_params_changed = true;
    m_particles_sorted = false;
    m_box_changed = false;
    
    m_sort_connection = m_pdata->connectParticleSort(bind(&CellList::slotParticlesSorted, this));
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&CellList::slotBoxChanged, this));
    }

CellList::~CellList()
    {
    m_sort_connection.disconnect();
    m_boxchange_connection.disconnect();
    }

/*! \returns Cell dimensions that match with the current width, box dimension, and max_cells setting
*/
uint3 CellList::computeDimensions()
    {
    uint3 dim;
    
    // calculate the bin dimensions
    const BoxDim& box = m_pdata->getBox();
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    dim.x = (unsigned int)((box.xhi - box.xlo) / (m_nominal_width));
    dim.y = (unsigned int)((box.yhi - box.ylo) / (m_nominal_width));
    
    if (m_sysdef->getNDimensions() == 2)
        {
        dim.z = 3;
    
        // decrease the number of bins if it exceeds the max
        if (dim.x * dim.y * dim.z > m_max_cells)
            {
            float scale_factor = powf(float(m_max_cells) / float(dim.x*dim.y*dim.z), 1.0f/2.0f);
            dim.x = int(float(dim.x)*scale_factor);
            dim.y = int(float(dim.y)*scale_factor);
            }
        }
    else
        {
        dim.z = (unsigned int)((box.zhi - box.zlo) / (m_nominal_width));

        // decrease the number of bins if it exceeds the max
        if (dim.x * dim.y * dim.z > m_max_cells)
            {
            float scale_factor = powf(float(m_max_cells) / float(dim.x*dim.y*dim.z), 1.0f/3.0f);
            dim.x = int(float(dim.x)*scale_factor);
            dim.y = int(float(dim.y)*scale_factor);
            dim.z = int(float(dim.z)*scale_factor);
            }
        }
    
    return dim;
    }

void CellList::compute(unsigned int timestep)
    {
    bool force = false;
    
    if (m_prof)
        m_prof->push("Cell");
    
    if (m_params_changed)
        {
        // need to fully reinitialize on any parameter change
        initializeAll();
        m_params_changed = false;
        force = true;
        }
    
    if (m_box_changed)
        {
        uint3 new_dim = computeDimensions();
        if (new_dim.x == m_dim.x && new_dim.y == m_dim.y && new_dim.z == m_dim.z)
            {
            // number of bins has not changed, only need to update width
            initializeWidth();
            }
        else
            {
            // number of bins has changed, need to fully reinitialize memory
            initializeAll();
            }
        
        m_box_changed = false;
        force = true;
        }
    
    if (m_particles_sorted)
        {
        // sorted particles simply need a forced update to get the proper indices in the data structure
        m_particles_sorted = false;
        force = true;
        }
    
    // only update if we need to
    if (shouldCompute(timestep) || force)
        {
        computeCellList();
        }
    
    if (m_prof)
        m_prof->pop();
    }

void CellList::initializeAll()
    {
    initializeWidth();
    initializeMemory();
    }

void CellList::initializeWidth()
    {
    if (m_prof)
        m_prof->push("init");
    
    // initialize dimensions and width
    m_dim = computeDimensions();

    const BoxDim& box = m_pdata->getBox();
    m_width.x = (box.xhi - box.xlo) / Scalar(m_dim.x);
    m_width.y = (box.yhi - box.ylo) / Scalar(m_dim.y);
    m_width.z = (box.zhi - box.zlo) / Scalar(m_dim.z);

    if (m_prof)
        m_prof->pop();

    }

void CellList::initializeMemory()
    {
    if (m_prof)
        m_prof->push("init");
    
    // estimate Nmax
    // TODO
    
    // initialize indexers
    m_cell_indexer = Index3D(m_dim.x, m_dim.y, m_dim.z);
    m_cell_list_indexer = Index2D(m_Nmax, m_cell_indexer.getNumElements());
    m_cell_adj_indexer = Index2D((m_radius*2+1)*(m_radius*2+1)*(m_radius*2+1), m_cell_indexer.getNumElements());
    
    // allocate memory
    // TODO

    if (m_prof)
        m_prof->pop();
    }

void CellList::computeCellList()
    {
    
    }