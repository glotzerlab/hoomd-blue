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

#include "NeighborListGPUBinned.h"
#include "NeighborListGPUBinned.cuh"

#include <boost/python.hpp>
using namespace boost::python;

NeighborListGPUBinned::NeighborListGPUBinned(boost::shared_ptr<SystemDefinition> sysdef,
                                             Scalar r_cut,
                                             Scalar r_buff,
                                             boost::shared_ptr<CellList> cl)
            : NeighborListGPU(sysdef, r_cut, r_buff), m_cl(cl)
    {
    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = boost::shared_ptr<CellList>(new CellList(sysdef));
    
    m_cl->setNominalWidth(r_cut + r_buff);
    m_cl->setRadius(1);
    m_cl->setComputeTDB(false);
    m_cl->setFlagIndex();

    gpu_setup_compute_nlist_binned();
    CHECK_CUDA_ERROR();
    
    // default to full mode
    m_storage_mode = full;
    }

void NeighborListGPUBinned::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborListGPU::setRCut(r_cut, r_buff);
    
    m_cl->setNominalWidth(r_cut + r_buff);
    }

void NeighborListGPUBinned::buildNlist(unsigned int timestep)
    {
    if (m_storage_mode != full)
        {
        cerr << endl << "***Error! Only full mode nlists can be generated on the GPU" << endl << endl;
        throw runtime_error("Error computing neighbor list");
        }

    m_cl->compute(timestep);

    if (m_prof)
        m_prof->push(exec_conf, "compute");

    // precompute scale factor
    Scalar3 width = m_cl->getWidth();
    Scalar3 scale = make_scalar3(Scalar(1.0) / width.x,
                                 Scalar(1.0) / width.y,
                                 Scalar(1.0) / width.z);
    
    // acquire the particle data
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(m_cl->getXYZFArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(m_cl->getCellAdjArray(), access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);

    gpu_compute_nlist_binned(d_nlist.data,
                             d_n_neigh.data,
                             d_last_pos.data,
                             d_conditions.data,
                             m_nlist_indexer,
                             d_pdata.pos,
                             m_pdata->getN(),
                             d_cell_size.data,
                             d_cell_xyzf.data,
                             d_cell_adj.data,
                             m_cl->getCellIndexer(),
                             m_cl->getCellListIndexer(),
                             m_cl->getCellAdjIndexer(),
                             scale,
                             m_cl->getDim(),
                             box,
                             (m_r_cut + m_r_buff)*(m_r_cut + m_r_buff),
                             96);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_pdata->release();
    
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_NeighborListGPUBinned()
    {
    class_<NeighborListGPUBinned, boost::shared_ptr<NeighborListGPUBinned>, bases<NeighborListGPU>, boost::noncopyable >
                     ("NeighborListGPUBinned", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<CellList> >())
                     ;
    }
