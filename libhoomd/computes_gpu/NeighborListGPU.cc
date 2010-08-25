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

/*! \file NeighborListGPU.cc
    \brief Implementation of the NeighborListGPU class
*/

#include "NeighborListGPU.h"
#include "NeighborListGPU.cuh"

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
using namespace std;

bool NeighborListGPU::distanceCheck()
    {
    // scan through the particle data arrays and calculate distances
    if (m_prof) m_prof->push(exec_conf, "dist-check");
    
        {
        // access data
        gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
        gpu_boxsize box = m_pdata->getBoxGPU();
        ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);
    
        // create a temporary copy of r_buff/2 sqaured
        Scalar maxshiftsq = (m_r_buff/Scalar(2.0)) * (m_r_buff/Scalar(2.0));
    
        gpu_nlist_needs_update_check_new(d_flags.data, d_last_pos.data, pdata.pos, m_pdata->getN(), box, maxshiftsq);
    
        if (exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_pdata->release();
        }

    if (m_prof) m_prof->pop(exec_conf);

        {
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);
        return h_flags.data[0];
        }
    }

/*! Calls gpu_nlsit_filter() to filter the neighbor list on the GPU
*/
void NeighborListGPU::filterNlist()
    {
    if (m_prof)
        m_prof->push(exec_conf, "filter");
    
    // access data
    
    ArrayHandle<unsigned int> d_n_ex_idx(m_n_ex_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_ex_list_idx(m_ex_list_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::readwrite);
    
    gpu_nlist_filter(d_n_neigh.data,
                     d_nlist.data,
                     m_nlist_indexer,
                     d_n_ex_idx.data,
                     d_ex_list_idx.data,
                     m_ex_list_indexer,
                     m_pdata->getN());
    
    if (m_prof)
        m_prof->pop(exec_conf);
    }


void export_NeighborListGPU()
    {
    class_<NeighborListGPU, boost::shared_ptr<NeighborListGPU>, bases<NeighborList>, boost::noncopyable >
                     ("NeighborListGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     ;
    }
