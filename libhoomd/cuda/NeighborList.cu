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

#include "NeighborList.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file NeighborList.cu
    \brief Defines data structures and methods used by NeighborList and descendants
*/

//! Checks if the neighbor list needs updating
/*! \param pdata Particle data to check
    \param nlist Current neighbor list build from that particle data
    \param r_buffsq A precalculated copy of r_buff*r_buff
    \param box Box dimensions for periodic boundary handling

    If any particle has moved a distance larger than r_buffsq since the last neighbor list update,
    nlist.needs_update is set to 1.
*/
__global__ void gpu_nlist_needs_update_check_kernel(gpu_pdata_arrays pdata,
                                                    gpu_nlist_array nlist,
                                                    float r_buffsq,
                                                    gpu_boxsize box)
    {
    // each thread will compare vs it's old position to see if the list needs updating
    // if that is true, write a 1 to nlist_needs_updating
    // it is possible that writes will collide, but at least one will succeed and that is all that matters
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pidx = idx + pdata.local_beg;
    
    if (idx < pdata.local_num)
        {
        float4 cur_pos = pdata.pos[pidx];
        float4 last_pos = nlist.last_updated_pos[pidx];
        float dx = cur_pos.x - last_pos.x;
        float dy = cur_pos.y - last_pos.y;
        float dz = cur_pos.z - last_pos.z;
        
        dx = dx - box.Lx * rintf(dx * box.Lxinv);
        dy = dy - box.Ly * rintf(dy * box.Lyinv);
        dz = dz - box.Lz * rintf(dz * box.Lzinv);
        
        float drsq = dx*dx + dy*dy + dz*dz;
        
        if (drsq >= r_buffsq && pidx < pdata.N)
            {
            *nlist.needs_update = 1;
            }
        }
    }

//! Check if the neighborlist needs updating
/*! \param pdata Particle data to check
    \param box Box dimensions for periodic boundary handling
    \param nlist Current neighbor list build from that particle data
    \param r_buffsq A precalculated copy of r_buff*r_buff
    \param result Pointer to write the result to

    If any particle has moved a distance larger than r_buffsq since the last neighbor list update,
    *result is set to 1. Otherwide *result is set to 0.
*/
cudaError_t gpu_nlist_needs_update_check(gpu_pdata_arrays *pdata,
                                         gpu_boxsize *box,
                                         gpu_nlist_array *nlist,
                                         float r_buffsq,
                                         int *result)
    {
    assert(pdata);
    assert(nlist);
    assert(result);
    
    // start by zeroing the value on the device
    *result = 0;
    cudaError_t error = cudaMemcpy(nlist->needs_update, result,
                                   sizeof(int), cudaMemcpyHostToDevice);
                                   
    // run the kernel
    int M = 256;
    dim3 grid( (pdata->local_num/M) + 1, 1, 1);
    dim3 threads(M, 1, 1);
    
    // run the kernel
    if (error == cudaSuccess)
        {
        gpu_nlist_needs_update_check_kernel<<< grid, threads >>>(*pdata, *nlist, r_buffsq, *box);
        error = cudaSuccess;
        }
        
    if (error == cudaSuccess)
        {
        error = cudaMemcpy(result, nlist->needs_update,
                           sizeof(int), cudaMemcpyDeviceToHost);
        }
    return error;
    }

// vim:syntax=cpp

