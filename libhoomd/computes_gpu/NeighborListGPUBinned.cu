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

#include "NeighborListGPUBinned.cuh"

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for O(N) neighbor list generation on the GPU
*/

//! Kernel call for generating neighbor list on the GPU
/*! \note optimized for Fermi
*/
__global__ void gpu_compute_nlist_binned_new_kernel(unsigned int *d_nlist,
                                                    unsigned int *d_n_neigh,
                                                    const Index2D nli,
                                                    const float4 *d_pos,
                                                    const unsigned int N,
                                                    const unsigned int *d_cell_size,
                                                    const float4 *d_cell_xyzf,
                                                    const unsigned int *d_cell_adj,
                                                    const Index3D ci,
                                                    const Index2D cli,
                                                    const Index2D cadji,
                                                    const float3 cell_scale,
                                                    const uint3 cell_dim,
                                                    const gpu_boxsize box,
                                                    const float r_maxsq)
    {
    // each thread is going to compute the neighbor list for a single particle
    int my_pidx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // quit early if we are past the end of the array
    if (my_pidx >= N)
        return;
    
    // first, determine which bin this particle belongs to
    float4 my_pos = d_pos[my_pidx];
    
    // FLOPS: 9
    unsigned int ib = (unsigned int)((my_pos.x+box.Lx/2.0f)*cell_scale.x);
    unsigned int jb = (unsigned int)((my_pos.y+box.Ly/2.0f)*cell_scale.y);
    unsigned int kb = (unsigned int)((my_pos.z+box.Lz/2.0f)*cell_scale.z);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == cell_dim.x)
        ib = 0;
    if (jb == cell_dim.y)
        jb = 0;
    if (kb == cell_dim.z)
        kb = 0;
        
    int my_cell = ci(ib,jb,kb);
    
    // each thread will determine the neighborlist of a single particle
    // count number of neighbors found so far in n_neigh
    int n_neigh = 0;

    // loop over all adjacent bins
    for (unsigned int cur_adj = 0; cur_adj < cadji.getW(); cur_adj++)
        {
        int neigh_cell = d_cell_adj[cadji(cur_adj, my_cell)];
        unsigned int size = d_cell_size[neigh_cell];
        
        // now, we are set to loop through the array
        for (int cur_offset = 0; cur_offset < size; cur_offset++)
            {
            float4 cur_xyzf = d_cell_xyzf[cli(cur_offset, neigh_cell)];
            
            float3 neigh_pos;
            neigh_pos.x = cur_xyzf.x;
            neigh_pos.y = cur_xyzf.y;
            neigh_pos.z = cur_xyzf.z;
            int cur_neigh = __float_as_int(cur_xyzf.w);
            
            // compute the distance between the two particles
            float dx = my_pos.x - neigh_pos.x;
            float dy = my_pos.y - neigh_pos.y;
            float dz = my_pos.z - neigh_pos.z;
            
            // wrap the periodic boundary conditions
            dx = dx - box.Lx * rintf(dx * box.Lxinv);
            dy = dy - box.Ly * rintf(dy * box.Lyinv);
            dz = dz - box.Lz * rintf(dz * box.Lzinv);
            
            // compute dr squared
            float drsq = dx*dx + dy*dy + dz*dz;
            
            if (drsq <= r_maxsq && my_pidx != cur_neigh)
                {
                d_nlist[nli(my_pidx, min(n_neigh,nli.getH()))] = cur_neigh;
                n_neigh++;
                }
            }
        }
    
    d_n_neigh[my_pidx] = n_neigh;
    }

cudaError_t gpu_compute_nlist_binned(unsigned int *d_nlist,
                                     unsigned int *d_n_neigh,
                                     const Index2D& nli,
                                     const float4 *d_pos,
                                     const unsigned int N,
                                     const unsigned int *d_cell_size,
                                     const float4 *d_cell_xyzf,
                                     const unsigned int *d_cell_adj,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Index2D& cadji,
                                     const float3& cell_scale,
                                     const uint3& cell_dim,
                                     const gpu_boxsize& box,
                                     const float r_maxsq,
                                     const unsigned int block_size)
    {
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    
    gpu_compute_nlist_binned_new_kernel<<<n_blocks, block_size>>>(d_nlist,
                                                                  d_n_neigh,
                                                                  nli,
                                                                  d_pos,
                                                                  N,
                                                                  d_cell_size,
                                                                  d_cell_xyzf,
                                                                  d_cell_adj,
                                                                  ci,
                                                                  cli,
                                                                  cadji,
                                                                  cell_scale,
                                                                  cell_dim,
                                                                  box,
                                                                  r_maxsq);
    
    return cudaSuccess;
    }
