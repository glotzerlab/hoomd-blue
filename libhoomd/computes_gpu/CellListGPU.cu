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

#include "CellListGPU.cuh"

/*! \file CellListGPU.cu
    \brief Defines GPU kernel code for cell list generation on the GPU
*/

//! Kernel that computes the cell list on the GPU
/*! 
    \note Optimized for Fermi
*/
__global__ void gpu_compute_cell_list_kernel(unsigned int *d_cell_size,
                                             float4 *d_xyzf,
                                             float4 *d_tdb,
                                             unsigned int *d_conditions,
                                             const float4 *d_pos,
                                             const float *d_charge,
                                             const float *d_diameter,
                                             const unsigned int *d_body,
                                             const unsigned int N,
                                             const unsigned int Nmax,
                                             const bool flag_charge,
                                             const Scalar3 scale,
                                             const gpu_boxsize box,
                                             const Index3D ci,
                                             const Index2D cli)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
        
    float4 pos = d_pos[idx];
    float flag = 0.0f;
    float diameter = 0.0f;
    float body = 0;
    float type = 0;
    if (d_tdb != NULL)
        {
        diameter = d_diameter[idx];
        body = __int_as_float(d_body[idx]);
        type = pos.w;
        }
        
    if (flag_charge)
        flag = d_charge[idx];
    else
        flag = __int_as_float(idx);

    // check for nan pos
    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z))
        {
        d_conditions[1] = idx+1;
        return;
        }
    
    // determine which bin it belongs in
    unsigned int ib = (unsigned int)((pos.x+box.Lx/2.0f)*scale.x);
    unsigned int jb = (unsigned int)((pos.y+box.Ly/2.0f)*scale.y);
    unsigned int kb = (unsigned int)((pos.z+box.Lz/2.0f)*scale.z);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW())
        ib = 0;
    if (jb == ci.getH())
        jb = 0;
    if (kb == ci.getD())
        kb = 0;
        
    unsigned int bin = ci(ib, jb, kb);

    // check if the particle is inside the dimensions
    if (bin >= ci.getNumElements())
        {
        d_conditions[2] = idx+1;
        return;
        }
    
    unsigned int size = atomicInc(&d_cell_size[bin], 0xffffffff);
    if (size < Nmax)
        {
        unsigned int write_pos = cli(size, bin);
        d_xyzf[write_pos] = make_float4(pos.x, pos.y, pos.z, flag);
        if (d_tdb != NULL)
            d_tdb[write_pos] = make_float4(type, diameter, body, 0.0f);
        }
    else
        {
        // handle overflow
        atomicMax(&d_conditions[0], size+1);
        }
    }

cudaError_t gpu_compute_cell_list(unsigned int *d_cell_size,
                                  float4 *d_xyzf,
                                  float4 *d_tdb,
                                  unsigned int *d_conditions,
                                  const float4 *d_pos,
                                  const float *d_charge,
                                  const float *d_diameter,
                                  const unsigned int *d_body,
                                  const unsigned int N,
                                  const unsigned int Nmax,
                                  const bool flag_charge,
                                  const Scalar3& scale,
                                  const gpu_boxsize& box,
                                  const Index3D& ci,
                                  const Index2D& cli)
    {
    unsigned int block_size = 256;
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    
    cudaError_t err;
    err = cudaMemset(d_cell_size, 0, sizeof(unsigned int)*ci.getNumElements());
    
    if (err != cudaSuccess)
        return err;
    
    gpu_compute_cell_list_kernel<<<n_blocks, block_size>>>(d_cell_size,
                                                           d_xyzf,
                                                           d_tdb,
                                                           d_conditions,
                                                           d_pos,
                                                           d_charge,
                                                           d_diameter,
                                                           d_body,
                                                           N,
                                                           Nmax,
                                                           flag_charge,
                                                           scale,
                                                           box,
                                                           ci,
                                                           cli);
    
    return cudaSuccess;
    }
