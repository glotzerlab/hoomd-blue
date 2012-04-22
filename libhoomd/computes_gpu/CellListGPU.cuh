/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: joaander

#ifndef __CELLLISTGPU_CUH__
#define __CELLLISTGPU_CUH__

#include <cuda_runtime.h>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "ParticleData.cuh"

/*! \file CellListGPU.cuh
    \brief Declares GPU kernel code for cell list generation on the GPU
*/

//! Kernel driver for gpu_compute_cell_list_kernel()
cudaError_t gpu_compute_cell_list(unsigned int *d_cell_size,
                                  float4 *d_xyzf,
                                  float4 *d_tdb,
                                  unsigned int *d_conditions,
                                  const float4 *d_pos,
                                  const float *d_charge,
                                  const float *d_diameter,
                                  const unsigned int *d_body,
                                  const unsigned int N,
                                  const unsigned int n_ghost,
                                  const unsigned int Nmax,
                                  const bool flag_charge,
                                  const BoxDim& box,
                                  const Index3D& ci,
                                  const Index2D& cli,
                                  const uint3 num_ghost_cells );

//! Kernel driver for gpu_compute_cell_list_1x_kernel()
cudaError_t gpu_compute_cell_list_1x(unsigned int *d_cell_size,
                                     float4 *d_xyzf,
                                     float4 *d_tdb,
                                     unsigned int *d_conditions,
                                     const float4 *d_pos,
                                     const float *d_charge,
                                     const float *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int N,
                                     const unsigned int n_ghost,
                                     const unsigned int Nmax,
                                     const bool flag_charge,
                                     const BoxDim& box,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const uint3 num_ghost_cells);

#endif

