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

// Maintainer: joaander

#ifndef __NEIGHBORLISTGPU_CUH__
#define __NEIGHBORLISTGPU_CUH__

/*! \file NeighborListGPU.cuh
    \brief Declares GPU kernel code for cell list generation on the GPU
*/

#include <cuda_runtime.h>

#include "Index1D.h"

//! Kernel driver for gpu_nlist_needs_update_check_new_kernel()
cudaError_t gpu_nlist_needs_update_check_new(unsigned int * d_result,
                                             const float4 *d_last_pos,
                                             const float4 *d_pos,
                                             const unsigned int N,
                                             const gpu_boxsize& box,
                                             const float maxshiftsq,
                                             const unsigned int checkn);

//! Kernel driver for gpu_nlist_filter_kernel()
cudaError_t gpu_nlist_filter(unsigned int *d_n_neigh,
                             unsigned int *d_nlist,
                             const Index2D& nli,
                             const unsigned int *d_n_ex,
                             const unsigned int *d_ex_list,
                             const Index2D& exli,
                             const unsigned int N,
                             const unsigned int block_size);

//! Kernel driver for gpu_compute_nlist_nsq_kernel()
cudaError_t gpu_compute_nlist_nsq(unsigned int *d_nlist,
                                  unsigned int *d_n_neigh,
                                  float4 *d_last_updated_pos,
                                  unsigned int *d_conditions,
                                  const Index2D& nli,
                                  const float4 *d_pos,
                                  const unsigned int N,
                                  const gpu_boxsize& box,
                                  const float r_maxsq);

#endif

