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
                                             const BoxDim& box,
                                             const float maxshiftsq,
                                             const float3 lambda,
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
                                  const unsigned int n_ghost,
                                  const BoxDim& box,
                                  const float r_maxsq);

//! GPU function to update the exclusion list on the device
cudaError_t gpu_update_exclusion_list(const unsigned int *d_tag,
                                const unsigned int *d_rtag,
                                const unsigned int *d_n_ex_tag,
                                const unsigned int *d_ex_list_tag,
                                const Index2D& ex_list_tag_indexer,
                                unsigned int *d_n_ex_idx,
                                unsigned int *d_ex_list_idx,
                                const Index2D& ex_list_indexer,
                                const unsigned int N);
 
#endif

