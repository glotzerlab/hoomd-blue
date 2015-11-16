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

// Maintainer: jglaser

#include "HOOMDMath.h"
#include "BondedGroupData.cuh"
#include "Index1D.h"
#include "BoxDim.h"

#include <cusolverDn.h>
#include <cublas_v2.h>

#ifndef __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__
#define __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__

cudaError_t gpu_fill_matrix_vector(unsigned int n_constraint,
                          unsigned int nptl_local,
                          Scalar *d_matrix,
                          Scalar *d_C,
                          const Scalar4 *d_pos,
                          const Scalar4 *d_vel,
                          const Scalar4 *d_netforce,
                          const group_storage<2> *d_gpu_clist,
                          const Index2D & gpu_clist_indexer,
                          const unsigned int *d_gpu_n_constraints,
                          const unsigned int *d_gpu_cpos,
                          const unsigned int *d_gpu_cidx,
                          Scalar deltaT,
                          const BoxDim box,
                          unsigned int block_size);

cudaError_t gpu_compute_constraint_forces_buffer_size(Scalar *d_matrix,
    unsigned int n_constraint,
    int &work_size,
    cusolverDnHandle_t solver_handle);

cudaError_t gpu_compute_constraint_forces(unsigned int n_constraint,
                                   Scalar *d_matrix,
                                   Scalar *d_C,
                                   const Scalar4 *d_pos,
                                   const group_storage<2> *d_gpu_clist,
                                   const Index2D & gpu_clist_indexer,
                                   const unsigned int *d_gpu_n_constraints,
                                   const unsigned int *d_gpu_cpos,
                                   const unsigned int *d_gpu_cidx,
                                   Scalar4 *d_force,
                                   const BoxDim box,
                                   unsigned int nptl_local,
                                   unsigned int block_size,
                                   cublasHandle_t cublas_handle,
                                   cusolverDnHandle_t solver_handle,
                                   Scalar *d_work,
                                   Scalar *d_tau,
                                   Scalar *d_Q,
                                   Scalar *d_R,
                                   Scalar *d_B,
                                   int *d_devinfo,
                                   unsigned int work_size);
#endif
