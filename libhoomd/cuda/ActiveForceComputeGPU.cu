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

// Maintainer: joaander

#include "ActiveForceComputeGPU.cuh"

#include <assert.h>

/*! \file ActiveForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveForceComputeGPU.
*/

//! Kernel for calculating blah
/*! 
*/
extern "C" __global__
void gpu_compute_active_force_set_constraints_kernel(const unsigned int *d_group_members,
                                                   unsigned int group_size,
                                                   const unsigned int N,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_actVec,
                                                   const Scalar4 *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
}

void gpu_compute_active_force_rotational_diffusion_kernel(const unsigned int *d_group_members,
                                                   unsigned int group_size,
                                                   const unsigned int N,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_actVec,
                                                   const Scalar4 *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
}

void gpu_compute_active_force_set_forces_kernel(const unsigned int *d_group_members,
                                                   unsigned int group_size,
                                                   const unsigned int N,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_actVec,
                                                   const Scalar4 *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
}










cudaError_t gpu_compute_active_force_set_constraints(const unsigned int *d_group_members,
                                                   unsigned int group_size,
                                                   const unsigned int N,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_actVec,
                                                   const Scalar4 *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
{
    assert(d_group_members);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_set_constraints_kernel<<< grid, threads>>>(d_group_members,
                                                                    group_size,
                                                                    N,
                                                                    d_pos,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);

    return cudaSuccess;
}

cudaError_t gpu_compute_active_force_rotational_diffusion(const unsigned int *d_group_members,
                                                       unsigned int group_size
                                                       const unsigned int N,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_actVec,
                                                       const Scalar4 *d_actMag,
                                                       const Scalar3& P,
                                                       Scalar rx,
                                                       Scalar ry,
                                                       Scalar rz,
                                                       unsigned int block_size)
{
    assert(d_group_members);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_rotational_diffusion_kernel<<< grid, threads>>>(d_group_members,
                                                                    group_size,
                                                                    N,
                                                                    d_pos,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);

    return cudaSuccess;
}

cudaError_t cudaError_t gpu_compute_active_force_set_forces(const unsigned int *d_group_members,
                                           unsigned int group_size,
                                           const unsigned int N,
                                           Scalar4* d_force,
                                           const Scalar4 *d_orientation,
                                           const Scalar4 *d_actVec,
                                           const Scalar4 *d_actMag,
                                           const Scalar3& P,
                                           Scalar rx,
                                           Scalar ry,
                                           Scalar rz,
                                           unsigned int block_size)
{
    assert(d_group_members);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_set_forces_kernel<<< grid, threads>>>(d_group_members,
                                                                    group_size,
                                                                    N,
                                                                    d_force,
                                                                    d_orientation,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);

    return cudaSuccess;
}










