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
// Maintainer: dnlebard

#include "AngleData.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file AngleData.cu
    \brief Implements the helper functions for updating the GPU angle table
*/


//! Kernel to find the maximum number of angles per particle
__global__ void gpu_find_max_angle_number_kernel(const uint3 *angles,
                                             const unsigned int *d_rtag,
                                             unsigned int *d_n_angles,
                                             unsigned int num_angles,
                                             unsigned int N)
    {
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (angle_idx >= num_angles)
        return;

    uint3 angle = angles[angle_idx];
    unsigned int tag1 = angle.x;
    unsigned int tag2 = angle.y;
    unsigned int tag3 = angle.z;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];
    unsigned int idx3 = d_rtag[tag3];

    if (idx1 < N)
        atomicInc(&d_n_angles[idx1], 0xffffffff);
    if (idx2 < N)
        atomicInc(&d_n_angles[idx2], 0xffffffff);
    if (idx3 < N)
        atomicInc(&d_n_angles[idx3], 0xffffffff);

    }

//! Kernel to fill the GPU angle table
__global__ void gpu_fill_gpu_angle_table(const uint3 *angles,
                                        const unsigned int *angle_type,
                                        uint4 *gpu_btable,
                                        const unsigned int pitch,
                                        const unsigned int *d_rtag,
                                        unsigned int *d_n_angles,
                                        unsigned int num_angles,
                                        unsigned int N)
    {
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (angle_idx >= num_angles)
        return;

    uint3 angle = angles[angle_idx];
    unsigned int tag1 = angle.x;
    unsigned int tag2 = angle.y;
    unsigned int tag3 = angle.z;
    unsigned int type = angle_type[angle_idx];
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];
    unsigned int idx3 = d_rtag[tag3];

    if (idx1 < N)
        {
        unsigned int num1 = atomicInc(&d_n_angles[idx1],0xffffffff);
        gpu_btable[num1*pitch+idx1] = make_uint4(idx2,idx3,type,0);
        }
    if (idx2 < N)
        {
        unsigned int num2 = atomicInc(&d_n_angles[idx2],0xffffffff);
        gpu_btable[num2*pitch+idx2] = make_uint4(idx1,idx3,type,1);
        }
    if (idx3 < N)
        {
        unsigned int num3 = atomicInc(&d_n_angles[idx3],0xffffffff);
        gpu_btable[num3*pitch+idx3] = make_uint4(idx1,idx2,type,2);
        }
    }

//! Find the maximum number of angles per particle
/*! \param max_angle_num Maximum number of angles (return value)
    \param d_n_angles Number of angles per particle (return array)
    \param d_angles Array of angles
    \param num_angles Size of angle array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag . particle index
 */
cudaError_t gpu_find_max_angle_number(unsigned int& max_angle_num,
                             unsigned int *d_n_angles,
                             const uint3 *d_angles,
                             const unsigned int num_angles,
                             const unsigned int N,
                             const unsigned int *d_rtag)
    {
    assert(d_angles);
    assert(d_rtag);
    assert(d_n_angles);

    unsigned int block_size = 512;

    // clear n_angles array
    cudaMemset(d_n_angles, 0, sizeof(unsigned int) * N);

    gpu_find_max_angle_number_kernel<<<num_angles/block_size + 1, block_size>>>(d_angles,
                                                                              d_rtag,
                                                                              d_n_angles,
                                                                              num_angles,
                                                                              N);

    thrust::device_ptr<unsigned int> n_angles_ptr(d_n_angles);
    max_angle_num = *thrust::max_element(n_angles_ptr, n_angles_ptr + N);
    return cudaSuccess;
    }

//! Construct the GPU angle table
/*! \param d_gpu_angletable Pointer to the angle table on the GPU
    \param d_n_angles Number of angles per particle (return array)
    \param d_angles Bonds array
    \param d_angle_type Array of angle types
    \param d_rtag Reverse-lookup tag->index
    \param num_angles Number of angles in angle list
    \param pitch Pitch of 2D angletable array
    \param N Number of particles
 */
cudaError_t gpu_create_angletable(uint4 *d_gpu_angletable,
                                  unsigned int *d_n_angles,
                                  const uint3 *d_angles,
                                  const unsigned int *d_angle_type,
                                  const unsigned int *d_rtag,
                                  const unsigned int num_angles,
                                  const unsigned int pitch,
                                  const unsigned int N)
    {
    unsigned int block_size = 512;

    // clear n_angles array
    cudaMemset(d_n_angles, 0, sizeof(unsigned int) * N);

    gpu_fill_gpu_angle_table<<<num_angles/block_size + 1, block_size>>>(d_angles,
                                                                      d_angle_type,
                                                                      d_gpu_angletable,
                                                                      pitch,
                                                                      d_rtag,
                                                                      d_n_angles,
                                                                      num_angles,
                                                                      N);

    return cudaSuccess;
    }

