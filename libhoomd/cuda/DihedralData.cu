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

#include "DihedralData.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file DihedralData.cu
    \brief Implements the helper functions for updating the GPU dihedral table
*/


//! Kernel to find the maximum number of dihedrals per particle
__global__ void gpu_find_max_dihedral_number_kernel(const uint4 *dihedrals,
                                             const unsigned int *d_rtag,
                                             unsigned int *d_n_dihedrals,
                                             unsigned int num_dihedrals,
                                             unsigned int N)
    {
    int dihedral_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dihedral_idx >= num_dihedrals)
        return;

    uint4 dihedral = dihedrals[dihedral_idx];
    unsigned int tag1 = dihedral.x;
    unsigned int tag2 = dihedral.y;
    unsigned int tag3 = dihedral.z;
    unsigned int tag4 = dihedral.w;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];
    unsigned int idx3 = d_rtag[tag3];
    unsigned int idx4 = d_rtag[tag4];

    if (idx1 < N)
        atomicInc(&d_n_dihedrals[idx1], 0xffffffff);
    if (idx2 < N)
        atomicInc(&d_n_dihedrals[idx2], 0xffffffff);
    if (idx3 < N)
        atomicInc(&d_n_dihedrals[idx3], 0xffffffff);
    if (idx4 < N)
        atomicInc(&d_n_dihedrals[idx4], 0xffffffff);
    }

//! Kernel to fill the GPU dihedral table
__global__ void gpu_fill_gpu_dihedral_table(const uint4 *dihedrals,
                                        const unsigned int *dihedral_type,
                                        uint4 *gpu_btable,
                                        uint1 *dihedrals_ABCD,
                                        const unsigned int pitch,
                                        const unsigned int *d_rtag,
                                        unsigned int *d_n_dihedrals,
                                        unsigned int num_dihedrals,
                                        unsigned int N)
    {
    int dihedral_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dihedral_idx >= num_dihedrals)
        return;

    uint4 dihedral = dihedrals[dihedral_idx];
    unsigned int tag1 = dihedral.x;
    unsigned int tag2 = dihedral.y;
    unsigned int tag3 = dihedral.z;
    unsigned int tag4 = dihedral.w;
    unsigned int type = dihedral_type[dihedral_idx];
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];
    unsigned int idx3 = d_rtag[tag3];
    unsigned int idx4 = d_rtag[tag4];

    if (idx1 < N)
        {
        unsigned int num1 = atomicInc(&d_n_dihedrals[idx1],0xffffffff);
        gpu_btable[num1*pitch+idx1] = make_uint4(idx2,idx3,idx4,type);
        dihedrals_ABCD[num1*pitch+idx1] = make_uint1(0);
        }
    if (idx2 < N)
        {
        unsigned int num2 = atomicInc(&d_n_dihedrals[idx2],0xffffffff);
        gpu_btable[num2*pitch+idx2] = make_uint4(idx1,idx3,idx4,type);
        dihedrals_ABCD[num2*pitch+idx2] = make_uint1(1);
        }
    if (idx3 < N)
        {
        unsigned int num3 = atomicInc(&d_n_dihedrals[idx3],0xffffffff);
        gpu_btable[num3*pitch+idx3] = make_uint4(idx1,idx2,idx4,type);
        dihedrals_ABCD[num3*pitch+idx3] = make_uint1(2);
        }
    if (idx4 < N)
        {
        unsigned int num4 = atomicInc(&d_n_dihedrals[idx4],0xffffffff);
        gpu_btable[num4*pitch+idx4] = make_uint4(idx1,idx2,idx3,type);
        dihedrals_ABCD[num4*pitch+idx4] = make_uint1(3);
        }
    }

//! Find the maximum number of dihedrals per particle
/*! \param max_dihedral_num Maximum number of dihedrals (return value)
    \param d_n_dihedrals Number of dihedrals per particle (return array)
    \param d_dihedrals Array of dihedrals
    \param num_dihedrals Size of dihedral array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag . particle index
 */
cudaError_t gpu_find_max_dihedral_number(unsigned int& max_dihedral_num,
                             unsigned int *d_n_dihedrals,
                             const uint4 *d_dihedrals,
                             const unsigned int num_dihedrals,
                             const unsigned int N,
                             const unsigned int *d_rtag)
    {
    assert(d_dihedrals);
    assert(d_rtag);
    assert(d_n_dihedrals);

    unsigned int block_size = 512;

    // clear n_dihedrals array
    cudaMemset(d_n_dihedrals, 0, sizeof(unsigned int) * N);

    gpu_find_max_dihedral_number_kernel<<<num_dihedrals/block_size + 1, block_size>>>(d_dihedrals,
                                                                              d_rtag,
                                                                              d_n_dihedrals,
                                                                              num_dihedrals,
                                                                              N);

    thrust::device_ptr<unsigned int> n_dihedrals_ptr(d_n_dihedrals);
    max_dihedral_num = *thrust::max_element(n_dihedrals_ptr, n_dihedrals_ptr + N);
    return cudaSuccess;
    }

//! Construct the GPU dihedral table
/*! \param d_gpu_dihedraltable Pointer to the dihedral table on the GPU
    \param d_dihedrals_ABCD Table of atom positions in the dihedrals
    \param d_n_dihedrals Number of dihedrals per particle (return array)
    \param d_dihedrals Bonds array
    \param d_dihedral_type Array of dihedral types
    \param d_rtag Reverse-lookup tag->index
    \param num_dihedrals Number of dihedrals in dihedral list
    \param pitch Pitch of 2D dihedraltable array
    \param N Number of particles
 */
cudaError_t gpu_create_dihedraltable(uint4 *d_gpu_dihedraltable,
                                  uint1 *d_dihedrals_ABCD,
                                  unsigned int *d_n_dihedrals,
                                  const uint4 *d_dihedrals,
                                  const unsigned int *d_dihedral_type,
                                  const unsigned int *d_rtag,
                                  const unsigned int num_dihedrals,
                                  const unsigned int pitch,
                                  const unsigned int N)
    {
    unsigned int block_size = 512;

    // clear n_dihedrals array
    cudaMemset(d_n_dihedrals, 0, sizeof(unsigned int) * N);

    gpu_fill_gpu_dihedral_table<<<num_dihedrals/block_size + 1, block_size>>>(d_dihedrals,
                                                                      d_dihedral_type,
                                                                      d_gpu_dihedraltable,
                                                                      d_dihedrals_ABCD,
                                                                      pitch,
                                                                      d_rtag,
                                                                      d_n_dihedrals,
                                                                      num_dihedrals,
                                                                      N);

    return cudaSuccess;
    }
