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

#include "BondData.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file BondData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bond table
*/

//! Kernel to find the maximum number of angles per particle
__global__ void gpu_find_max_bond_number_kernel(const uint2 *bonds,
                                             const unsigned int *d_rtag,
                                             unsigned int *d_n_bonds,
                                             unsigned int num_bonds,
                                             unsigned int N,
                                             bool ghost_bonds)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    if (idx1 < N && ((idx2 < N && !ghost_bonds) || (idx2 >= N && ghost_bonds)))
        atomicInc(&d_n_bonds[idx1], 0xffffffff);
    if (idx2 < N && ((idx1 < N && !ghost_bonds) || (idx1 >= N && ghost_bonds)))
        atomicInc(&d_n_bonds[idx2], 0xffffffff);

    }

//! Kernel to fill the GPU bond table
__global__ void gpu_fill_gpu_bond_table(const uint2 *bonds,
                                        const unsigned int *bond_type,
                                        uint2 *gpu_btable,
                                        const unsigned int pitch,
                                        const unsigned int *d_rtag,
                                        unsigned int *d_n_bonds,
                                        unsigned int num_bonds,
                                        unsigned int N,
                                        bool ghost_bonds)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int type = bond_type[bond_idx];
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    if (idx1 < N && ((idx2 < N && !ghost_bonds) || (idx2 >= N && ghost_bonds)))
        {
        unsigned int num1 = atomicInc(&d_n_bonds[idx1],0xffffffff);
        gpu_btable[num1*pitch+idx1] = make_uint2(idx2,type);
        }
    if (idx2 < N && ((idx1 < N && !ghost_bonds) || (idx1 >= N && ghost_bonds)))
        {
        unsigned int num2 = atomicInc(&d_n_bonds[idx2],0xffffffff);
        gpu_btable[num2*pitch+idx2] = make_uint2(idx1,type);
        }
    }


//! Find the maximum number of bonds per particle
/*! \param max_bond_num Maximum number of bonds (return value)
    \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Array of bonds
    \param num_bonds Size of bond array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag . particle index
    \param use_ghost_bonds True if we are only considering bonds with ghost particles
 */
cudaError_t gpu_find_max_bond_number(unsigned int& max_bond_num,
                                     unsigned int *d_n_bonds,
                                     const uint2 *d_bonds,
                                     const unsigned int num_bonds,
                                     const unsigned int N,
                                     const unsigned int *d_rtag,
                                     bool use_ghost_bonds)
    {
    assert(d_bonds);
    assert(d_rtag);
    assert(d_n_bonds);

    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_find_max_bond_number_kernel<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                              d_rtag,
                                                                              d_n_bonds,
                                                                              num_bonds,
                                                                              N,
                                                                              use_ghost_bonds);

    thrust::device_ptr<unsigned int> n_bonds_ptr(d_n_bonds);
    max_bond_num = *thrust::max_element(n_bonds_ptr, n_bonds_ptr + N);
    return cudaSuccess;
    }

//! Construct the GPU bond table
/*! \param d_gpu_bondtable Pointer to the bond table on the GPU
    \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Bonds array
    \param d_bond_type Array of bond types
    \param d_rtag Reverse-lookup tag->index
    \param num_bonds Number of bonds in bond list
    \param pitch Pitch of 2D bondtable array
    \param N Number of particles
    \param use_ghost_bonds True if we are only considering bonds with ghost particles
 */
cudaError_t gpu_create_bondtable(uint2 *d_gpu_bondtable,
                                 unsigned int *d_n_bonds,
                                 const uint2 *d_bonds,
                                 const unsigned int *d_bond_type,
                                 const unsigned int *d_rtag,
                                 const unsigned int num_bonds,
                                 unsigned int pitch,
                                 unsigned int N,
                                 bool use_ghost_bonds)
    {
    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_fill_gpu_bond_table<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                      d_bond_type,
                                                                      d_gpu_bondtable,
                                                                      pitch,
                                                                      d_rtag,
                                                                      d_n_bonds,
                                                                      num_bonds,
                                                                      N,
                                                                      use_ghost_bonds);
    return cudaSuccess;
    }

