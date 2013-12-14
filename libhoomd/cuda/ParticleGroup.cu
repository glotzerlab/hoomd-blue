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

// Maintainer: jglaser

#include "ParticleData.cuh"

#include "moderngpu/kernels/scan.cuh"

/*! \file ParticleGroup.cu
    \brief Contains GPU kernel code used by ParticleGroup
*/

//! GPU kernel to translate between global and local membership lookup table
__global__ void gpu_rebuild_index_list_kernel(unsigned int N,
                                              unsigned int *d_tag,
                                              unsigned char *d_is_member_tag,
                                              unsigned char *d_is_member)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    unsigned int tag = d_tag[idx];

    d_is_member[idx] = d_is_member_tag[tag];
    }

__global__ void gpu_scatter_member_indices(unsigned int N,
    const unsigned int *d_scan,
    const unsigned char *d_is_member,
    unsigned *d_member_idx)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx >= N) return;

    if (d_is_member[idx])
        d_member_idx[d_scan[idx]] = idx;
    }

//! GPU method for rebuilding the index list of a ParticleGroup
/*! \param N number of local particles
    \param d_is_member_tag Global lookup table for tag -> group membership
    \param d_is_member Array of membership flags
    \param d_member_idx Array of member indices
    \param d_tag Array of tags
    \param num_local_members Number of members on the local processor (return value)
*/
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned char *d_is_member_tag,
                                   unsigned char *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int *d_tag,
                                   unsigned int &num_local_members,
                                   cached_allocator& alloc,
                                   mgpu::ContextPtr mgpu_context)
    {
    assert(d_is_member);
    assert(d_is_member_tag);
    assert(d_member_idx);
    assert(d_tag);

    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;
    gpu_rebuild_index_list_kernel<<<n_blocks,block_size>>>(N,
                                                         d_tag,
                                                         d_is_member_tag,
                                                         d_is_member);

    // allocate temporary array
    unsigned int *d_tmp = (unsigned int *)alloc.allocate(N*sizeof(unsigned int));

    // compute member_idx offsets
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_is_member, N, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *) NULL, &num_local_members, d_tmp, *mgpu_context);


    // fill member_idx array
    gpu_scatter_member_indices<<<n_blocks, block_size>>>(N, d_tmp, d_is_member, d_member_idx);

    // release temporary array
    alloc.deallocate((char *) d_tmp,0);

    return cudaSuccess;
    }
