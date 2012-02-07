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

#include<thrust/iterator/permutation_iterator.h>
#include<thrust/iterator/constant_iterator.h>
#include<thrust/fill.h>
#include<thrust/set_operations.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ParticleGroup.cu
    \brief Contains GPU kernel code used by ParticleGroup
*/

//! GPU method for rebuilding the index list of a ParticleGroup
/*! \param N number of local particles
    \param num_members number of local members of the group
    \param d_member_tag tags of local members
    \param d_is_member array of membership flags
    \param d_member_idx array of member indices
    \param d_rtag array of reverse-lookup global tag -> index
*/
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned int num_members,
                                   unsigned int *d_member_tag,
                                   unsigned char *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int *d_rtag)
    {
    assert(d_member_tag);
    assert(d_is_member);
    assert(d_member_idx);

    thrust::device_ptr<unsigned int> member_tag_ptr(d_member_tag);
    thrust::device_ptr<unsigned char> is_member_ptr(d_is_member);
    thrust::device_ptr<unsigned int> member_idx_ptr(d_member_idx);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // clear membership flags
    thrust::fill(is_member_ptr, is_member_ptr + N, 0);

    thrust::permutation_iterator<thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int> > member_indices(rtag_ptr, member_tag_ptr);

    // fill member_idx array
    thrust::copy(member_indices, member_indices + num_members, member_idx_ptr);

    // set membership flags
    thrust::constant_iterator<int> const_one(1);
    thrust::scatter(const_one, const_one + num_members, member_indices, is_member_ptr);

    return cudaSuccess;
    }

//! Implementation of GPU code for ParticleSelectorGlobalTagListGPU
/*! \param num_global_members Global number of members
    \param N number of local particles
    \param d_tag Array of global particle tags of local particles
    \param d_global_member_tags Array of global particle tags that are members of the group
    \param d_member_tags Array to fill with local member tags that are also group members
    \param num_local_members Number of local members (return value)

    \pre d_member_tags has to be allocated with sufficient capacity (i.e. for a maximum of N
         members)

    \post d_member_tags will contain the intersection of d_global_member_tags and d_tag,
          and num_local_members will contain the size of the intersection
 */
cudaError_t gpu_particle_selector_tag_list(const unsigned int num_global_members,
                                           const unsigned int N,
                                           unsigned int *d_tag,
                                           unsigned int *d_global_member_tags,
                                           unsigned int & num_local_members,
                                           unsigned int *d_member_tags)
    {
    assert(N>0);
    assert(d_tag);
    assert(d_global_member_tags);
    assert(d_member_tags);

    thrust::device_ptr<unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> global_member_tags_ptr(d_global_member_tags);
    thrust::device_ptr<unsigned int> member_tags_ptr(d_member_tags);
    num_local_members = thrust::set_intersection(tag_ptr,
                             tag_ptr+N,
                             global_member_tags_ptr,
                             global_member_tags_ptr+num_global_members,
                             member_tags_ptr) - member_tags_ptr;
    return cudaSuccess;
    }
