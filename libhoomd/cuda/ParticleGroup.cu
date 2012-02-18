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

#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/scatter.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include "ParticleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ParticleGroup.cu
    \brief Contains GPU kernel code used by ParticleGroup
*/

//! Predicate to select particles that are local
struct own_particle
    {
    unsigned int N; //! Number of local particles

    //! Constructor
    /*!
     */
    own_particle(const unsigned int _N) : N(_N) { }

    //! Return true if particle is local
    /*! \param idx Index of particle to check
     */
    __host__ __device__ bool operator() (const unsigned int & idx)
        {
        return (idx < N);
        }
    };

//! Predicate to check if particle is member
struct is_member
    {
    //! Returns true if argument (is_member) is one
    __host__ __device__ bool operator() (const unsigned char & is_member)
        {
        return (is_member == 1);
        }
    };

//! GPU method to clear the membership flags
cudaError_t gpu_clear_membership_flags(unsigned int N,
                                       unsigned char *d_is_member)
    {
    thrust::device_ptr<unsigned char> is_member_ptr(d_is_member);
    thrust::fill(is_member_ptr, is_member_ptr + N, 0);
    return cudaSuccess;
    }


//! GPU method for rebuilding the index list of a ParticleGroup
/*! \param N number of local particles
    \param num_members Number of local members of the group
    \param d_member_tag Tags of local members
    \param d_is_member Array of membership flags
    \param d_member_idx Array of member indices
    \param d_rtag Array of reverse-lookup global tag -> index
    \param num_local_members Number of members on the local processor (return value)
*/
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned int num_members,
                                   unsigned int *d_member_tag,
                                   unsigned char *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int *d_rtag,
                                   unsigned int &num_local_members)
    {
    assert(d_member_tag);
    assert(d_is_member);
    assert(d_member_idx);

    thrust::device_ptr<unsigned int> member_tag_ptr(d_member_tag);
    thrust::device_ptr<unsigned char> is_member_ptr(d_is_member);
    thrust::device_ptr<unsigned int> member_idx_ptr(d_member_idx);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);



   // set membership flags
    thrust::constant_iterator<unsigned int> const_one(1);
    thrust::permutation_iterator<thrust::device_ptr<unsigned int>, thrust::device_ptr<unsigned int> > member_indices(rtag_ptr, member_tag_ptr);

    thrust::scatter_if(const_one,
                       const_one + num_members,
                       member_indices,
                       member_indices,
                       is_member_ptr,
                       own_particle(N));

    thrust::counting_iterator<unsigned int> idx(0);
    // fill member_idx array
    num_local_members = thrust::copy_if(idx, idx + N, is_member_ptr, member_idx_ptr, is_member()) - member_idx_ptr;

    return cudaSuccess;
    }

