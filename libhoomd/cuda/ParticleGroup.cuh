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

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ParticleGroup.cuh
    \brief Contains GPU kernel code used by ParticleGroup
*/

#ifndef __PARTICLE_GROUP_CUH__
#define __PARTICLE_GROUP_CUH__

//! Helper structure to pass arguments to particle selection GPU method
struct selector_args
    {
    selector_args(const unsigned int _N,
                  unsigned int & _num_members,
                  float4 *_d_pos,
                  unsigned int *_d_tag,
                  unsigned int *_d_body,
                  unsigned int *_d_member_tag)
        : N(_N),
          num_members(_num_members),
          d_pos(_d_pos),
          d_tag(_d_tag),
          d_body(_d_body),
          d_member_tag(_d_member_tag)
        {
        }
    const unsigned int N;        //!< Number of particles in local box
    unsigned int &num_members;   //!< number of members included (output value)
    float4 *d_pos;         //!< array of particle positions
    unsigned int *d_tag;   //!< array of particle tags
    unsigned int *d_body;  //!< array of particle body ids
    unsigned int *d_member_tag;  //!< array to which global tags of local group members are written
    };

#ifdef NVCC
#include<thrust/copy.h>
#include<thrust/iterator/zip_iterator.h>


//! GPU method for selecting global particle tags that are present in the local particle data
/*! \param sel_args standard arguments for this selector
    \param params parameters for the selection rule
*/
template<class rule>
cudaError_t gpu_apply_particle_selection_rule(selector_args sel_args,
                                              const typename rule::param_type params)
    {
    assert(sel_args.d_pos);
    assert(sel_args.d_tag);
    assert(sel_args.d_body);
    assert(sel_args.d_member_tag);
    assert(sel_args.N > 0);

    thrust::device_ptr<float4> pos_ptr(sel_args.d_pos);
    thrust::device_ptr<unsigned int> tag_ptr(sel_args.d_tag);
    thrust::device_ptr<unsigned int> body_ptr(sel_args.d_body);
    thrust::device_ptr<unsigned int> member_tag_ptr(sel_args.d_member_tag);

    sel_args.num_members = thrust::copy_if(tag_ptr,
                        tag_ptr + sel_args.N,
                        thrust::make_zip_iterator( thrust::make_tuple(tag_ptr, body_ptr, pos_ptr )),
                        member_tag_ptr, rule(params)) - member_tag_ptr;
    return cudaSuccess;
    }
#endif

//! GPU method for rebuilding the index list of a ParticleGroup
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned int num_members,
                                   unsigned int *d_member_tag,
                                   unsigned char *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int *d_rtag);


//! Implementation of GPU code for ParticleSelectorGlobalTagListGPU
cudaError_t gpu_particle_selector_tag_list(const unsigned int num_global_members,
                                           const unsigned int N,
                                           unsigned int *d_tag,
                                           unsigned int *d_global_member_tags,
                                           unsigned int& num_local_members,
                                           unsigned int *d_member_tags);

#endif
