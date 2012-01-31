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

/*! \file ParticleGroupGPU.h
    \brief Declares GPU classes related to ParticleGroup
*/


#ifndef __PARTICLE_GROUP_GPU_H__
#define __PARTICLE_GROUP_GPU_H__

#include "ParticleGroup.h"

#ifdef ENABLE_CUDA
#include "AllParticleSelectors.cuh"

template< class T, cudaError_t gpu_psr(selector_args sel_args, const typename T::param_type params) >
class ParticleSelectorRuleGPU : public ParticleSelectorRule<T>
    {
    public:
        //! constructs a particle selector
        //! \param sysdef the system definition to build the particle group from
        ParticleSelectorRuleGPU(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelectorRuleGPU() {}


        //! Get the list of selected tags
        /*! \param the GPU array to store the member tags in
         * \return the number of local particles included
         * \pre  member_tags must be allocated and of sufficient size to accomodate
         *       all local members of the group (i.e.
         *       the current maximum number of particles returned by ParticleData::getMaxN() )
        */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags);
    };

//! Constructor
template< class T, cudaError_t gpu_psr(selector_args sel_args, const typename T::param_type params) >
ParticleSelectorRuleGPU<T, gpu_psr>::ParticleSelectorRuleGPU(boost::shared_ptr<SystemDefinition> sysdef)
    : ParticleSelectorRule<T>(sysdef)
    {
    }

//! get list of selected tags
template< class T, cudaError_t gpu_psr(selector_args sel_args, const typename T::param_type params) >
unsigned int ParticleSelectorRuleGPU<T, gpu_psr>::getMemberTags(const GPUArray<unsigned int>& member_tags)
    {
    assert(this->m_pdata->getExecConf()->isCUDAEnabled());

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(this->m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_global_tag(this->m_pdata->getGlobalTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_tags(member_tags, access_location::device, access_mode::readwrite);

    unsigned int num_members;

    // invoke particle tag selection on the GPU
    gpu_psr(selector_args(this->m_pdata->getN(),
            num_members,
            d_pos.data,
            d_global_tag.data,
            d_body.data,
            d_member_tags.data),
            this->m_params);

    return num_members;
    }

//! ParticleSelector to select particles on the GPU based on the tag rule
class ParticleSelectorTagGPU : public ParticleSelectorRuleGPU<GlobalTagRule, gpu_select_particles_tag>
    {
    public:
        //! Initializes the selector
        ParticleSelectorTagGPU(boost::shared_ptr<SystemDefinition> sysdef, unsigned int tag_min, unsigned int tag_max);
        virtual ~ParticleSelectorTagGPU() {}
    };

//! ParticleSelector to select particles on the GPU based on the type rule
class ParticleSelectorTypeGPU : public ParticleSelectorRuleGPU<TypeRule, gpu_select_particles_type>
    {
    public:
        //! Initializes the selector
        ParticleSelectorTypeGPU(boost::shared_ptr<SystemDefinition> sysdef, unsigned int typ_min, unsigned int typ_max);
        virtual ~ParticleSelectorTypeGPU() {}
    };

//! ParticleSelector to select particles on the GPU based on the body
class ParticleSelectorRigidGPU : public ParticleSelectorRuleGPU<RigidRule, gpu_select_particles_body>
    {
    public:
        //! Initializes the selector
        ParticleSelectorRigidGPU(boost::shared_ptr<SystemDefinition> sysdef, bool rigid);
        virtual ~ParticleSelectorRigidGPU() {}
    };
#endif

#endif
