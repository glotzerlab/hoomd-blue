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

#include "ParticleGroupGPU.h"

/*! \file ParticleGroupGPU.cc
    \brief Defines GPU classes related to ParticleGroup
*/

#ifdef ENABLE_CUDA
/*! \param sysdef System the particles are to be selected from
    \param tag_min Minimum tag to select (inclusive)
    \param tag_max Maximum tag to select (inclusive)
*/
ParticleSelectorTagGPU::ParticleSelectorTagGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                         unsigned int tag_min,
                                         unsigned int tag_max)
    : ParticleSelectorRuleGPU<GlobalTagRule, gpu_select_particles_tag>(sysdef)
    {
    // make a quick check on the sanity of the input data
    if (tag_max < tag_min)
        cout << "***Warning! max < min specified when selecting particle tags" << endl;

    if (tag_max >= m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Cannot select particles with tags larger than the number of particles "
             << endl << endl;
        throw runtime_error("Error selecting particles");
        }

    setParams(make_uint2(tag_min, tag_max));
    }

/*! \param sysdef System the particles are to be selected from
    \param typ_min Minimum type id to select (inclusive)
    \param typ_max Maximum type id to select (inclusive)
    */
ParticleSelectorTypeGPU::ParticleSelectorTypeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                           unsigned int typ_min,
                                           unsigned int typ_max)
    : ParticleSelectorRuleGPU<TypeRule, gpu_select_particles_type>(sysdef)
    {
    // make a quick check on the sanity of the input data
    if (typ_max < typ_min)
        cout << "***Warning! max < min specified when selecting particle types" << endl;

    if (typ_max >= m_pdata->getNTypes())
        cout << "***Warning! Requesting for the selection of a non-existant particle type" << endl;

    setParams(make_uint2(typ_min, typ_max));
    }

/*! \param sysdef System the particles are to be selected from
    \param rigid true selects particles that are in rigid bodies, false selects particles that are not part of a body
*/
ParticleSelectorRigidGPU::ParticleSelectorRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                             bool rigid)
    : ParticleSelectorRuleGPU<RigidRule, gpu_select_particles_body>(sysdef)
    {
    setParams(rigid);
    }

//! GPU Implementation of a ParticleSelector that takes a list of global member tags as input

//! Constructor
ParticleSelectorGlobalTagListGPU::ParticleSelectorGlobalTagListGPU(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list)
    : ParticleSelectorGlobalTagList(sysdef, global_tag_list)
    {
    }

//! Get local group members
unsigned int ParticleSelectorGlobalTagListGPU::getMemberTags(const GPUArray<unsigned int>& member_tags)
    {
    assert(member_tags.getNumElements() >= m_pdata->getN());

    ArrayHandle<unsigned int> d_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

    // global tags of local group members
    ArrayHandle<unsigned int> d_member_tags(member_tags, access_location::device, access_mode::overwrite);

    // global list of group members
    ArrayHandle<unsigned int> d_global_member_tags(m_global_member_tags, access_location::device, access_mode::read);

    unsigned int num_members;
    // calculate intersection of local tags with all group member tags
    gpu_particle_selector_tag_list(m_global_member_tags.getNumElements(),
                                          m_pdata->getN(),
                                          d_tag.data,
                                          d_global_member_tags.data,
                                          num_members,
                                          d_member_tags.data);
    return num_members;
    }
#endif
