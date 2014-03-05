/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

#include "RigidBodyGroup.h"

#include <vector>
using namespace std;

/*! \file RigidBodyGroup.cc
    \brief Defines the RigidBodyGroup class
*/

/*! \param sysdef System definition to build the group from
    \param group Particle group to pick rigid bodies from
*/
RigidBodyGroup::RigidBodyGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group)
    : m_sysdef(sysdef),
      m_rdata(sysdef->getRigidData()),
      m_pdata(sysdef->getParticleData()),
      m_is_member(m_rdata->getNumBodies())
    {
    m_exec_conf = m_pdata->getExecConf();

    // don't generate the body group unless there are bodies in the simulation
    if (m_rdata->getNumBodies() == 0)
        return;

    // start by initializing a count of the number of particles in the group that belong to each body
    vector<unsigned int> particle_count(m_rdata->getNumBodies());
    particle_count.assign(m_rdata->getNumBodies(), 0);

    {
    ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    for (unsigned int group_idx = 0; group_idx < group->getNumMembers(); group_idx++)
        {
        unsigned int p_index = group->getMemberIndex(group_idx);
        unsigned int b_index = h_body.data[p_index];
        if (b_index == NO_INDEX)
            {
            m_exec_conf->msg->warning() << "group: Attempting to include a free particle in a rigid body group, ignoring particle" << endl
                 << "with tag: " << group->getMemberTag(group_idx) << endl;
            }
        else
            {
            assert(b_index < m_rdata->getNumBodies());
            particle_count[b_index]++;
            }
        }

    }

    // validate that all bodies are completely selected
    // also count up the number of selected bodies
    unsigned int n_selected_bodies = 0;

        {
        ArrayHandle<unsigned int> h_body_size(m_rdata->getBodySize(), access_location::host, access_mode::read);

        for (unsigned int body_idx = 0; body_idx < m_rdata->getNumBodies(); body_idx++)
            {
            if (particle_count[body_idx] != 0)
                {
                n_selected_bodies++;
                m_is_member[body_idx] = true;

                if (particle_count[body_idx] != h_body_size.data[body_idx])
                    m_exec_conf->msg->warning() << "group: Only a portion of body " << body_idx << " is included in the group" << endl;
                }
            }
        }

    // allocate memory for the gpu list
    GPUArray<unsigned int> member_idx(n_selected_bodies, m_pdata->getExecConf());
    m_member_idx.swap(member_idx);

    // assign all of the bodies that belong to the group
    ArrayHandle<unsigned int> h_member_idx(m_member_idx, access_location::host, access_mode::overwrite);
    unsigned int count = 0;
    for (unsigned int body_idx = 0; body_idx < m_rdata->getNumBodies(); body_idx++)
        {
        if (isMember(body_idx))
            {
            h_member_idx.data[count] = body_idx;
            count++;
            }
        }
    }
