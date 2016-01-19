/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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


#include "ConstraintEllipsoidGPU.h"
#include "ConstraintEllipsoidGPU.cuh"

#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace boost::python;

using namespace std;

/*! \file ConstraintEllipsoidGPU.cc
    \brief Contains code for the ConstraintEllipsoidGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the Ellipsoid
    \param rx radius of the Ellipsoid in the X direction
    \param ry radius of the Ellipsoid in the Y direction
    \param rz radius of the Ellipsoid in the Z direction
    NOTE: For the algorithm to work, we must have _rx >= _rz, ry >= _rz, and _rz > 0.
*/
ConstraintEllipsoidGPU::ConstraintEllipsoidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   Scalar3 P,
                                   Scalar rx,
                                   Scalar ry,
                                   Scalar rz)
        : ConstraintEllipsoid(sysdef, group, P, rx, ry, rz), m_block_size(256)
{
    if (!m_exec_conf->isCUDAEnabled())
    {
        m_exec_conf->msg->error() << "Creating a ConstraintEllipsoidGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ConstraintEllipsoidGPU");
    }
}

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintEllipsoidGPU::update(unsigned int timestep)
{
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("ConstraintEllipsoid");

    assert(m_pdata);

    // access the particle data arrays
    const GPUArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

    // run the kernel in parallel on all GPUs
    gpu_compute_constraint_ellipsoid_constraint(d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_P,
                                         m_rx,
                                         m_ry,
                                         m_rz,
                                         m_block_size);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);    
}

void export_ConstraintEllipsoidGPU()
{
    class_< ConstraintEllipsoidGPU, boost::shared_ptr<ConstraintEllipsoidGPU>, bases<ConstraintEllipsoid>, boost::noncopyable >
    ("ConstraintEllipsoidGPU", init< boost::shared_ptr<SystemDefinition>,
                                                 boost::shared_ptr<ParticleGroup>,
                                                 Scalar3,
                                                 Scalar,
                                                 Scalar,
                                                 Scalar >())
    ;
}
