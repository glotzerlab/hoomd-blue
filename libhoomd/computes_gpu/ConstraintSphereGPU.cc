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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace boost::python;
using namespace boost;

#include "ConstraintSphereGPU.h"
#include "ConstraintSphereGPU.cuh"

using namespace std;

/*! \file ConstraintSphereGPU.cc
    \brief Contains code for the ConstraintSphereGPU class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the sphere
    \param r radius of the sphere
*/
ConstraintSphereGPU::ConstraintSphereGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                         boost::shared_ptr<ParticleGroup> group,
                                         Scalar3 P,
                                         Scalar r)
        : ConstraintSphere(sysdef, group, P, r), m_block_size(256)
    {
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a ConstraintSphereGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing ConstraintSphereGPU");
        }
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintSphereGPU::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    if (m_prof) m_prof->push(exec_conf, "ConstraintSphere");
    
    assert(m_pdata);
    
    // access the particle data arrays
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

    const GPUArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    // run the kernel in parallel on all GPUs
    gpu_compute_constraint_sphere_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         d_vel.data,
                                         d_net_force.data,
                                         m_P,
                                         m_r,
                                         m_deltaT,
                                         m_block_size);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
   
    if (m_prof)
        m_prof->pop(exec_conf);
    }


void export_ConstraintSphereGPU()
    {
    class_< ConstraintSphereGPU, boost::shared_ptr<ConstraintSphereGPU>, bases<ConstraintSphere>, boost::noncopyable >
    ("ConstraintSphereGPU", init< boost::shared_ptr<SystemDefinition>,
                                  boost::shared_ptr<ParticleGroup>,
                                  Scalar3,
                                  Scalar >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

