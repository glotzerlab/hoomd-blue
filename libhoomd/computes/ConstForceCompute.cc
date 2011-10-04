/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

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
using namespace boost::python;

#include "ConstForceCompute.h"

using namespace std;

/*! \file ConstForceCompute.cc
    \brief Contains code for the ConstForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstForceCompute::ConstForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar fx, Scalar fy, Scalar fz)
        : ForceCompute(sysdef)
    {
    setForce(fx,fy,fz);
    }

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group A group of particles
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstForceCompute::ConstForceCompute(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz)
        : ForceCompute(sysdef)
    {
    setGroupForce(group,fx,fy,fz);
    }

/*! \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstForceCompute::setForce(Scalar fx, Scalar fy, Scalar fz)
    {
    assert(m_pdata != NULL);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite); 
    //Don't need to zero data for force calculation.

    assert(h_force.data);

    // setting the force is simple, just fill out every element of the force array
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_force.data[i].x = fx;
        h_force.data[i].y = fy;
        h_force.data[i].z = fz;
        h_force.data[i].w = 0;
        }
   }

/*! \param i Index of the particle to set
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstForceCompute::setParticleForce(unsigned int i, Scalar fx, Scalar fy, Scalar fz)
    {
        
    assert(m_pdata != NULL);
    assert(i < m_pdata->getN());

    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite); 
    assert(h_force.data);

    h_force.data[i].x = fx;
    h_force.data[i].y = fy;
    h_force.data[i].z = fz;
    h_force.data[i].w = 0;
    }

/*! \param group Group to set the force for
    \param fx x-component of the force
    \param fy y-component of the force
    \param fz z-component of the force
*/
void ConstForceCompute::setGroupForce(boost::shared_ptr<ParticleGroup> group, Scalar fx, Scalar fy, Scalar fz)
    {
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);

    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();

    m_fx = fx;
    m_fy = fy;
    m_fz = fz;
    m_group = group;

    // Reset force array
    for (unsigned int i = 0;i < m_pdata->getN();i++)
        {
        h_force.data[i].x = 0;
        h_force.data[i].y = 0;
        h_force.data[i].z = 0;
        h_force.data[i].w = 0;
        }

    for (unsigned int i = 0; i < group->getNumMembers(); i++)
        {
        // get the tag for the current group member from the group
        unsigned int tag = group->getMemberTag(i);
        // identify the index of the current particle tag
        unsigned int idx = arrays.rtag[tag];
        h_force.data[idx].x = fx;
        h_force.data[idx].y = fy;
        h_force.data[idx].z = fz;
        h_force.data[idx].w = 0;
        }

    m_pdata->release();
    }

void ConstForceCompute::rearrangeForces()
    {
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();

    // Reset force array
    for (unsigned int i = 0;i < m_pdata->getN();i++)
        {
        h_force.data[i].x = 0;
        h_force.data[i].y = 0;
        h_force.data[i].z = 0;
        h_force.data[i].w = 0;
        }

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        // get the tag for the current group member from the group
        unsigned int tag = m_group->getMemberTag(i);
        // identify the index of the current particle tag
        unsigned int idx = arrays.rtag[tag];
        h_force.data[idx].x = m_fx;
        h_force.data[idx].y = m_fy;
        h_force.data[idx].z = m_fz;
        h_force.data[idx].w = 0;
        }

    m_pdata->release();

    }

/*! This function calls rearrangeForces() whenever the particles have been sorted
    \param timestep Current timestep
*/
void ConstForceCompute::computeForces(unsigned int timestep)
    {
    if (m_particles_sorted==true) rearrangeForces();
    }


void export_ConstForceCompute()
    {
    class_< ConstForceCompute, boost::shared_ptr<ConstForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("ConstForceCompute", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar >())
    .def("setForce", &ConstForceCompute::setForce)
    .def("setGroupForce", &ConstForceCompute::setGroupForce)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

