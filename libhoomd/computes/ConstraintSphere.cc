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

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "ConstraintSphere.h"
#include "EvaluatorConstraint.h"
#include "EvaluatorConstraintSphere.h"

using namespace std;

/*! \file ConstraintSphere.cc
    \brief Contains code for the ConstraintSphere class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the sphere
    \param r radius of the sphere
*/
ConstraintSphere::ConstraintSphere(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   Scalar3 P,
                                   Scalar r)
        : ForceConstraint(sysdef), m_group(group), m_P(P), m_r(r)
    {
    }

/*!
    \param P position of the sphere
    \param r radius of the sphere
*/
void ConstraintSphere::setSphere(Scalar3 P, Scalar r)
    {
    m_P = P;
    m_r = r;
    }

/*! ConstraintSphere removes 1 degree of freedom per particle in the group
*/
unsigned int ConstraintSphere::getNDOFRemoved()
    {
    return m_group->getNumMembers();
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintSphere::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    if (m_prof) m_prof->push("ConstraintSphere");
    
    assert(m_pdata);
    // access the particle data arrays
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);
    
    // need to start from a zero force
    // MEM TRANSFER: 5*N Scalars
    memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_pe, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_virial, 0, sizeof(Scalar) * m_pdata->getN());
    
    // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(arrays.x[j], arrays.y[j], arrays.z[j]);
        Scalar3 V = make_scalar3(arrays.vx[j], arrays.vy[j], arrays.vz[j]);
        Scalar3 F = make_scalar3(h_net_force.data[j].x, h_net_force.data[j].y, h_net_force.data[j].z);
        Scalar m = arrays.mass[j];
        
        // evaluate the constraint position
        EvaluatorConstraint constraint(X, V, F, m, m_deltaT);
        EvaluatorConstraintSphere sphere(m_P, m_r);
        Scalar3 C = sphere.evalClosest(constraint.evalU());
        
        // evaluate the constraint force
        Scalar3 FC;
        Scalar virial;
        constraint.evalConstraintForce(FC, virial, C);
        
        // apply the constraint force
        m_fx[j] = FC.x;
        m_fy[j] = FC.y;
        m_fz[j] = FC.z;
        m_virial[j] = virial;
        }
        
    m_pdata->release();

    
    #ifdef ENABLE_CUDA
    // the data is now only up to date on the CPU
    m_data_location = cpu;
    #endif
    
    if (m_prof)
        m_prof->pop();
    }


void export_ConstraintSphere()
    {
    class_< ConstraintSphere, boost::shared_ptr<ConstraintSphere>, bases<ForceConstraint>, boost::noncopyable >
    ("ConstraintSphere", init< boost::shared_ptr<SystemDefinition>,
                                                 boost::shared_ptr<ParticleGroup>,
                                                 Scalar3,
                                                 Scalar >())
    .def("setSphere", &ConstraintSphere::setSphere)
    .def("getNDOFRemoved", &ConstraintSphere::getNDOFRemoved)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

