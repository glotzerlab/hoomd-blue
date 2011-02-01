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
    validate();
    }

/*!
    \param P position of the sphere
    \param r radius of the sphere
*/
void ConstraintSphere::setSphere(Scalar3 P, Scalar r)
    {
    m_P = P;
    m_r = r;
    validate();
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
    
   
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // Zero data for force calculation.
    memset((void*)h_force,0,sizeof(Scalar4)*m_force.getNumElements);
    memset((void*)h_virial,0,sizeof(Scalar)*m_virial.getNumElements);

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

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
        h_force.data[j].x = FC.x;
        h_force.data[j].y = FC.y;
        h_force.data[j].z = FC.z;
        h_virial.data[j]  = virial;
        }
        
    m_pdata->release();

    if (m_prof)
        m_prof->pop();
    }

/*! Print warning messages if the sphere is outside the box.
    Generate an error if any particle in the group is not near the sphere.
*/
void ConstraintSphere::validate()
    {
    BoxDim box = m_pdata->getBox();
    
    if (m_P.x + m_r > box.xhi || m_P.x - m_r < box.xlo ||
        m_P.y + m_r > box.yhi || m_P.y - m_r < box.ylo ||
        m_P.z + m_r > box.zhi || m_P.z - m_r < box.zlo)
        {
        cout << "***Warning! Sphere constraint is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        }
    
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();

    // for each of the particles in the group
    bool errors = false;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(arrays.x[j], arrays.y[j], arrays.z[j]);
        
        // evaluate the constraint position
        EvaluatorConstraintSphere sphere(m_P, m_r);
        Scalar3 C = sphere.evalClosest(X);
        Scalar3 V;
        V.x = C.x - X.x;
        V.y = C.y - X.y;
        V.z = C.z - X.z;
        Scalar dist = sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
        
        if (dist > 1.0f)
            {
            cerr << endl
                 << "**Error! Particle " << arrays.tag[j] << " is more than 1 unit of distance away from the closest"
                 << " point on the sphere constraint" << endl;
            errors = true;
            }
        }
        
    m_pdata->release();
    
    if (errors)
        {
        cout << endl;
        throw std::runtime_error("Invalid constraint specified");
        }
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

