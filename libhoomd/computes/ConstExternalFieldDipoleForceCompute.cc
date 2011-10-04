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
// Maintainer: grva

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "ConstExternalFieldDipoleForceCompute.h"
#include "QuaternionMath.h"

using namespace std;

/*! \file ConstExternalFieldDipoleForceCompute.cc
    \brief Contains code for the ConstExternalFieldDipoleForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param field_x x component of field
    \param field_y y component of field
    \param field_z z component of field
    \param p p component of field
    \note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstExternalFieldDipoleForceCompute::ConstExternalFieldDipoleForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar field_x=0.0,Scalar field_y=0.0, Scalar field_z=0.0,Scalar p=0.0)
        : ForceCompute(sysdef)
    {
    setParams(field_x,field_y,field_z,p);
    }

/*! \param field_x x component of field
    \param field_y y component of field
    \param field_z z component of field
    \param p p component of field

    f.{x,y,z} are components of the field, f.w is the magnitude of the
    moment in the z direction
*/
void ConstExternalFieldDipoleForceCompute::setParams(Scalar field_x,Scalar field_y, Scalar field_z,Scalar p)
    {
    field=make_scalar4(field_x,field_y,field_z,p);
    }

/*! \brief Compute the torque applied = Cross[p,Field]
    \param timestep Current timestep
*/
void ConstExternalFieldDipoleForceCompute::computeForces(unsigned int timestep)
    {
    // array handles
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);

    // number of particles
    unsigned int num_particles = m_pdata->getN();

    // compute the torques
    for (unsigned int i=0; i<num_particles; i++)
        {
        // rotation operator for this particle
        Scalar4 rot = h_orientation.data[i];
        //Hermitian conjugate
        Scalar4 rot_H;
        // compute the Hermitian conjugate
        quatconj(rot,rot_H);
        // base particle orientation is in z direction
        //
        Scalar4 base = {0,0,1,0};
        // to be filled by the actual moment
        Scalar4 moment;
        // a temporary variable
        Scalar4 temp;
        // do half the rotation
        quatvec(rot,base,temp);
        // do the other half
        quatquat(temp,rot_H,moment);

        // tricky bit:
        // the resulting vector is the last three components of moment
        // because we got it by doing a quat * quat

        // Torque = moment X field
        // that means recipe for cross product is
        // [(zz-yw),(xw-zy),(yy-xz)]
        // cf. usual [(yz-zy),(zx-xz),(xy-yx)]
            
        // also field.w stores magnitude of dipole moment, so, here we go

        // reuse temp to compute the torque
        h_torque.data[i].x = field.w*(field.z*moment.z-field.y*moment.w);
        h_torque.data[i].y = field.w*(field.x*moment.w-field.z*moment.y);
        h_torque.data[i].z = field.w*(field.y*moment.y-field.x*moment.z);
        h_torque.data[i].w = Scalar(0);
        }

    }

void export_ConstExternalFieldDipoleForceCompute()
    {
    class_< ConstExternalFieldDipoleForceCompute, boost::shared_ptr<ConstExternalFieldDipoleForceCompute>,
            bases<ForceCompute>, boost::noncopyable >
    ("ConstExternalFieldDipoleForceCompute", init< boost::shared_ptr<SystemDefinition>, Scalar,Scalar,Scalar,Scalar >())
    .def("setParams", &ConstExternalFieldDipoleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

