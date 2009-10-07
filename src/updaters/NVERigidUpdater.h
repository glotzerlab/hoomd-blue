/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: NVERigidUpdater.h 1468 2008-11-13 03:36:34Z joaander $
// $URL: https://svn2.assembla.com/svn/hoomd/trunk/src/updaters/NVEUpdater.h $

/*! \file NVEUpdater.h
    \brief Declares an updater that implements NVE dynamics
*/

#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __NVERIGIDUPDATER_H__
#define __NVERIGIDUPDATER_H__

#include "RigidData.h"
#include "GPUArray.h"

class SystemDefinition;

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant energy (NVE) dynamics. Particle positions and 
	velocities are updated according to the velocity verlet algorithm. The forces that drive this motion are defined 
	external to this class in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be 
	summed to produce a net force on each particle.

    \ingroup updaters
*/
class NVERigidUpdater
    {
    public:
        //! Constructor
        NVERigidUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);
        
        //! Setup the initial net forces, torques and angular momenta
        void setup();
        
        //! First step of velocit Verlet integration
        void initialIntegrate();
        
        //! Summing the net forces and torques
        void computeForceAndTorque();
        
        //! Second step of velocit Verlet integration
        void finalIntegrate();
        
    protected:
        void set_xv();
        void set_v();
        
        //! Private member functions using parameters to avoid duplicate array handles declaration
        void exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space);
        void computeAngularVelocity(Scalar4 &angmom, Scalar4 &moment_inertia,
                                    Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space,
                                    Scalar4 &angvel);
        void advanceQuaternion(Scalar4 &angmom, Scalar4 &moment_inertia, Scalar4 &angvel,
                               Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space, Scalar4 &quat);
        void multiply(Scalar4 &a, Scalar4 &b, Scalar4 &c);
        void normalize(Scalar4 &q);
        
        Scalar m_deltaT;
        unsigned int m_n_bodies;
        boost::shared_ptr<RigidData> m_rigid_data;
        boost::shared_ptr<ParticleData> m_pdata;
        
        GPUArray<Scalar4> m_force;
        GPUArray<Scalar4> m_torque;
    };


#endif

