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

// $Id: TwoStepNVERigid.h 2159 2009-11-19 14:32:38Z ndtrung $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/rigid-bodies/src/updaters/TwoStepNVERigid.h $

/*! \file TwoStepNVERigid.h
    \brief Declares an updater that implements NVE dynamics for rigid bodies
*/

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_NVE_RIGID_H__
#define __TWO_STEP_NVE_RIGID_H__

#include "RigidData.h"
#include "GPUArray.h"

/*! \file TwoStepNVERigid.h
 \brief Declares the TwoStepNVERigid class
 */

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface
 
 \ingroup updaters
*/
class TwoStepNVERigid : public IntegrationMethodTwoStep
    {
    public:
        //! Constructor
        TwoStepNVERigid(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group);
        
        //! Sets the movement limit
        void setLimit(Scalar limit);
        
        //! Removes the limit
        void removeLimit();
        
		//! Sets the zero force option
        /*! \param zero_force Set to true to specify that the integration with a zero net force on each of the particles
		 in the group
		 */
        void setZeroForce(bool zero_force) 
		{
            m_zero_force = zero_force;
		}

        //! Setup the initial net forces, torques and angular momenta
       virtual void setup();
        
		//! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);
        
        //! Performs the second step of the 
		virtual void integrateStepTwo(unsigned int timestep);        
       
		//! Summing the net forces and torques
        void computeForceAndTorque();
        
    protected:
        //! Set positions and velocities for particles in rigid bodies at the first step
        void set_xv();
        
        //! Set velocities for particles in rigid bodies at the second step
        void set_v();
        
        //! Convert quaternion to rotation matrix
        void exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space);
        
        //! Compute angular velocity from angular momentum and rotation matrix
        void computeAngularVelocity(Scalar4 &angmom, Scalar4 &moment_inertia, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space,Scalar4 &angvel);
        
        //! Advance quaternion using Richarson's correction
        void advanceQuaternion(Scalar4 &angmom, Scalar4 &moment_inertia, Scalar4 &angvel, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space, Scalar4 &quat);
        
        //! Quaternion multiply: c = a times b
        void multiply(Scalar4 &a, Scalar4 &b, Scalar4 &c);
        
        //! Quaternion normalize
        void normalize(Scalar4 &q);
        
        unsigned int m_n_bodies;                    //!< Number of rigid bodies
        boost::shared_ptr<RigidData> m_rigid_data;  //!< Pointer to rigid data
        boost::shared_ptr<ParticleData> m_pdata;    //!< Pointer to particle data
	    
        bool m_first_step;  //!< True if first step
        bool m_limit;       //!< True if we should limit the distance a particle moves in one step
        Scalar m_limit_val; //!< The maximum distance a particle is to move in one step
        bool m_zero_force;  //!< True if the integration step should ignore computed forces
    };

//! Exports the TwoStepNVERigid class to python
void export_TwoStepNVERigid();

#endif

