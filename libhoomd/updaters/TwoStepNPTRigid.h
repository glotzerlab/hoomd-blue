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

// $Id: TwoStepNPTRigid.h 2593 2010-01-28 21:22:50Z ndtrung $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/branches/rigid-bodies/libhoomd/updaters/TwoStepNPTRigid.h $
// Maintainer: ndtrung

#include "TwoStepNVERigid.h"
#include "Variant.h"

#ifndef __TWO_STEP_NPT_RIGID_H__
#define __TWO_STEP_NPT_RIGID_H__

/*! \file TwoStepNPTRigid.h
    \brief Declares the TwoStepNPTRigid class
*/

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! Implements Nose-Hoover NPT integration through the IntegrationMethodTwoStep interface
    
    This class and TwoStepNVTRigid are supposed to be re-organized due to shared member functions
    
    \ingroup updaters
*/
class TwoStepNPTRigid : public TwoStepNVERigid
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNPTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   Scalar tau,
                   Scalar tauP,
                   boost::shared_ptr<Variant> T,
                   boost::shared_ptr<Variant> P);
        virtual ~TwoStepNPTRigid() {};
        
        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(boost::shared_ptr<Variant> T)
            {
            m_temperature = T;
            }
        
        //! Update the pressure
        /*! \param P New pressure to set
        */
        virtual void setP(boost::shared_ptr<Variant> P)
            {
            m_pressure = P;
            }
        
        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            t_freq = tau;
            }
        
        //! Update the nuP value
        /*! \param tauP New pressure constant to set
        */
        virtual void setTauP(Scalar tauP)
            {
            p_freq = tauP;
            }
        
        //! Set the partial scale option
        /*! \param partial_scale New partial_scale option to set
        */
        void setPartialScale(bool partial_scale)
            {
            m_partial_scale = partial_scale;
            }
        
        //! Computes initial forces and torques and initializes thermostats/barostats
        virtual void setup();
        
        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);
        
        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);
    
        //! Computes pressure
        virtual Scalar computePressure(unsigned int timestep);
        
        //! Computes temperature
        virtual Scalar computeTemperature(unsigned int timestep);
        
    protected:
        bool m_partial_scale;                       //!< True if only the particles in the group should be scaled to the new box
        Scalar t_freq;                              //!< tau value for Nose-Hoover
        Scalar p_freq;                              //!< tauP value for the barostat
        boost::shared_ptr<Variant> m_temperature;   //!< Temperature set point
        boost::shared_ptr<Variant> m_pressure;      //!< Pressure set point
                
    protected:
        //! Set positions and velocities for particles in rigid bodies at the first step
        void set_xv(unsigned int timestep);
        
        //! Set velocities for particles in rigid bodies at the second step
        void set_v(unsigned int timestep);
        
        //! Update thermostats
        void update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep);
        
        //! Update barostats
        void update_nhcb(unsigned int timestep);

        //! Update thermostat momenta and positions
        void no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt);
        
        //! Remap the particles from the old box to the new one
        void remap();
        
        //! Adjust rigid body center of mass with deformed box
        void deform(unsigned int flag);
        
        //! Quaternion multiply
        void quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
        
        //! Inverse quaternion multiply
        void inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
        
        //! Matrix multiply
        void matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
        
        //! Transposed matrix multiply
        void transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
        
        //! Maclaurin expansion
        inline Scalar maclaurin_series(Scalar x);
        
        unsigned int dimension;                     //!< System dimension
        Scalar boltz;                               //!< Boltzmann constant
        Scalar nf_t;                                //!< Translational degrees of freedom        
        Scalar nf_r;                                //!< Rotational degrees of freedom 
        Scalar m_dof;                               //!< Total number degrees of freedom used for system temperature compute 
        unsigned int chain;                         //!< Number of thermostats
        
        Scalar  dilation;                           //!< Box size change
        Scalar  epsilon;                            //!< Volume scaling "position" 
        Scalar  epsilon_dot;                        //!< Volume scaling "velocity" 
        Scalar  f_epsilon;                          //!< Volume scaling "force"
        Scalar  w;                                  //!< Volume scaling "mass"
        GPUArray<Scalar>    q_t;                    //!< Thermostat translational mass
        GPUArray<Scalar>    q_r;                    //!< Thermostat rotational mass
        GPUArray<Scalar>    q_b;                    //!< Barostat rotational mass
        GPUArray<Scalar>    eta_t;                  //!< Thermostat translational position
        GPUArray<Scalar>    eta_r;                  //!< Thermostat rotational position
        GPUArray<Scalar>    eta_b;                  //!< Barostat rotational position
        GPUArray<Scalar>    eta_dot_t;              //!< Thermostat translational velocity
        GPUArray<Scalar>    eta_dot_r;              //!< Thermostat rotational velocity
        GPUArray<Scalar>    eta_dot_b;              //!< Barostat rotational velocity
        GPUArray<Scalar>    f_eta_t;                //!< Thermostat translational force
        GPUArray<Scalar>    f_eta_r;                //!< Thermostat rotational force
        GPUArray<Scalar>    f_eta_b;                //!< Barostat rotational force
        
        GPUArray<Scalar4>   conjqm;                 //!< Thermostat conjugate quaternion momentum
        
        GPUArray<Scalar>    virial_rigid;          //!< Virial contribution from rigid bodies
    };

//! Exports the TwoStepNVTRigid class to python
void export_TwoStepNPTRigid();

#endif // #ifndef __TWO_STEP_NPT_RIGID_H__

