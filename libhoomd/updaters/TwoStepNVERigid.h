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

// Maintainer: ndtrung

/*! \file TwoStepNVERigid.h
    \brief Declares an updater that implements NVE dynamics for rigid bodies
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_NVE_RIGID_H__
#define __TWO_STEP_NVE_RIGID_H__

#include "RigidData.h"
#include "GPUArray.h"
#include "RigidBodyGroup.h"
#include "Variant.h"
#include "ComputeThermo.h"

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
        TwoStepNVERigid(boost::shared_ptr<SystemDefinition> sysdef,
                        boost::shared_ptr<ParticleGroup> group,
                        bool skip_restart=false);
        virtual ~TwoStepNVERigid();

        //! Computes the initial net forces, torques and angular momenta
       virtual void setup();

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the
        virtual void integrateStepTwo(unsigned int timestep);

        //! Computes the body forces and torques
        void computeForceAndTorque(unsigned int timestep);

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF(boost::shared_ptr<ParticleGroup> query_group);

        //! Validate that all members in the particle group are valid (throw an exception if they are not)
        virtual void validateGroup();

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

        //! Update thermostats
        void update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep);

        //! Update the thermostat chain coupled with barostat
        void update_nhcb(unsigned int timestep);

        //! Remap the particles from the old box to the new one
        void remap();

        //! Adjust rigid body center of mass with deformed box
        void deform(unsigned int flag);

        //! Get needed pdata flags
        /*! NPT and NPH need the pressure, so the isotropic_virial flag is set
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            flags[pdata_flag::isotropic_virial] = 1;
            return flags;
            }

    protected:
        unsigned int m_n_bodies;                    //!< Number of rigid bodies
        boost::shared_ptr<RigidData> m_rigid_data;  //!< Pointer to rigid data
        boost::shared_ptr<ParticleData> m_pdata;    //!< Pointer to particle data
        boost::shared_ptr<RigidBodyGroup> m_body_group; //!< Group of rigid bodies to work with

        bool m_first_step;                          //!< True if first step

        bool t_stat;                                //!< True if using thermostat (NVT or NPT)
        bool p_stat;                                //!< True if using barostat (NPT or NPH)
        boost::shared_ptr<ComputeThermo> m_thermo_group;   //!< ComputeThermo operating on the integrated group
        boost::shared_ptr<ComputeThermo> m_thermo_all;     //!< ComputeThermo operating on the group of all particles

        bool m_partial_scale;                       //!< True if only the particles in the group should be scaled to the new box
        Scalar t_freq;                              //!< tau value for Nose-Hoover
        Scalar p_freq;                              //!< tauP value for the barostat
        boost::shared_ptr<Variant> m_temperature;   //!< Temperature set point
        boost::shared_ptr<Variant> m_pressure;      //!< Pressure set point
        Scalar m_curr_group_T;                      //!< Current group temperature
        Scalar m_curr_P;                            //!< Current system pressure

        Scalar boltz;                               //!< Boltzmann constant
        Scalar nf_t;                                //!< Translational degrees of freedom
        Scalar nf_r;                                //!< Rotational degrees of freedom
        Scalar g_f, onednft, onednfr;

        unsigned int dimension;                     //!< System dimension
        Scalar m_dof;                               //!< Total number degrees of freedom used for system temperature compute
        unsigned int chain;                         //!< Number of thermostats
        unsigned int iter;                          //!< Number of iterations
        unsigned int order;                         //!< Number of thermostat per chain

        Scalar  dilation;                           //!< Box size change
        Scalar  epsilon;                            //!< Volume scaling "position"
        Scalar  epsilon_dot;                        //!< Volume scaling "velocity"
        Scalar  f_epsilon;                          //!< Volume scaling "force"
        Scalar  W;                                  //!< Volume scaling "mass"

        Scalar* q_t;                                //!< Thermostat translational mass
        Scalar* q_r;                                //!< Thermostat rotational mass
        Scalar* q_b;                                //!< Thermostat mass, which is coupled with the barostat
        Scalar* eta_t;                              //!< Thermostat translational position
        Scalar* eta_r;                              //!< Thermostat rotational position
        Scalar* eta_b;                              //!< Thermostat position, which is coupled with the barostat
        Scalar* eta_dot_t;                          //!< Thermostat translational velocity
        Scalar* eta_dot_r;                          //!< Thermostat rotational velocity
        Scalar* eta_dot_b;                          //!< Thermostat velocity, which is coupled with the barostat
        Scalar* f_eta_t;                            //!< Thermostat translational force
        Scalar* f_eta_r;                            //!< Thermostat rotational force
        Scalar* f_eta_b;                            //!< Thermostat force, which is coupled with the barostat

        Scalar* w;                                  //!< Thermostat chain multi-step integration coeffs
        Scalar* wdti1;                              //!< Thermostat chain multi-step integration coeffs
        Scalar* wdti2;                              //!< Thermostat chain multi-step integration coeffs
        Scalar* wdti4;                              //!< Thermostat chain multi-step integration coeffs

        //! Maclaurin expansion
        inline Scalar maclaurin_series(Scalar x)
            {
            Scalar x2, x4;
            x2 = x * x;
            x4 = x2 * x2;
            return (1.0 + (1.0/6.0) * x2 + (1.0/120.0) * x4 + (1.0/5040.0) * x2 * x4 + (1.0/362880.0) * x4 * x4);
            }

    protected:
        //! Integrator variables
        virtual void setRestartIntegratorVariables();

    };

//! Exports the TwoStepNVERigid class to python
void export_TwoStepNVERigid();

#endif
