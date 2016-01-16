/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

/*! \file TwoStepNHRigid.h
    \brief Declares an updater that implements NVE/NVT/NPT/NPH dynamics for rigid bodies
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_NH_RIGID_H__
#define __TWO_STEP_NH_RIGID_H__

#include "RigidData.h"
#include "GPUArray.h"
#include "RigidBodyGroup.h"
#include "Variant.h"
#include "ComputeThermo.h"

/*! \file TwoStepNHRigid.h
 \brief Defines the TwoStepNHRigid class, the base class for NVE, NVT, NPT and NPH rigid body integrators
 */

//! Integrates part of the system forward in two steps
/*! Implements velocity-Verlet integration through the IntegrationMethodTwoStep interface
    for rigid bodies
 \ingroup updaters
*/
class TwoStepNHRigid : public IntegrationMethodTwoStep
    {
    public:
        //! Specify possible couplings between the diagonal elements of the pressure tensor
        enum couplingMode
            {
            couple_none = 0,
            couple_xy = 1,
            couple_xz = 2,
            couple_yz = 3,
            couple_xyz = 4
            };

        /*! Flags to indicate which degrees of freedom of the simulation box should be put under
            barostat control
         */
        enum baroFlags
            {
            baro_x = 1,
            baro_y = 2,
            baro_z = 4,
            baro_xy = 8,
            baro_xz = 16,
            baro_yz = 32
            };

        //! Constructor
        TwoStepNHRigid(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       const std::string& suffix,
                       unsigned int tchain,
                       unsigned int pchain,
                       unsigned int iter);
        virtual ~TwoStepNHRigid();

        //! Computes the initial net forces, torques and angular momenta
       virtual void setup();

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
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
            m_tfreq = tau;
            }

        //! Update the nuP value
        /*! \param tauP New pressure constant to set
        */
        virtual void setTauP(Scalar tauP)
            {
            m_pfreq = tauP;
            }

        //! Set the partial scale option
        /*! \param partial_scale New partial_scale option to set
        */
        void setPartialScale(bool partial_scale)
            {
            m_partial_scale = partial_scale;
            }

        //! Update thermostats
        virtual void update_nh_tchain(Scalar akin_t, Scalar akin_r, unsigned int timestep);

        //! Update the thermostat chain coupled with barostat
        virtual void update_nh_pchain(unsigned int timestep);

        //! Update barostat
        virtual void update_nh_barostat(Scalar akin_t, Scalar akin_r);

        //! Remap the particles from the old box to the new one
        void remap();

        //! Compute current pressure
        virtual void compute_current_pressure(unsigned int timestep);

        //! Compute current pressure
        virtual void compute_target_pressure(unsigned int timestep);

        //! Get needed pdata flags
        /*! NPT and NPH need the pressure, so the isotropic_virial flag is set
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            if (m_pstat) flags[pdata_flag::pressure_tensor] = 1;
            return flags;
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

    protected:
        unsigned int m_n_bodies;                    //!< Number of rigid bodies
        boost::shared_ptr<RigidData> m_rigid_data;  //!< Pointer to rigid data
        boost::shared_ptr<ParticleData> m_pdata;    //!< Pointer to particle data
        boost::shared_ptr<RigidBodyGroup> m_body_group; //!< Group of rigid bodies to work with

        bool m_first_step;                          //!< True if first step

        bool m_tstat;                                //!< True if using thermostat (NVT or NPT)
        bool m_pstat;                                //!< True if using barostat (NPT or NPH)
        boost::shared_ptr<ComputeThermo> m_thermo_group;   //!< ComputeThermo operating on the integrated group
        boost::shared_ptr<ComputeThermo> m_thermo_all;     //!< ComputeThermo operating on the group of all particles

        bool m_partial_scale;                       //!< True if only the particles in the group should be scaled to the new box
        couplingMode m_couple;                      //!< Coupling of diagonal elements
        unsigned int m_flags;                       //!< Coupling flags for barostat
        unsigned int m_pdim;
        Scalar m_tfreq;                             //!< tau value for Nose-Hoover
        Scalar m_pfreq;                             //!< tauP value for the barostat
        boost::shared_ptr<Variant> m_temperature;   //!< Temperature set point
        boost::shared_ptr<Variant> m_pressure;      //!< Pressure set point
        Scalar m_curr_group_T;                      //!< Current group temperature
        Scalar m_curr_P[6];                         //!< Current system pressure
        Scalar m_target_P[6];                       //!< Target pressure tensor
        Scalar m_p_hydro;                           //!< Hydrostatic pressure

        Scalar m_boltz;                             //!< Boltzmann constant
        Scalar m_nf_t;                              //!< Translational degrees of freedom
        Scalar m_nf_r;                              //!< Rotational degrees of freedom
        Scalar m_g_f;                               //!< Total degrees of freedom
        Scalar m_akin_t;                            //!< Translational kinetic energy
        Scalar m_akin_r;                            //!< Rotational kinetic energy
        Scalar m_dt_half;                           //!< Half time step
        Scalar m_mtk_term1, m_mtk_term2;            //!< MTK terms

        unsigned int m_dimension;                   //!< System dimension
        Scalar m_dof;                               //!< Total number degrees of freedom used for system temperature compute
        unsigned int m_tchain;                      //!< Number of thermostats
        unsigned int m_pchain;                      //!< Number of thermostats
        unsigned int m_iter;                        //!< Number of iterations
        unsigned int m_order;                       //!< Number of thermostat per chain

        Scalar m_dilation[6];                       //!< Box size change, xx, yy, zz, xy, xz, yz
        Scalar m_epsilon[6];                        //!< Volume scaling "position"
        Scalar m_epsilon_dot[6];                    //!< Volume scaling "velocity"
        Scalar m_epsilon_mass[6];                   //!< Volume scaling "mass"

        Scalar* m_q_t;                              //!< Thermostat translational mass
        Scalar* m_q_r;                              //!< Thermostat rotational mass
        Scalar* m_q_b;                              //!< Thermostat mass, which is coupled with the barostat
        Scalar* m_eta_t;                            //!< Thermostat translational position
        Scalar* m_eta_r;                            //!< Thermostat rotational position
        Scalar* m_eta_b;                            //!< Thermostat position, which is coupled with the barostat
        Scalar* m_eta_dot_t;                        //!< Thermostat translational velocity
        Scalar* m_eta_dot_r;                        //!< Thermostat rotational velocity
        Scalar* m_eta_dot_b;                        //!< Thermostat velocity, which is coupled with the barostat
        Scalar* m_f_eta_t;                          //!< Thermostat translational force
        Scalar* m_f_eta_r;                          //!< Thermostat rotational force
        Scalar* m_f_eta_b;                          //!< Thermostat force, which is coupled with the barostat

        Scalar* m_w;                                //!< Thermostat chain multi-step integration coeffs
        Scalar* m_wdti1;                            //!< Thermostat chain multi-step integration coeffs
        Scalar* m_wdti2;                            //!< Thermostat chain multi-step integration coeffs
        Scalar* m_wdti4;                            //!< Thermostat chain multi-step integration coeffs

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
        virtual void setRestartIntegratorVariables() {}
        void allocate_tchain();
        void allocate_pchain();
        void deallocate_tchain();
        void deallocate_pchain();

        //! Names of log variables
        std::vector<std::string> m_log_names;
    };

//! Exports the TwoStepNHRigid class to python
void export_TwoStepNHRigid();

#endif

