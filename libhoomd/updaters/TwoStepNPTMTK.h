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

// Maintainer: jglaser

#include "IntegrationMethodTwoStep.h"
#include "Variant.h"
#include "ComputeThermo.h"

#ifndef __TWO_STEP_NPT_MTK_H__
#define __TWO_STEP_NPT_MTK_H__

/*! \file TwoStepNPTMTK.h
    \brief Declares the TwoStepNPTMTK class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! Implements the Martyna Tobias Klein (MTK) equations for rigorous integration in the NPT ensemble.
    The update equations are derived from a strictly measure-preserving and
    time-reversal symmetric integration scheme, closely following the one proposed by Tuckerman,
    Alejandre et al..
    Supports anisotropic (orthorhombic or tetragonal) integration modes, by implementing a special
    version of the the fully flexible cell update equations proposed in Yu, Alejandre et al.

    \cite{Martyna1994}
    \cite{Tuckerman2006}
    \cite{Yu2010}
    \ingroup updaters
*/
class TwoStepNPTMTK : public IntegrationMethodTwoStep
    {
    public:
        //! Enum to specify possible integration modes
        enum integrationMode
            {
            cubic = 0,
            orthorhombic,
            tetragonal
            };

        //! Constructs the integration method and associates it with the system
        TwoStepNPTMTK(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   boost::shared_ptr<ComputeThermo> thermo_group,
                   Scalar tau,
                   Scalar tauP,
                   boost::shared_ptr<Variant> T,
                   boost::shared_ptr<Variant> P,
                   integrationMode mode);
        virtual ~TwoStepNPTMTK();

        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(boost::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Update the pressure
        /*! \param P New pressure to set
        */
        virtual void setP(boost::shared_ptr<Variant> P)
            {
            m_P = P;
            }

        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }

        //! Update the nuP value
        /*! \param tauP New pressure constant to set
        */
        virtual void setTauP(Scalar tauP)
            {
            m_tauP = tauP;
            }

        //! Set the partial scale option
        /*! \param partial_scale New partial_scale option to set
        */
        void setPartialScale(bool partial_scale)
            {
            m_exec_conf->msg->error() << "integrate.npt_mtk: partial_scale option not supported" << endl;
            throw runtime_error("Error setting params in TwoStepNPTMTK");
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Get needed pdata flags
        /*! TwoStepNPTMTK needs the pressure, so the isotropic_virial or pressure_tensor flag is set,
            depending on the integration mode
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            if (m_mode == cubic)
                flags[pdata_flag::isotropic_virial] = 1;
            else if (m_mode == orthorhombic || m_mode == tetragonal)
                flags[pdata_flag::pressure_tensor] = 1;
            return flags;
            }

        //! Returns a list of log quantities this compute calculates
        std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

    protected:
        boost::shared_ptr<ComputeThermo> m_thermo_group;   //!< ComputeThermo operating on the integrated group

        Scalar m_tau;                   //!< tau value for Nose-Hoover
        Scalar m_tauP;                  //!< tauP value for the barostat
        boost::shared_ptr<Variant> m_T; //!< Temperature set point
        boost::shared_ptr<Variant> m_P; //!< Pressure set point
        Scalar m_curr_group_T;          //!< Current group temperature
        Scalar m_curr_P;                //!< Current isotropic system pressure
        Scalar3 m_curr_P_diag;          //!< Current pressure tensor (diagonal elements)
        Scalar m_V;                     //!< Current volume
        Scalar3 m_L;                    //!< Box lengths

        Scalar3 m_exp_v_fac;            //!< Factor multiplying velocity
        Scalar3 m_sinhx_fac_v;          //!< Factor multiplying acceleration

        integrationMode m_mode;         //!< Current integration mode

        std::vector<std::string> m_log_names; //!< Name of the barostat and thermostat quantities that we log
    };

//! Exports the TwoStepNPTMTK class to python
void export_TwoStepNPTMTK();

#endif // #ifndef __TWO_STEP_NPT_MTK_H__

