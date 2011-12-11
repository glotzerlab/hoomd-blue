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

// Maintainer: joaander

#include "IntegrationMethodTwoStep.h"
#include "Variant.h"
#include "ComputeThermo.h"

#ifndef __TWO_STEP_NPH_H__
#define __TWO_STEP_NPH_H__

/*! \file TwoStepNPH.h
    \brief Declares the TwoStepNPH class
*/

//! Integrates part of the system forward in two steps in the NPH ensemble
/*! Implements explicitly reversible sympletic NPH integration

    TwoStepNPH can be used in three modes:
    - isotropic volume flucutations, cubic box
    - anisotropic volume fluctuations, orthorhombic box
      (three independent lengths)
    - anisotropic volume fluctuations, tetragonal box
      (two independent lengths, Ly = Lz)

    The first mode is an implementation of the Anderson barostat,
    the second and the third mode are based on the adapted Parrinello-Rahman
    equations of motion.

    The integration scheme used to implement the equations of motion is
    explicitly reversible and measure-preserving. It is based
    on the Trotter expansion technique introduced by Tuckerman et al. J. Chem.
    Phys.  97, pp. 1990 (1992).

    Integrator variables mapping:

     - [0] -> etax (momentum conjugate to box length in x direction)
     - [1] -> etay (momentum conjugate to box length in y direction)
     - [2] -> etaz (momentum conjugate to box length in z direction)

    \ingroup updaters
*/
class TwoStepNPH : public IntegrationMethodTwoStep
    {
    public:
        enum integrationMode
            {
            cubic = 0,
            orthorhombic,
            tetragonal
            };

        //! Constructs the integration method and associates it with the system
        TwoStepNPH(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   boost::shared_ptr<ComputeThermo> thermo,
                   Scalar W,
                   boost::shared_ptr<Variant> P,
                   integrationMode mode,
                   const std::string& suffix);

        virtual ~TwoStepNPH() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Get needed pdata flags
        /*! TwoStepNPT needs the diagonal components of the virial tensor, so the full_virial flag is set
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

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

    protected:
        boost::shared_ptr<ComputeThermo> m_thermo;  //!< ComputeThermo for all particles
        Scalar m_W;                                 //!< the generalized mass of the barostat
        boost::shared_ptr<Variant> m_P;             //!< isotropic Pressure set point
        std::string m_log_name;                     //!< Name of the barostat quantity that we log
        Scalar m_volume;                            //!< current volume
        integrationMode m_mode;                     //!< integration mode
        Scalar3 m_curr_P_diag;                      //!< diagonal elements of the current pressure tensor
        bool m_state_initialized;                   //!< is the integrator initialized?
    };

//! Exports the TwoStepNPH class to python
void export_TwoStepNPH();

#endif // #ifndef __TWO_STEP_NPH_H__
