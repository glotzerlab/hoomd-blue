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

#include "TwoStepNVE.h"
#include "Variant.h"
#include "saruprng.h"

#ifndef __TWO_STEP_BDNVT_H__
#define __TWO_STEP_BDNVT_H__

/*! \file TwoStepBDNVT.h
    \brief Declares the TwoStepBDNVT class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps in the NVT ensemble (via brownian dynamics)
/*! Implements velocity-verlet NVE integration with additional brownian dynamics forces through the
    IntegrationMethodTwoStep interface

    Brownian dyanmics modifies standard NVE integration with two additional forces, a random force and a drag force.
    To implement this as simply as possible, we will leveraging the existing TwoStepNVE clas and derive from it. The
    additions needed are a random number generator and some storage for gamma and temperature settings. The NVE
    integration is modified by overrideing integrateStepTwo() to add in the needed bd forces.

    \ingroup updaters
*/
class TwoStepBDNVT : public TwoStepNVE
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepBDNVT(boost::shared_ptr<SystemDefinition> sysdef,
                     boost::shared_ptr<ParticleGroup> group,
                     boost::shared_ptr<Variant> T,
                     unsigned int seed,
                     bool gamma_diam,
                     const std::string& suffix = std::string(""));
        virtual ~TwoStepBDNVT();

        //! Set a new temperature
        /*! \param T new temperature to set */
        void setT(boost::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Sets gamma for a given particle type
        void setGamma(unsigned int typ, Scalar gamma);

        //! Turn on or off Tally
        /*! \param tally if true, tallies energy exchange from bd thermal reservoir */
        void setTally(bool tally)
            {
            m_tally= tally;
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        boost::shared_ptr<Variant> m_T;   //!< The Temperature of the Stochastic Bath
        unsigned int m_seed;              //!< The seed for the RNG of the Stochastic Bath
        bool m_gamma_diam;                //!< flag to enable gamma set to the diameter of each particle
        Scalar m_reservoir_energy;         //!< The energy of the reservoir the bd couples the system to.
        Scalar m_extra_energy_overdeltaT;             //!< An energy packet that isn't added until the next time step
        bool m_tally;                      //!< If true, changes to the energy of the reservoir are calculated
        std::string m_log_name;           //!< Name of the reservior quantity that we log

        GPUArray<Scalar> m_gamma;         //!< List of per type gammas to use
    };

//! Exports the TwoStepBDNVT class to python
void export_TwoStepBDNVT();

#endif // #ifndef __TWO_STEP_BDNVT_H__
