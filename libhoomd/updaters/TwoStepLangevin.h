/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

// Maintainer: joaander

#include "TwoStepLangevinBase.h"

#ifndef __TWO_STEP_LANGEVIN_H__
#define __TWO_STEP_LANGEVIN_H__

/*! \file TwoStepLangevin.h
    \brief Declares the TwoStepLangevin class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps with Langevin dynamics
/*! Implements Langevin dynamics.

    Langevin dynamics modifies standard NVE integration with two additional forces, a random force and a drag force.
    This implementation is very similar to TwoStepNVE with the additional forces. Note that this is not a really proper
    Langevin integrator, but it works well in practice.

    \ingroup updaters
*/
class TwoStepLangevin : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepLangevin(boost::shared_ptr<SystemDefinition> sysdef,
                     boost::shared_ptr<ParticleGroup> group,
                     boost::shared_ptr<Variant> T,
                     unsigned int seed,
                     bool use_lambda,
                     Scalar lambda,
                     bool noiseless_t,
                     bool noiseless_r,
                     const std::string& suffix = std::string(""));
        virtual ~TwoStepLangevin();

        //! Turn on or off Tally
        /*! \param tally if true, tallies energy exchange from the thermal reservoir */
        void setTally(bool tally)
            {
            m_tally= tally;
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        Scalar m_reservoir_energy;         //!< The energy of the reservoir the system is coupled to.
        Scalar m_extra_energy_overdeltaT;  //!< An energy packet that isn't added until the next time step
        bool m_tally;                      //!< If true, changes to the energy of the reservoir are calculated
        std::string m_log_name;            //!< Name of the reservoir quantity that we log
        bool m_noiseless_t;
        bool m_noiseless_r;
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepLangevin();

#endif // #ifndef __TWO_STEP_LANGEVIN_H__
