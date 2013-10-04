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

// Maintainer: ndtrung

/*! \file TwoStepBDNVTRigid.h
    \brief Declares an updater that implements BD NVT dynamics for rigid bodies
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepNVERigid.h"
#include "Variant.h"
#include "saruprng.h"

#ifndef __TWO_STEP_BD_NVT_RIGID_H__
#define __TWO_STEP_BD_NVT_RIGID_H__


/*! \file TwoStepBDNVTRigid.h
 \brief Declares the TwoStepBDNVTRigid class
 */

//! Integrates part of the system forward in two steps in the NVE ensemble with Langevin thermostat
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface

 \ingroup updaters
*/
class TwoStepBDNVTRigid : public TwoStepNVERigid
    {
    public:
        //! Constructor
        TwoStepBDNVTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                          boost::shared_ptr<ParticleGroup> group,
                          boost::shared_ptr<Variant> T,
                          unsigned int seed,
                          bool gamma_diam);
        virtual ~TwoStepBDNVTRigid();

        //! Sets gamma for a given particle type
        void setGamma(unsigned int typ, Scalar gamma);

        //! Performs the second step
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        boost::shared_ptr<Variant> m_T;   //!< The Temperature of the Stochastic Bath
        unsigned int m_seed;              //!< The seed for the RNG of the Stochastic Bath
        bool m_gamma_diam;                //!< flag to enable gamma set to the diameter of each particle

        GPUArray<Scalar> m_gamma;         //!< List of per type gammas to use
    };

//! Exports the TwoStepBDNVTRigid class to python
void export_TwoStepBDNVTRigid();

#endif
