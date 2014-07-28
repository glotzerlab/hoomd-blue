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

/*! \file TwoStepNVTRigid.h
    \brief Declares an updater that implements NVT dynamics for rigid bodies
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepNVERigid.h"

#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __TWO_STEP_NVT_RIGID_H__
#define __TWO_STEP_NVT_RIGID_H__

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant temperature (NVT) dynamics. Particle positions and velocities are
    updated according to the velocity verlet algorithm. The forces that drive this motion are defined external to this class
    in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be summed to produce a net force on
    each particle.

    Integrator variables mapping:
     - [0] -> eta_t
     - [1] -> eta_r
     - [2] -> eta_dot_t
     - [3] -> eta_dot_r

    \ingroup updaters
*/

class TwoStepNVTRigid : public TwoStepNVERigid
    {
    public:
        //! Constructor
        TwoStepNVTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                        boost::shared_ptr<ParticleGroup> group,
                        boost::shared_ptr<ComputeThermo> thermo,
                        boost::shared_ptr<Variant> T,
                        Scalar tau,
                        const std::string& suffix,
                        bool skip_restart=false);
        ~TwoStepNVTRigid();

        //! Setup the initial net forces, torques and angular momenta
        void setup();

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! First step of velocit Verlet integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Second step of velocit Verlet integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        //! Integrator variables
        virtual void setRestartIntegratorVariables();

        //! Names of log variables
        std::vector<std::string> m_log_names;
    };

//! Exports the TwoStepNVTRigid class to python
void export_TwoStepNVTRigid();

#endif
