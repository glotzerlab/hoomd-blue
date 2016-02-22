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

// Maintainer: jglaser

#include "ForceComposite.h"
#include "NeighborList.h"

/*! \file ForceCompositeGPU.h
    \brief Implementation of a rigid body force compute, GPU version
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ForceCompositeGPU_H__
#define __ForceCompositeGPU_H__

class ForceCompositeGPU : public ForceComposite
    {
    public:
        //! Constructs the compute
        ForceCompositeGPU(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~ForceCompositeGPU();

        //! Update the constituent particles of a composite particle
        /*  Using the position, velocity and orientation of the central particle
         * \param remote If true, consider remote bodies, otherwise bodies
         *        with a local central particle
         */
        virtual void updateCompositeParticles(unsigned int timestep, bool remote);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ForceComposite::setAutotunerParams(enable, period);

            m_tuner_force->setPeriod(period);
            m_tuner_force->setEnabled(enable);

            m_tuner_update->setPeriod(period);
            m_tuner_update->setEnabled(enable);
            }


    protected:
        //! Compute the forces and torques on the central particle
        virtual void computeForces(unsigned int timestep);

        boost::scoped_ptr<Autotuner> m_tuner_force; //!< Autotuner for block size and threads per particle
        boost::scoped_ptr<Autotuner> m_tuner_update;//!< Autotuner for block size of update kernel
    };

//! Exports the ForceCompositeGPU to python
void export_ForceCompositeGPU();

#endif
