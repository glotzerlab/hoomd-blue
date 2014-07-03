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

// Maintainer: jglaser

#include "TwoStepNVTMTK.h"

#ifndef __TWO_STEP_NVT_MTK_GPU_H__
#define __TWO_STEP_NVT_MTK_GPU_H__

/*! \file TwoStepNVTMTKGPU.h
    \brief Declares the TwoStepNVTMTKGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Autotuner.h"

//! Integrates part of the system forward in two steps in the NVT ensemble on the GPU
/*! Implements Nose-Hoover NVT integration through the IntegrationMethodTwoStep interface, runs on the GPU

    In order to compute efficiently and limit the number of kernel launches integrateStepOne() performs a first
    pass reduction on the sum of m*v^2 and stores the partial reductions. A second kernel is then launched to recude
    those to a final \a sum2K, which is a scalar but stored in a GPUArray for convenience.

    \ingroup updaters
*/
class TwoStepNVTMTKGPU : public TwoStepNVTMTK
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVTMTKGPU(boost::shared_ptr<SystemDefinition> sysdef,
                      boost::shared_ptr<ParticleGroup> group,
                      boost::shared_ptr<ComputeThermo> thermo,
                      Scalar tau,
                      boost::shared_ptr<Variant> T,
                      const std::string& suffix = std::string(""));
        virtual ~TwoStepNVTMTKGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepNVTMTK::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            }

    protected:
        boost::scoped_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        boost::scoped_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)

        GPUArray<Scalar> m_scratch;     //!< Scratch space for reduction of squared velocities
        GPUArray<Scalar> m_temperature; //!< Stores temperature after reduction step

        unsigned int m_num_blocks;             //!< Number of blocks participating in the reduction
        unsigned int m_reduction_block_size;   //!< Block size executed
    };

//! Exports the TwoStepNVTMTKGPU class to python
void export_TwoStepNVTMTKGPU();

#endif // #ifndef __TWO_STEP_NVT_MTK_GPU_H__
