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

#include "TwoStepNPTMTK.h"
#include "Variant.h"
#include "ComputeThermo.h"

#ifndef __TWO_STEP_NPT_MTK_GPU_H__
#define __TWO_STEP_NPT_MTK_GPU_H__

/*! \file TwoStepNPTMTKGPU.h
    \brief Declares the TwoStepNPTMTKGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! This is a version of TwoStepNPTMTK that runs on the GPU.
 *
    \ingroup updaters
*/
class TwoStepNPTMTKGPU : public TwoStepNPTMTK
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNPTMTKGPU(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   boost::shared_ptr<ComputeThermo> thermo_group,
                   Scalar tau,
                   Scalar tauP,
                   boost::shared_ptr<Variant> T,
                   boost::shared_ptr<Variant> P,
                   couplingMode couple,
                   unsigned int flags,
                   const bool nph=false);
        virtual ~TwoStepNPTMTKGPU();

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        GPUArray<Scalar> m_scratch;     //!< Scratch space for reduction of squared velocities
        GPUArray<Scalar> m_temperature; //!< Stores temperature after reduction step

        unsigned int m_num_blocks;             //!< Number of blocks participating in the reduction
        unsigned int m_reduction_block_size;   //!< Block size executed
    };

//! Exports the TwoStepNPTMTKGPU class to python
void export_TwoStepNPTMTKGPU();

#endif // #ifndef __TWO_STEP_NPT_MTK_GPU_H__
