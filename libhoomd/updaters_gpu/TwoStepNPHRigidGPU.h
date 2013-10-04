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

#include "TwoStepNPHRigid.h"

#ifndef __TWO_STEP_NPH_RIGID_GPU_H__
#define __TWO_STEP_NPH_RIGID_GPU_H__

/*! \file TwoStepNPHRigidGPU.h
    \brief Declares the TwoStepNPHRigidGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps in the NPH ensemble on the GPU
/*! Implements velocity-verlet NPH integration through the IntegrationMethodTwoStep interface, runs on the GPU

    \ingroup updaters
*/
class TwoStepNPHRigidGPU : public TwoStepNPHRigid
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNPHRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                            boost::shared_ptr<ParticleGroup> group,
                            boost::shared_ptr<ComputeThermo> thermo_group,
                            boost::shared_ptr<ComputeThermo> thermo_all,
                            Scalar tauP,
                            boost::shared_ptr<Variant> P,
                            bool skip_restart=false);

        virtual ~TwoStepNPHRigidGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        GPUArray<Scalar> m_partial_Ksum_t;  //!< Translational kinetic energy per body
        GPUArray<Scalar> m_partial_Ksum_r;  //!< Rotational kinetic energy per body
        GPUArray<Scalar> m_Ksum_t;          //!< Translational kinetic energy
        GPUArray<Scalar> m_Ksum_r;          //!< Rotational kinetic energy
        GPUArray<Scalar4> m_new_box;        //!< New box size
        GPUArray<Scalar> m_partial_sum_virial_rigid;  //!< Partial sum for first pass reduction
        GPUArray<Scalar> m_sum_virial_rigid;  //!< Total sum of virial on the GPU

        unsigned int m_block_size;        //!< Block size to launch on the GPU (must be a power of two)
        unsigned int m_group_num_blocks;  //!< Number of blocks of \a block_size to launch when updating the group
        unsigned int m_full_num_blocks;   //!< Number of blocks to launch when updating all particles
        GPUArray<Scalar> m_partial_sum2K;  //!< Partial sums from the first pass reduction
        GPUArray<Scalar> m_sum2K;          //!< Total sum of 2K on the GPU
        GPUArray<Scalar> m_partial_sumW;   //!< Partial sums for the first pass reduction of W
        GPUArray<Scalar> m_sumW;           //!< Total sum of W on the GPU

    };

//! Exports the TwoStepNPHRigidGPU class to python
void export_TwoStepNPHRigidGPU();

#endif // #ifndef __TWO_STEP_NPH_RIGID_GPU_H__
