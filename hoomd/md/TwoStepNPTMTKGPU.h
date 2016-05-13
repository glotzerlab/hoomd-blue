// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "TwoStepNPTMTK.h"
#include "hoomd/Variant.h"
#include "hoomd/ComputeThermo.h"

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
                   boost::shared_ptr<ComputeThermo> thermo_group_t,
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
