// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard
#include "HarmonicAngleForceCompute.h"
#include "HarmonicAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

/*! \file HarmonicAngleForceComputeGPU.h
    \brief Declares the HarmonicAngleForceGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __HARMONICANGLEFORCECOMPUTEGPU_H__
#define __HARMONICANGLEFORCECOMPUTEGPU_H__

//! Implements the harmonic angle force calculation on the GPU
/*! HarmonicAngleForceComputeGPU implements the same calculations as HarmonicAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in angleforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT HarmonicAngleForceComputeGPU : public HarmonicAngleForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
        //! Destructor
        ~HarmonicAngleForceComputeGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            HarmonicAngleForceCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

    protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<Scalar2>  m_params;          //!< Parameters stored on the GPU

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the AngleForceComputeGPU class to python
void export_HarmonicAngleForceComputeGPU(pybind11::module& m);

#endif
