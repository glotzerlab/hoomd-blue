// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard
#include "CGCMMAngleForceCompute.h"
#include "CGCMMAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file HarmonicAngleForceComputeGPU.h
    \brief Declares the HarmonicAngleForceGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CGCMMANGLEFORCECOMPUTEGPU_H__
#define __CGCMMANGLEFORCECOMPUTEGPU_H__

//! Implements the CGCMM harmonic angle force calculation on the GPU
/*! CGCMMAngleForceComputeGPU implements the same calculations as CGCMMAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in angleforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT CGCMMAngleForceComputeGPU : public CGCMMAngleForceCompute
    {
    public:
        //! Constructs the compute
        CGCMMAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
        //! Destructor
        ~CGCMMAngleForceComputeGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            CGCMMAngleForceCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma);

    protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<Scalar2> m_params;           //!< k, t0 Parameters stored on the GPU

        // below are just for the CG-CMM angle potential
        GPUArray<Scalar2>  m_CGCMMsr;    //!< GPU copy of the angle's epsilon/sigma/rcut (esr)
        GPUArray<Scalar4>  m_CGCMMepow;  //!< GPU copy of the angle's powers (pow1,pow2) and prefactor

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the CGCMMAngleForceComputeGPU class to python
void export_CGCMMAngleForceComputeGPU(pybind11::module& m);

#endif
