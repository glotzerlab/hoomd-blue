// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PCNDAngleForceCompute.h"
#include "PCNDAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file PCNDAngleForceComputeGPU.h
    \brief Declares the PCNDAngleForceGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __PCNDANGLEFORCECOMPUTEGPU_H__
#define __PCNDANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the PCND harmonic angle force calculation on the GPU
/*! PCNDAngleForceComputeGPU implements the same calculations as PCNDAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in angleforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT PCNDAngleForceComputeGPU : public PCNDAngleForceCompute
    {
    public:
        //! Constructs the compute
        PCNDAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
        //! Destructor
        ~PCNDAngleForceComputeGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PCNDAngleForceCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar Xi, Scalar Tau, unsigned int PCND_type, uint16_t particle_sum, Scalar particle_index);

        protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<Scalar2> m_params;           //!< k, t0 Parameters stored on the GPU

        // below are just for the PCND angle potential
        GPUArray<Scalar2>  m_PCNDsr;    //!< GPU copy of the angle's epsilon/sigma/rcut (esr)
        GPUArray<Scalar4>  m_PCNDepow;  //!< GPU copy of the angle's powers (pow1,pow2) and prefactor

        //! Actually compute the forces
        virtual void computeForces(uint64_t timestep);
        };

namespace detail
    {
//! Export the PCNDAngleForceComputeGPU class to python
void export_PCNDAngleForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
