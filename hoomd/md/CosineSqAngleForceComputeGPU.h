// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CosineSqAngleForceCompute.h"
#include "CosineSqAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file CosineSqAngleForceComputeGPU.h
    \brief Declares the CosineSqAngleForceGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __COSINESQANGLEFORCECOMPUTEGPU_H__
#define __COSINESQANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the cosine squared angle force calculation on the GPU
/*! CosineSqAngleForceComputeGPU implements the same calculations as CosineSqAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in angleforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT CosineSqAngleForceComputeGPU : public CosineSqAngleForceCompute
    {
    public:
    //! Constructs the compute
    CosineSqAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~CosineSqAngleForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar2> m_params;            //!< Parameters stored on the GPU

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
