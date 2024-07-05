// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HarmonicImproperForceCompute.h"
#include "HarmonicImproperForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file HarmonicImproperForceComputeGPU.h
    \brief Declares the HarmonicImproperForceGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __HARMONICIMPROPERFORCECOMPUTEGPU_H__
#define __HARMONICIMPROPERFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the harmonic improper force calculation on the GPU
/*! HarmonicImproperForceComputeGPU implements the same calculations as
   HarmonicImproperForceCompute, but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in improperforce_kernel.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT HarmonicImproperForceComputeGPU : public HarmonicImproperForceCompute
    {
    public:
    //! Constructs the compute
    HarmonicImproperForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~HarmonicImproperForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar chi);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar2> m_params;            //!< Parameters stored on the GPU (k,chi)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
