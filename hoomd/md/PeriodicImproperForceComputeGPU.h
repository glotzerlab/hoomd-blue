// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproperForceCompute.h"
#include "PeriodicImproperForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

#pragma once

namespace hoomd
    {
namespace md
    {
//! Implements the periodic improper force calculation on the GPU
/*! PeriodicImproperForceComputeGPU implements the same calculations as
   PeriodicImproperForceCompute, but executing on the GPU.
*/
class PYBIND11_EXPORT PeriodicImproperForceComputeGPU : public PeriodicImproperForceCompute
    {
    public:
    //! Constructs the compute
    PeriodicImproperForceComputeGPU(std::shared_ptr<SystemDefinition> system);
    //! Destructor
    ~PeriodicImproperForceComputeGPU();

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd
