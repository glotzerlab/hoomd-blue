// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TableAngleForceCompute.h"
#include "TableAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file TableAngleForceComputeGPU.h
    \brief Declares the TableAngleForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __TABLEANGLEFORCECOMPUTEGPU_H__
#define __TABLEANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Compute table based bond potentials on the GPU
/*! Calculates exactly the same thing as TableAngleForceCompute, but on the GPU

    The GPU kernel for calculating this can be found in TableAngleForceComputeGPU.cu/
    \ingroup computes
*/
class PYBIND11_EXPORT TableAngleForceComputeGPU : public TableAngleForceCompute
    {
    public:
    //! Constructs the compute
    TableAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, unsigned int table_width);

    //! Destructor
    virtual ~TableAngleForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;        //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
