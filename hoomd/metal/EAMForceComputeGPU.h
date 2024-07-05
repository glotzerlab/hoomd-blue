// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EAMForceCompute.h"
#include "EAMForceGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/md/NeighborList.h"

#include <memory>

/*! \file EAMForceComputeGPU.h
 \brief Declares the class EAMForceComputeGPU
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __EAMForceComputeGPU_H__
#define __EAMForceComputeGPU_H__

namespace hoomd
    {
namespace metal
    {
//! Computes EAM forces on each particle using the GPU
/*! Calculates the same forces as EAMForceCompute, but on the GPU by using texture
 * memory(CUDAArray).
 */
class EAMForceComputeGPU : public EAMForceCompute
    {
    public:
    //! Constructs the compute
    EAMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, char* filename, int type_of_file);

    //! Destructor
    virtual ~EAMForceComputeGPU();

    protected:
    GlobalArray<kernel::EAMTexInterData> m_eam_data; //!< EAM parameters to be communicated
    std::shared_ptr<Autotuner<1>> m_tuner;           //!< autotuner for block size
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the EAMForceComputeGPU class to python
void export_EAMForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace metal
    } // end namespace hoomd

#endif
