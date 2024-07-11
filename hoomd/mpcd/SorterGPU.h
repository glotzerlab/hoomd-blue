// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SorterGPU.h
 * \brief Declares mpcd::SorterGPU, which sorts particles in the cell list on the GPU
 */

#ifndef MPCD_SORTER_GPU_H_
#define MPCD_SORTER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "Sorter.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

namespace hoomd
    {
namespace mpcd
    {
//! Sorts MPCD particles on the GPU
/*!
 * See mpcd::Sorter for design details.
 */
class PYBIND11_EXPORT SorterGPU : public mpcd::Sorter
    {
    public:
    //! Constructor
    SorterGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);

    protected:
    /// Kernel tuner for filling sentinels in cell list.
    std::shared_ptr<Autotuner<1>> m_sentinel_tuner;

    /// Kernel tuner for setting reverse map.
    std::shared_ptr<Autotuner<1>> m_reverse_tuner;

    //!< Kernel tuner for applying sorted order.
    std::shared_ptr<Autotuner<1>> m_apply_tuner;

    //! Compute the sorting order at the current timestep on the GPU
    virtual void computeOrder(uint64_t timestep);

    //! Apply the sorting order on the GPU
    virtual void applyOrder() const;
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SORTER_GPU_H_
