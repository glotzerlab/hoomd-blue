// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SorterGPU.h
 * \brief Declares mpcd::SorterGPU, which sorts particles in the cell list on the GPU
 */

#ifndef MPCD_SORTER_GPU_H_
#define MPCD_SORTER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Sorter.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

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
        SorterGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                  unsigned int cur_timestep,
                  unsigned int period);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::Sorter::setAutotunerParams(enable, period);

            m_sentinel_tuner->setEnabled(enable); m_sentinel_tuner->setPeriod(period);
            m_reverse_tuner->setEnabled(enable); m_reverse_tuner->setPeriod(period);
            m_apply_tuner->setEnabled(enable); m_apply_tuner->setPeriod(period);
            }

    protected:
        std::unique_ptr<Autotuner> m_sentinel_tuner;    //!< Kernel tuner for filling sentinels in cell list
        std::unique_ptr<Autotuner> m_reverse_tuner;     //!< Kernel tuner for setting reverse map
        std::unique_ptr<Autotuner> m_apply_tuner;       //!< Kernel tuner for applying sorted order

        //! Compute the sorting order at the current timestep on the GPU
        virtual void computeOrder(unsigned int timestep);

        //! Apply the sorting order on the GPU
        virtual void applyOrder() const;
    };

namespace detail
{
//! Exports the mpcd::SorterGPU to python
void export_SorterGPU(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_SORTER_GPU_H_
