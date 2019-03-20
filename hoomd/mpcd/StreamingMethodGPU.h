// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingMethodGPU.h
 * \brief Declaration of mpcd::StreamingMethodGPU
 */

#ifndef MPCD_STREAMING_METHOD_GPU_H_
#define MPCD_STREAMING_METHOD_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "StreamingMethod.h"
#include "hoomd/Autotuner.h"

namespace mpcd
{

//! MPCD streaming method on the GPU
class PYBIND11_EXPORT StreamingMethodGPU : public mpcd::StreamingMethod
    {
    public:
        //! Constructor
        StreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                           unsigned int cur_timestep,
                           unsigned int period,
                           int phase);

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         *
         * Derived classes should override this to set the parameters of their autotuners.
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            StreamingMethod::setAutotunerParams(enable, period);
            m_tuner->setEnabled(enable); m_tuner->setPeriod(period);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner;
    };

namespace detail
{
//! Export StreamingMethodGPU to python
void export_StreamingMethodGPU(pybind11::module& m);
}; // end namespace detail

} // end namespace mpcd

#endif // MPCD_STREAMING_METHOD_GPU_H_
