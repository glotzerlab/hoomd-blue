// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "FIREEnergyMinimizer.h"

#include <memory>

#ifndef __FIRE_ENERGY_MINIMIZER_GPU_H__
#define __FIRE_ENERGY_MINIMIZER_GPU_H__

/*! \file FIREEnergyMinimizer.h
    \brief Declares a base class for all energy minimization methods
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Finds the nearest basin in the potential energy landscape
/*! \b Overview

    \ingroup updaters
*/
class PYBIND11_EXPORT FIREEnergyMinimizerGPU : public FIREEnergyMinimizer
    {
    public:
    //! Constructs the minimizer and associates it with the system
    FIREEnergyMinimizerGPU(std::shared_ptr<SystemDefinition>, Scalar);

    //! Destroys the minimizer
    virtual ~FIREEnergyMinimizerGPU() { }

    //! Iterates forward one step
    virtual void update(uint64_t timestep);

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory

    GPUVector<Scalar> m_partial_sum1; //!< memory space for partial sum over P and E
    GPUVector<Scalar> m_partial_sum2; //!< memory space for partial sum over vsq
    GPUVector<Scalar> m_partial_sum3; //!< memory space for partial sum over asq
    GPUArray<Scalar> m_sum;           //!< memory space for sum over E
    GPUArray<Scalar> m_sum3;          //!< memory space for the sum over P, vsq, asq

    private:
    //! allocate the memory needed to store partial sums
    void resizePartialSumArrays();
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __FIRE_ENERGY_MINIMIZER_GPU_H__
