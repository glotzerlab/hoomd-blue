// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoHMA.h"

/*! \file ComputeThermoHMAGPU.h
    \brief Declares a class for computing HMA thermodynamic quantities on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_HMA_GPU_H__
#define __COMPUTE_THERMO_HMA_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes HMA thermodynamic properties of a group of particles on the GPU
/*! ComputeThermoHMAGPU is a GPU accelerated implementation of ComputeThermoHMA
    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoHMAGPU : public ComputeThermoHMA
    {
    public:
    //! Constructs the compute
    ComputeThermoHMAGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        const double temperature,
                        const double harmonicPressure);
    virtual ~ComputeThermoHMAGPU();

    protected:
    GlobalVector<Scalar3> m_scratch; //!< Scratch space for partial sums
    unsigned int m_block_size;       //!< Block size executed
    hipEvent_t m_event;              //!< CUDA event for synchronization

    //! Does the actual computation
    virtual void computeProperties();
    };

    } // end namespace md
    } // end namespace hoomd

#endif
