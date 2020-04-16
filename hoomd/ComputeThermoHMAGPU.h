// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ajs42

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

//! Computes HMA thermodynamic properties of a group of particles on the GPU
/*! ComputeThermoHMAGPU is a GPU accelerated implementation of ComputeThermoHMA
    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoHMAGPU : public ComputeThermoHMA
    {
    public:
        //! Constructs the compute
        ComputeThermoHMAGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group, const double temperature,
                         const double harmonicPressure, const std::string& suffix = std::string(""));
        virtual ~ComputeThermoHMAGPU();

    protected:
        GlobalVector<Scalar3> m_scratch;  //!< Scratch space for partial sums
        unsigned int m_block_size;   //!< Block size executed
        cudaEvent_t m_event;         //!< CUDA event for synchronization

        //! Does the actual computation
        virtual void computeProperties();
    };

//! Exports the ComputeThermoHMAGPU class to python
void export_ComputeThermoHMAGPU(pybind11::module& m);

#endif
