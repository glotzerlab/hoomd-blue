// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ComputeThermo.h"

/*! \file ComputeThermoGPU.h
    \brief Declares a class for computing thermodynamic quantities on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_GPU_H__
#define __COMPUTE_THERMO_GPU_H__

//! Computes thermodynamic properties of a group of particles on the GPU
/*! ComputeThermoGPU is a GPU accelerated implementation of ComputeThermo
    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoGPU : public ComputeThermo
    {
    public:
        //! Constructs the compute
        ComputeThermoGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         const std::string& suffix = std::string(""));
        virtual ~ComputeThermoGPU();

    protected:
        GlobalVector<Scalar4> m_scratch;  //!< Scratch space for partial sums
        GlobalVector<Scalar> m_scratch_pressure_tensor; //!< Scratch space for pressure tensor partial sums
        GlobalVector<Scalar> m_scratch_rot; //!< Scratch space for rotational kinetic energy partial sums
        unsigned int m_block_size;   //!< Block size executed
        cudaEvent_t m_event;         //!< CUDA event for synchronization

#ifdef ENABLE_MPI
        //! Reduce properties over MPI
        virtual void reduceProperties();
#endif

        //! Does the actual computation
        virtual void computeProperties();
    };

//! Exports the ComputeThermoGPU class to python
void export_ComputeThermoGPU(pybind11::module& m);

#endif
