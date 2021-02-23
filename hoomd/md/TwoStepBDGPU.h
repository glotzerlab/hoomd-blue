// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepBD.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

//! Implements Brownian dynamics on the GPU
/*! GPU accelerated version of TwoStepBD

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepBDGPU : public TwoStepBD
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepBDGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<Variant> T,
                     unsigned int seed);

        virtual ~TwoStepBDGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        unsigned int m_block_size;               //!< block size
    };

//! Exports the TwoStepBDGPU class to python
void export_TwoStepBDGPU(pybind11::module& m);
