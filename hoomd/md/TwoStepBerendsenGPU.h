// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepBerendsen.h"

#ifndef __BERENDSEN_GPU_H__
#define __BERENDSEN_GPU_H__

/*! \file TwoStepBerendsenGPU.h
    \brief Declaration of the Berendsen thermostat on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

/*! Implements the Berendsen thermostat on the GPU
*/
class PYBIND11_EXPORT TwoStepBerendsenGPU : public TwoStepBerendsen
    {
    public:
        //! Constructor
        TwoStepBerendsenGPU(std::shared_ptr< SystemDefinition > sysdef,
                            std::shared_ptr< ParticleGroup > group,
                            std::shared_ptr< ComputeThermo > thermo,
                            Scalar tau,
                            std::shared_ptr< Variant > T);
        virtual ~TwoStepBerendsenGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        unsigned int m_block_size; //!< Block size to launch on the GPU
    };

//! Export the Berendsen GPU class to python
void export_BerendsenGPU(pybind11::module& m);

#endif // __BERENDSEN_GPU_H__
