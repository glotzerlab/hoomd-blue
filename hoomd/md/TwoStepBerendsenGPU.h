// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "TwoStepBerendsen.h"

#ifndef __BERENDSEN_GPU_H__
#define __BERENDSEN_GPU_H__

/*! \file TwoStepBerendsenGPU.h
    \brief Declaration of the Berendsen thermostat on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/*! Implements the Berendsen thermostat on the GPU
 */
class PYBIND11_EXPORT TwoStepBerendsenGPU : public TwoStepBerendsen
    {
    public:
    //! Constructor
    TwoStepBerendsenGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        std::shared_ptr<ComputeThermo> thermo,
                        Scalar tau,
                        std::shared_ptr<Variant> T);
    virtual ~TwoStepBerendsenGPU() {};

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    unsigned int m_block_size; //!< Block size to launch on the GPU
    };

namespace detail
    {
//! Export the Berendsen GPU class to python
void export_BerendsenGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __BERENDSEN_GPU_H__
