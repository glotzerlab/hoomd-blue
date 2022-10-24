// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
#include "TwoStepNVTBaseGPU.h"
namespace hoomd
    {
namespace md
    {
/*! Implements the Berendsen thermostat on the GPU
 */
class PYBIND11_EXPORT TwoStepBerendsenGPU : public TwoStepBerendsen, public TwoStepNVTBaseGPU
    {
    public:
    //! Constructor
    TwoStepBerendsenGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        std::shared_ptr<ComputeThermo> thermo,
                        Scalar tau,
                        std::shared_ptr<Variant> T);
    virtual ~TwoStepBerendsenGPU() {};
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BERENDSEN_GPU_H__
