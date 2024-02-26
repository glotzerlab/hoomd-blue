// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BondTablePotential.h"
#include "BondTablePotentialGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file BondTablePotentialGPU.h
    \brief Declares the BondTablePotentialGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __BONDTABLEPOTENTIALGPU_H__
#define __BONDTABLEPOTENTIALGPU_H__

namespace hoomd
    {
namespace md
    {
//! Compute table based bond potentials on the GPU
/*! Calculates exactly the same thing as BondTablePotential, but on the GPU

    The GPU kernel for calculating this can be found in BondTablePotentialGPU.cu/
    \ingroup computes
*/
class PYBIND11_EXPORT BondTablePotentialGPU : public BondTablePotential
    {
    public:
    //! Constructs the compute
    BondTablePotentialGPU(std::shared_ptr<SystemDefinition> sysdef, unsigned int table_width);

    //! Destructor
    virtual ~BondTablePotentialGPU();

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;        //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
