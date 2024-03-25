// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "OPLSDihedralForceCompute.h"
#include "OPLSDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file OPLSDihedralForceComputeGPU.h
    \brief Declares the OPLSDihedralForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __OPLSDIHEDRALFORCECOMPUTEGPU_H__
#define __OPLSDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes OPLS-style dihedral potentials on the GPU
/*! Calculates the OPLS type dihedral force on the GPU

    The GPU kernel for calculating this can be found in OPLSDihedralForceComputeGPU.cu
    \ingroup computes
*/
class PYBIND11_EXPORT OPLSDihedralForceComputeGPU : public OPLSDihedralForceCompute
    {
    public:
    //! Constructs the compute
    OPLSDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~OPLSDihedralForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
