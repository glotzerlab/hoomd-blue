// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/GPUArray.h"
#include "hoomd/md/NeighborList.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#include "DEM3DForceCompute.h"

/*! \file DEM3DForceComputeGPU.h
  \brief Declares the class DEM3DForceComputeGPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM3DFORCECOMPUTEGPU_H__
#define __DEM3DFORCECOMPUTEGPU_H__

#ifdef ENABLE_CUDA

//! Computes DEM3D forces on each particle using the GPU
/*! Calculates the same forces as DEM3DForceCompute, but on the GPU.

  The GPU kernel for calculating the forces is in DEM3DForceGPU.cu.
  \ingroup computes
*/
template<typename Real, typename Real4, typename Potential>
class DEM3DForceComputeGPU: public DEM3DForceCompute<Real, Real4, Potential>
    {
    public:
        //! Constructs the compute
        DEM3DForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
            Real r_cut, Potential potential);

        //! Destructor
        virtual ~DEM3DForceComputeGPU();

        //! Set parameters for the builtin autotuner
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Find the maximum number of GPU threads (2*vertices + edges) among all shapes
        size_t maxGPUThreads() const;

    protected:
        std::unique_ptr<Autotuner> m_tuner;     //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

#include "DEM3DForceComputeGPU.cc"

#endif

#endif
