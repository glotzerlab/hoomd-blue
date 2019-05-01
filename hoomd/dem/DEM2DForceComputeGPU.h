// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/Autotuner.h"
#include "hoomd/GPUArray.h"
#include "hoomd/md/NeighborList.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#include "DEM2DForceCompute.h"
#include "DEM2DForceGPU.cuh"

/*! \file DEM2DForceComputeGPU.h
  \brief Declares the class DEM2DForceComputeGPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM2DFORCECOMPUTEGPU_H__
#define __DEM2DFORCECOMPUTEGPU_H__

#ifdef ENABLE_CUDA

//! Computes DEM2D forces on each particle using the GPU
/*! Calculates the same forces as DEM2DForceCompute, but on the GPU.

  The GPU kernel for calculating the forces is in DEM2DForceGPU.cu.
  \ingroup computes
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
class DEM2DForceComputeGPU : public DEM2DForceCompute<Real, Real4, Potential>
    {
    public:
        //! Constructs the compute
        DEM2DForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
            Scalar r_cut, Potential potential);

        //! Destructor
        virtual ~DEM2DForceComputeGPU();

        //! Set the vertices for a particle type
        virtual void setParams(unsigned int type,
            const pybind11::list &vertices);

        //! Set parameters for the builtin autotuner
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        GPUArray<Real2> m_vertices;     //!< Vertices for all shapes
        GPUArray<unsigned int> m_num_shape_vertices;    //!< Number of vertices for each shape
        std::unique_ptr<Autotuner> m_tuner;     //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Re-send the list of vertices and links to the GPU
        void createGeometry();

        //! Find the total number of vertices in the current set of shapes
        size_t numVertices() const;

        //! Find the maximum number of vertices in the current set of shapes
        size_t maxVertices() const;
    };

#include "DEM2DForceComputeGPU.cc"

#endif

#endif
