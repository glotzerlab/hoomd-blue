// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFillerGPU.h
 * \brief Definition of virtual particle filler for mpcd::ParallelPlateGeometry on the GPU.
 */

#ifndef MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_GPU_H_
#define MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ParallelPlateGeometryFiller.h"
#include "hoomd/Autotuner.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data for ParallelPlateGeometry using the GPU
class PYBIND11_EXPORT ParallelPlateGeometryFillerGPU : public mpcd::ParallelPlateGeometryFiller
    {
    public:
    //! Constructor
    ParallelPlateGeometryFillerGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   const std::string& type,
                                   Scalar density,
                                   std::shared_ptr<Variant> T,
                                   std::shared_ptr<const mpcd::ParallelPlateGeometry> geom);

    protected:
    //! Draw particles within the fill volume on the GPU
    virtual void drawParticles(uint64_t timestep);

    private:
    std::shared_ptr<hoomd::Autotuner<1>> m_tuner; //!< Autotuner for drawing particles
    };

namespace detail
    {
//! Export ParallelPlateGeometryFillerGPU to python
void export_ParallelPlateGeometryFillerGPU(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_GPU_H_
