// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SlitPoreGeometryFillerGPU.h
 * \brief Definition of virtual particle filler for mpcd::detail::SlitPoreGeometry on the GPU.
 */

#ifndef MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_H_
#define MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "SlitPoreGeometryFiller.h"
#include "hoomd/Autotuner.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data for SlitPoreGeometry using the GPU
class PYBIND11_EXPORT SlitPoreGeometryFillerGPU : public mpcd::SlitPoreGeometryFiller
    {
    public:
    //! Constructor
    SlitPoreGeometryFillerGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                              Scalar density,
                              unsigned int type,
                              std::shared_ptr<Variant> T,
                              uint16_t seed,
                              std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom);

    //! Set autotuner parameters
    /*!
     * \param enable Enable/disable autotuning
     * \param period period (approximate) in time steps when returning occurs
     */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        mpcd::SlitPoreGeometryFiller::setAutotunerParams(enable, period);

        m_tuner->setEnabled(enable);
        m_tuner->setPeriod(period);
        }

    protected:
    //! Draw particles within the fill volume on the GPU
    virtual void drawParticles(uint64_t timestep);

    private:
    std::unique_ptr<hoomd::Autotuner> m_tuner; //!< Autotuner for drawing particles
    };

namespace detail
    {
//! Export SlitPoreGeometryFillerGPU to python
void export_SlitPoreGeometryFillerGPU(pybind11::module& m);
    }  // end namespace detail
    }  // end namespace mpcd
    }  // end namespace hoomd
#endif // MPCD_SLIT_PORE_GEOMETRY_FILLER_GPU_H_
