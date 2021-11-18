// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ConfinedStreamingMethodGPU.h
 * \brief Declaration of mpcd::ConfinedStreamingMethodGPU
 */

#ifndef MPCD_CONFINED_STREAMING_METHOD_GPU_H_
#define MPCD_CONFINED_STREAMING_METHOD_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ConfinedStreamingMethod.h"
#include "ConfinedStreamingMethodGPU.cuh"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace mpcd
    {
//! MPCD confined geometry streaming method
/*!
 * This method implements the GPU version of ballistic propagation of MPCD
 * particles in a confined geometry.
 */
template<class Geometry>
class PYBIND11_EXPORT ConfinedStreamingMethodGPU : public mpcd::ConfinedStreamingMethod<Geometry>
    {
    public:
    //! Constructor
    /*!
     * \param sysdata MPCD system data
     * \param cur_timestep Current system timestep
     * \param period Number of timesteps between collisions
     * \param phase Phase shift for periodic updates
     * \param geom Streaming geometry
     */
    ConfinedStreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                               unsigned int cur_timestep,
                               unsigned int period,
                               int phase,
                               std::shared_ptr<const Geometry> geom)
        : mpcd::ConfinedStreamingMethod<Geometry>(sysdata, cur_timestep, period, phase, geom)
        {
        m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_stream", this->m_exec_conf));
        }

    //! Implementation of the streaming rule
    virtual void stream(uint64_t timestep);

    //! Set autotuner parameters
    /*!
     * \param enable Enable/disable autotuning
     * \param period period (approximate) in time steps when returning occurs
     *
     * Derived classes should override this to set the parameters of their autotuners.
     */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        ConfinedStreamingMethod<Geometry>::setAutotunerParams(enable, period);
        m_tuner->setEnabled(enable);
        m_tuner->setPeriod(period);
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner;
    };

/*!
 * \param timestep Current time to stream
 */
template<class Geometry> void ConfinedStreamingMethodGPU<Geometry>::stream(uint64_t timestep)
    {
    if (!this->shouldStream(timestep))
        return;

    // the validation step currently proceeds on the cpu because it is done infrequently.
    // if it becomes a performance concern, it can be ported to the gpu
    if (this->m_validate_geom)
        {
        this->validate();
        this->m_validate_geom = false;
        }

    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "MPCD stream");
    ArrayHandle<Scalar4> d_pos(this->m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    mpcd::gpu::stream_args_t args(d_pos.data,
                                  d_vel.data,
                                  this->m_mpcd_pdata->getMass(),
                                  (this->m_field) ? this->m_field->get(access_location::device)
                                                  : nullptr,
                                  this->m_mpcd_sys->getCellList()->getCoverageBox(),
                                  this->m_mpcd_dt,
                                  this->m_mpcd_pdata->getN(),
                                  m_tuner->getParam());

    m_tuner->begin();
    mpcd::gpu::confined_stream<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    // particles have moved, so the cell cache is no longer valid
    this->m_mpcd_pdata->invalidateCellCache();
    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

namespace detail
    {
//! Export mpcd::StreamingMethodGPU to python
/*!
 * \param m Python module to export to
 */
template<class Geometry> void export_ConfinedStreamingMethodGPU(pybind11::module& m)
    {
    const std::string name = "ConfinedStreamingMethodGPU" + Geometry::getName();
    pybind11::class_<mpcd::ConfinedStreamingMethodGPU<Geometry>,
                     mpcd::ConfinedStreamingMethod<Geometry>,
                     std::shared_ptr<mpcd::ConfinedStreamingMethodGPU<Geometry>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<mpcd::SystemData>,
                            unsigned int,
                            unsigned int,
                            int,
                            std::shared_ptr<const Geometry>>());
    }
    }  // end namespace detail
    }  // end namespace mpcd
    }  // end namespace hoomd
#endif // MPCD_CONFINED_STREAMING_METHOD_GPU_H_
