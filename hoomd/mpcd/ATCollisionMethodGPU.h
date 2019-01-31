// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethodGPU.h
 * \brief Declaration of mpcd::ATCollisionMethodGPU
 */

#ifndef MPCD_AT_COLLISION_METHOD_GPU_H_
#define MPCD_AT_COLLISION_METHOD_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ATCollisionMethod.h"
#include "hoomd/Autotuner.h"

namespace mpcd
{

class PYBIND11_EXPORT ATCollisionMethodGPU : public mpcd::ATCollisionMethod
    {
    public:
        //! Constructor
        ATCollisionMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                             unsigned int cur_timestep,
                             unsigned int period,
                             int phase,
                             unsigned int seed,
                             std::shared_ptr<mpcd::CellThermoCompute> thermo,
                             std::shared_ptr<mpcd::CellThermoCompute> rand_thermo,
                             std::shared_ptr<::Variant> T);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::ATCollisionMethod::setAutotunerParams(enable, period);

            m_tuner_draw->setPeriod(period); m_tuner_draw->setEnabled(enable);
            m_tuner_apply->setPeriod(period); m_tuner_apply->setEnabled(enable);
            }

    protected:
        //! Draw velocities for particles in each cell on the GPU
        virtual void drawVelocities(unsigned int timestep);

        //! Apply the random velocities to particles in each cell on the GPU
        virtual void applyVelocities();

    private:
        std::unique_ptr<Autotuner> m_tuner_draw;    //!< Tuner for drawing random velocities
        std::unique_ptr<Autotuner> m_tuner_apply;   //!< Tuner for applying random velocities
    };

namespace detail
{
//! Export ATCollisionMethodGPU to python
void export_ATCollisionMethodGPU(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_AT_COLLISION_METHOD_GPU_H_
