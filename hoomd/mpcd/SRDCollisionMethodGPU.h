// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SRDCollisionMethodGPU.h
 * \brief Declaration of mpcd::SRDCollisionMethodGPU
 */

#ifndef MPCD_SRD_COLLISION_METHOD_GPU_H_
#define MPCD_SRD_COLLISION_METHOD_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SRDCollisionMethod.h"
#include "hoomd/Autotuner.h"

namespace mpcd
{

class PYBIND11_EXPORT SRDCollisionMethodGPU : public mpcd::SRDCollisionMethod
    {
    public:
        //! Constructor
        SRDCollisionMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                              unsigned int cur_timestep,
                              unsigned int period,
                              int phase,
                              unsigned int seed,
                              std::shared_ptr<mpcd::CellThermoCompute> thermo);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::SRDCollisionMethod::setAutotunerParams(enable, period);

            m_tuner_rotvec->setPeriod(period); m_tuner_rotvec->setEnabled(enable);
            m_tuner_rotate->setPeriod(period); m_tuner_rotate->setEnabled(enable);
            }

    protected:
        //! Randomly draw cell rotation vectors
        virtual void drawRotationVectors(unsigned int timestep);

        //! Apply rotation matrix to velocities
        virtual void rotate(unsigned int timestep);

    private:
        std::unique_ptr<Autotuner> m_tuner_rotvec;  //!< Tuner for drawing rotation vectors
        std::unique_ptr<Autotuner> m_tuner_rotate;  //!< Tuner for rotating velocities
    };

namespace detail
{
//! Export SRDCollisionMethodGPU to python
void export_SRDCollisionMethodGPU(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_SRD_COLLISION_METHOD_GPU_H_
