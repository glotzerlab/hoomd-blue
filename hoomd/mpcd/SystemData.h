// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SystemData.h
 * \brief Declares the mpcd::SystemData class
 */

#ifndef MPCD_SYSTEM_DATA_H_
#define MPCD_SYSTEM_DATA_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"
#include "ParticleData.h"
#include "SystemDataSnapshot.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

class PYBIND11_EXPORT SystemData
    {
    public:
        //! Construct from MPCD ParticleData
        SystemData(std::shared_ptr<::SystemDefinition> sysdef,
                   std::shared_ptr<mpcd::ParticleData> mpcd_pdata);

        //! Construct from a snapshot
        SystemData(std::shared_ptr<mpcd::SystemDataSnapshot> snapshot);

        //! Destructor
        ~SystemData();

        //! Get the MPCD particle data
        std::shared_ptr<mpcd::ParticleData> getParticleData() const
            {
            return m_particles;
            }

        //! Get the MPCD cell list
        std::shared_ptr<mpcd::CellList> getCellList() const
            {
            return m_cl;
            }

        //! Get the HOOMD system definition
        std::shared_ptr<::SystemDefinition> getSystemDefinition() const
            {
            return m_sysdef;
            }

        //! Get the current global simulation box
        const BoxDim& getGlobalBox() const
            {
            return m_global_box;
            }

        //! Return a snapshot of the current system data
        std::shared_ptr<mpcd::SystemDataSnapshot> takeSnapshot(bool particles);

        //! Re-initialize the system from a snapshot
        void initializeFromSnapshot(std::shared_ptr<mpcd::SystemDataSnapshot> snapshot);

        //! Sets the profiler for the particle data to use
        /*
         * \param prof System profiler to use, nullptr if profiling is disabled
         */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_particles->setProfiler(prof);
            m_cl->setProfiler(prof);
            }

        //! Set autotuner parameters
        /*!
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_particles->setAutotunerParams(enable, period);
            }

    private:
        std::shared_ptr<::SystemDefinition> m_sysdef;       //!< HOOMD system definition
        std::shared_ptr<mpcd::ParticleData> m_particles;    //!< MPCD particle data
        std::shared_ptr<mpcd::CellList> m_cl;               //!< MPCD cell list
        const BoxDim m_global_box;  //!< Global simulation box

        //! Check that the simulation box has not changed from the cached value on initialization
        void checkBox() const
            {
            const BoxDim& new_box = m_sysdef->getParticleData()->getGlobalBox();

            const Scalar3 cur_L = m_global_box.getL();
            const Scalar3 new_L = new_box.getL();

            const Scalar tol = 1.e-6;
            if (std::fabs(new_L.x-cur_L.x) > tol ||
                std::fabs(new_L.y-cur_L.y) > tol ||
                std::fabs(new_L.z-cur_L.z) > tol ||
                std::fabs(new_box.getTiltFactorXY() - m_global_box.getTiltFactorXY()) > tol ||
                std::fabs(new_box.getTiltFactorXZ() - m_global_box.getTiltFactorXZ()) > tol ||
                std::fabs(new_box.getTiltFactorYZ() - m_global_box.getTiltFactorYZ()) > tol)
                {
                m_sysdef->getParticleData()->getExecConf()->msg->error() << "mpcd: changing simulation box not supported" << std::endl;
                throw std::runtime_error("Changing global simulation box not supported");
                }
            }
    };

namespace detail
{
//! Exports mpcd::SystemData to python
void export_SystemData(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_SYSTEM_DATA_H_
