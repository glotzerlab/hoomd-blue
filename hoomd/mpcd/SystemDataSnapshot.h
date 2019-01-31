// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SystemDataSnapshot.h
 * \brief Declares the mpcd::SystemDataSnapshot class
 */

#ifndef MPCD_SYSTEM_DATA_SNAPSHOT_H_
#define MPCD_SYSTEM_DATA_SNAPSHOT_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleDataSnapshot.h"
#include "hoomd/ParticleData.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Structure for initializing system data
class PYBIND11_EXPORT SystemDataSnapshot
    {
    public:
        //! Constructor
        SystemDataSnapshot(std::shared_ptr<::SystemDefinition> sysdef)
            : m_sysdef(sysdef), m_hoomd_pdata(m_sysdef->getParticleData()),
              m_global_box(m_hoomd_pdata->getGlobalBox())
            {
            particles = std::make_shared<mpcd::ParticleDataSnapshot>();
            }

        //! Replicate the snapshot
        void replicate(unsigned int nx, unsigned int ny, unsigned int nz);

        //! Get the system definition
        std::shared_ptr<::SystemDefinition> getSystemDefinition() const
            {
            return m_sysdef;
            }

        //! Get the global box
        const BoxDim& getGlobalBox() const
            {
            return m_global_box;
            }

        //! Get the dimensions of the snapshot
        unsigned int getDimensions() const
            {
            return m_sysdef->getNDimensions();
            }

        //! Get the execution configuration
        std::shared_ptr<const ::ExecutionConfiguration> getExecutionConfiguration() const
            {
            return m_hoomd_pdata->getExecConf();
            }

        //! Get the domain decomposition
        std::shared_ptr<::DomainDecomposition> getDomainDecomposition() const
            {
            #ifdef ENABLE_MPI
            return m_hoomd_pdata->getDomainDecomposition();
            #else
            return std::shared_ptr<::DomainDecomposition>();
            #endif // ENABLE_MPI
            }

        std::shared_ptr<mpcd::ParticleDataSnapshot> particles;   //!< MPCD particle data snapshot

    private:
        std::shared_ptr<::SystemDefinition> m_sysdef;   //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_hoomd_pdata;  //!< Standard HOOMD particle data
        BoxDim m_global_box;                            //!< Global simulation box when snapshot is created
    };

namespace detail
{
//! Export mpcd::SystemDataSnapshot to python
void export_SystemDataSnapshot(pybind11::module& m);
} // // end namespace detail

} // end namespace mpcd

#endif // MPCD_SYSTEM_DATA_SNAPSHOT_H_
