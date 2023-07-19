// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SystemDataSnapshot.h
 * \brief Declares the mpcd::SystemDataSnapshot class
 */

#ifndef MPCD_SYSTEM_DATA_SNAPSHOT_H_
#define MPCD_SYSTEM_DATA_SNAPSHOT_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ParticleDataSnapshot.h"
#include "hoomd/ParticleData.h"
#include "hoomd/SystemDefinition.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Structure for initializing system data
class PYBIND11_EXPORT SystemDataSnapshot
    {
    public:
    //! Constructor
    SystemDataSnapshot(std::shared_ptr<hoomd::SystemDefinition> sysdef)
        : m_sysdef(sysdef), m_hoomd_pdata(m_sysdef->getParticleData()),
          m_global_box(m_hoomd_pdata->getGlobalBox())
        {
        particles = std::make_shared<mpcd::ParticleDataSnapshot>();
        }

    //! Replicate the snapshot
    void replicate(unsigned int nx, unsigned int ny, unsigned int nz);

    //! Get the system definition
    std::shared_ptr<hoomd::SystemDefinition> getSystemDefinition() const
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
    std::shared_ptr<const hoomd::ExecutionConfiguration> getExecutionConfiguration() const
        {
        return m_hoomd_pdata->getExecConf();
        }

    //! Get the domain decomposition
    std::shared_ptr<hoomd::DomainDecomposition> getDomainDecomposition() const
        {
#ifdef ENABLE_MPI
        return m_hoomd_pdata->getDomainDecomposition();
#else
        return std::shared_ptr<hoomd::DomainDecomposition>();
#endif // ENABLE_MPI
        }

    std::shared_ptr<mpcd::ParticleDataSnapshot> particles; //!< MPCD particle data snapshot

    private:
    std::shared_ptr<hoomd::SystemDefinition> m_sysdef;  //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_hoomd_pdata; //!< Standard HOOMD particle data
    BoxDim m_global_box; //!< Global simulation box when snapshot is created
    };

namespace detail
    {
//! Export mpcd::SystemDataSnapshot to python
void export_SystemDataSnapshot(pybind11::module& m);
    }  // namespace detail

    }  // end namespace mpcd
    }  // end namespace hoomd
#endif // MPCD_SYSTEM_DATA_SNAPSHOT_H_
