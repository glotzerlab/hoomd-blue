// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproper.h"
#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>

#include <vector>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#pragma once

namespace hoomd
    {
namespace md
    {
//! Computes periodic improper on each particle
/*! Periodic improper forces are computed on every particle in the simulation.

    The impropers which forces are computed on are accessed from ParticleData::getimproperData
    \ingroup computes
*/
class PYBIND11_EXPORT PeriodicImproperForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    PeriodicImproperForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~PeriodicImproperForceCompute();

    void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a particular type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    /// Potential parameters
    GPUArray<periodic_improper_params> m_params;

    std::shared_ptr<ImproperData> m_improper_data; //!< Improper data to use in computing impropers

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd
