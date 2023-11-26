// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>

#include <vector>

/*! \file PeriodicImproperForceCompute.h
    \brief Declares a class for computing periodic impropers
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __PERIODICIMPROPERFORCECOMPUTE_H__
#define __PERIODICIMPROPERFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct periodic_improper_params
    {
    Scalar k;
    Scalar d;
    int n;
    Scalar chi_0;

#ifndef __HIPCC__
    periodic_improper_params() : k(0.), d(0.), n(0), chi_0(0.) { }

    periodic_improper_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()), n(v["n"].cast<int>()),
          chi_0(v["chi0"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["chi0"] = chi_0;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

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

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar chi_0);

    virtual void setParamsPython(std::string type, pybind11::dict params);

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
    Scalar* m_K;     //!< K parameter for multiple improper tyes
    Scalar* m_sign;  //!< sign parameter for multiple improper types
    int* m_multi;    //!< multiplicity parameter for multiple improper types
    Scalar* m_chi_0; //!< chi_0 parameter for multiple improper types

    std::shared_ptr<ImproperData> m_improper_data; //!< Improper data to use in computing impropers

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
