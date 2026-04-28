// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "LineTensionForceCompute.h"
#include "HelfrichMeshParameters.h"

#include <memory>

/*! \file HelfrichGeneralLineTensionForceCompute.h
    \brief Declares a class for computing helfrich energy forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __HELFRICHGENERALMESHFORCECOMPUTE_H__
#define __HELFRICHGENERALMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Computes helfrich energy forces on the mesh
/*! HelfrichGeneral energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT HelfrichGeneralLineTensionForceCompute : public LineTensionForceCompute
    {
    public:
    //! Constructs the compute
    HelfrichGeneralLineTensionForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~HelfrichGeneralLineTensionForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, const helfrich_param_t& params);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    CommFlags getRequestedCommFlags(uint64_t timestep) override
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    GPUArray<helfrich_param_t> m_params; //!< Parameters

    GPUArray<Scalar3>
        m_sigma_dash; //! sum of the distances weighted by the bending angle over all neighbors

    GPUArray<Scalar>
        m_sigma; //! sum of the vectors weighted by the bending angle over all neighbors
		 //
    GPUArray<Scalar3>
        m_normal; //! normal vector at each vertex

    Scalar m_sigma_diff_a;
    Scalar m_sigma_diff_b;
    Scalar m_sigma_diff_c;
    Scalar m_sigma_diff_d;

    Scalar m_factor_a;
    Scalar m_factor_b;
    Scalar m_factor_c;
    Scalar m_factor_d;

    Scalar3 m_sigma_dash_diff_a;
    Scalar3 m_sigma_dash_diff_b;
    Scalar3 m_sigma_dash_diff_c;
    Scalar3 m_sigma_dash_diff_d;

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;

    Scalar energyDiff(unsigned int idx_a,
                      unsigned int idx_b,
                      unsigned int idx_c,
                      unsigned int idx_d,
                      unsigned int type_id) override;

    //! compute sigmas
    void precomputeParameter() override;

    void postcomputeParameter(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id) override;
    };

namespace detail
    {
//! Exports the HelfrichGeneralLineTensionForceCompute class to python
void export_HelfrichGeneralLineTensionForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
