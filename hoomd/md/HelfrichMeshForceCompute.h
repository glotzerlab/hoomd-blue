// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichMeshParameters.h"
#include "MeshForceCompute.h"
#include "hoomd/PythonLocalDataAccess.h"

#include <memory>

/*! \file HelfrichMeshForceCompute.h
    \brief Declares a class for computing helfrich energy forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __HELFRICHMESHFORCECOMPUTE_H__
#define __HELFRICHMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Computes helfrich energy forces on the mesh
/*! Helfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT HelfrichMeshForceCompute : public MeshForceCompute
    {
    public:
    //! Constructs the compute
    HelfrichMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~HelfrichMeshForceCompute();

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

    /** Get per particle energies

        @returns a Numpy array with per particle energies in increasing tag order.
    */
    pybind11::object getCurvaturesPython();

    protected:
    GPUArray<helfrich_param_t> m_params; //!< Parameters

    bool m_has_spontaneous_curvature
        = false; //! Check if any particle type has a non-zero spontaneous curvature H0

    bool m_force_normals = false; //! Compute the normals in precomputeParameter() even without a
                                  //! spontaneous curvature

    GPUArray<Scalar3>
        m_sigma_dash; //! sum of the distances weighted by the bending angle over all neighbors

    GPUArray<Scalar>
        m_sigma; //! sum of the vectors weighted by the bending angle over all neighbors
                 //
    GPUArray<Scalar4> m_normal; //! normal vector at each vertex

    Scalar m_sigma_diff_a;
    Scalar m_sigma_diff_b;
    Scalar m_sigma_diff_c;
    Scalar m_sigma_diff_d;

    Scalar4 m_normal_diff_a;
    Scalar4 m_normal_diff_b;
    Scalar4 m_normal_diff_c;
    Scalar4 m_normal_diff_d;

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
//! Exports the HelfrichMeshForceCompute class to python
void export_HelfrichMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
