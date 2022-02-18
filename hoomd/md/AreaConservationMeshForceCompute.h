// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

#include <vector>

/*! \file AreaConservationMeshForceCompute.h
    \brief Declares a class for computing area constraint forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __AREACONSERVATIONMESHFORCECOMPUTE_H__
#define __AREACONSERVATIONMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct aconstraint_params
    {
    Scalar k;
    Scalar A_mesh;

#ifndef __HIPCC__
    aconstraint_params() : k(0), A_mesh(0) { }

    aconstraint_params(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), A_mesh(params["A_mesh"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["A_mesh"] = A_mesh;
        return v;
        }
#endif
    }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes area constraint forces on the mesh
/*! Area constraint forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT AreaConservationMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    AreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~AreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A_mesh);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    Scalar getArea()
        {
        return m_area;
        };

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
    Scalar* m_K; //!< K parameter for multiple mesh triangles

    Scalar* m_Amesh;

    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing helfich energy

    Scalar m_area; //! sum of the triangle areas within the mesh

    Scalar m_area_diff;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute areas
    virtual void precomputeParameter();

    virtual void
    postcompute(unsigned int idx_a, unsigned int idx_b, unsigned int idx_c, unsigned int idx_d)
        {
        m_area += m_area_diff;
        };

    virtual Scalar energyDiff(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id);
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceCompute class to python
void export_AreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
