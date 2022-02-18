// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

#include <vector>

/*! \file MeshAreaConservationForceCompute.h
    \brief Declares a class for computing area conservation forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_H__
#define __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct triangle_area_conservation_params
    {
    Scalar k;
    Scalar A_mesh;

#ifndef __HIPCC__
    triangle_area_conservation_params() : k(0), A_mesh(0) { }

    triangle_area_conservation_params(pybind11::dict params)
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

//! Computes area conservation forces on the mesh
/*! Triangle Area Conservation forces are computed on every triangle in a mesh.
    \ingroup computes
*/
class PYBIND11_EXPORT TriangleAreaConservationMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    TriangleAreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~TriangleAreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A_mesh);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual Scalar getArea()
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
    Scalar m_area;

    std::shared_ptr<MeshDefinition>
        m_mesh_data; //!< Mesh data to use in computing area conservation energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the TriangleAreaConservationMeshForceCompute class to python
void export_TriangleAreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
