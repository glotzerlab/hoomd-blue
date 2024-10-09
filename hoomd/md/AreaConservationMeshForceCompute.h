// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

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
struct area_conservation_params
    {
    Scalar k;
    Scalar A0;

#ifndef __HIPCC__
    area_conservation_params() : k(0), A0(0) { }

    area_conservation_params(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), A0(params["A0"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["A0"] = A0;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
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
                                     std::shared_ptr<MeshDefinition> meshdef,
                                     bool ignore_type);

    //! Destructor
    virtual ~AreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A0);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual pybind11::array_t<Scalar> getArea()
        {
        return pybind11::array(m_mesh_data->getMeshTriangleData()->getNTypes(), m_area);
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
    GPUArray<Scalar2> m_params;                  //!< Parameters
    Scalar* m_area;                              //! sum of the triangle areas within a mesh type
                                                 //
    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing energy
    bool m_ignore_type;                          //! ignore type to calculate global area if true

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute areas
    virtual void precomputeParameter();
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceCompute class to python
void export_AreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
