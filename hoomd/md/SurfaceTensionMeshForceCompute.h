// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

#include <vector>

/*! \file SurfaceTensionForceCompute.h
    \brief Declares a class for computing surface tension forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __SURFACETENSIONMESHFORCECOMPUTE_H__
#define __SURFACETENSIONMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct surface_tension_params
    {
    Scalar sigma;

#ifndef __HIPCC__
    surface_tension_params() : sigma(0) { }

    surface_tension_params(pybind11::dict params) : sigma(params["sigma"].cast<Scalar>()) { }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["sigma"] = sigma;
        return v;
        }
#endif
    }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes surface tension forces on the mesh
/*! Surface tension forces are computed on every triangle in a mesh.
    \ingroup computes
*/
class PYBIND11_EXPORT SurfaceTensionMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    SurfaceTensionMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~SurfaceTensionMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar sigma);

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
    Scalar* m_sigma; //!< sigma surface tension term
    Scalar m_area;

    std::shared_ptr<MeshDefinition>
        m_mesh_data; //!< Mesh data to use in computing area conservation energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the SurfaceTensionMeshForceCompute class to python
void export_SurfaceTensionMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
