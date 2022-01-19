// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: dnlebard
#include "hoomd/MeshDefinition.h"
#include "hoomd/ForceCompute.h"

#include <memory>

#include <vector>

/*! \file MeshAreaConservationForceCompute.h
    \brief Declares a class for computing area conservation forces
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
#ifdef SINGLE_PRECISION
__attribute__((aligned(8)));
#else
__attribute__((aligned(16)));
#endif

//! Computes area conservation forces on the mesh
/*! Area Conservation forces are computed on every triangle in a mesh.
    \ingroup computes
*/
class PYBIND11_EXPORT AreaConservationMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    AreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~AreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A0);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
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
    Scalar* m_K;   //!< K parameter for multiple mesh triangles
    Scalar*m_A0;

    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing area conservation energy

    GlobalVector<Scalar> m_numerator_base; //! base of numerator term of area conservation energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute base of numerator term 
    virtual void computeNumeratorBase();
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceCompute class to python
void export_AreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
