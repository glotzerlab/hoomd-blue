// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

#include <vector>

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
struct helfrich_params
    {
    Scalar k;

#ifndef __HIPCC__
    helfrich_params() : k(0) { }

    helfrich_params(pybind11::dict params) : k(params["k"].cast<Scalar>()) { }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        return v;
        }
#endif
    }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Computes helfrich energy forces on the mesh
/*! Helfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT HelfrichMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    HelfrichMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~HelfrichMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K);

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
    Scalar* m_K; //!< K parameter for multiple mesh triangles

    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing helfich energy

    GlobalArray<Scalar3>
        m_sigma_dash; //! sum of the distances weighted by the bending angle over all neighbors

    GlobalArray<Scalar>
        m_sigma; //! sum of the vectors weighted by the bending angle over all neighbors

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute sigmas
    virtual void computeSigma();
    };

namespace detail
    {
//! Exports the HelfrichMeshForceCompute class to python
void export_HelfrichMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
