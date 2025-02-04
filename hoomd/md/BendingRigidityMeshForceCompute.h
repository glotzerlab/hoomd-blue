// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

/*! \file BendingRigidityMeshForceCompute.h
    \brief Declares a class for computing bending rigidity energy forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __BENDINGRIGIDITYMESHFORCECOMPUTE_H__
#define __BENDINGRIGIDITYMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct bending_params
    {
    Scalar k;

#ifndef __HIPCC__
    bending_params() : k(0) { }

    bending_params(pybind11::dict params) : k(params["k"].cast<Scalar>()) { }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

//! Computes rigidity energy forces on the mesh
/*! BendingRigidity energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT BendingRigidityMeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    BendingRigidityMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~BendingRigidityMeshForceCompute();

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
    GPUArray<Scalar> m_params;                   //!< Parameters
    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing
                                                 // the bending energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the BendingRigidityMeshForceCompute class to python
void export_BendingRigidityMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
