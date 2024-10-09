// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

/*! \file VolumeConservationMeshForceCompute.h
    \brief Declares a class for computing volume constraint forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __VOLUMECONSERVATIONMESHFORCECOMPUTE_H__
#define __VOLUMECONSERVATIONMESHFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {

//! Computes volume constraint forces on the mesh
/*! Volume constraint forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT VolumeConservationMeshForceCompute : public ForceCompute
    {
    struct volume_conservation_params
        {
        Scalar k;
        Scalar V0;

#ifndef __HIPCC__
        volume_conservation_params() : k(0), V0(0) { }

        volume_conservation_params(pybind11::dict params)
            : k(params["k"].cast<Scalar>()), V0(params["V0"].cast<Scalar>())
            {
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["k"] = k;
            v["V0"] = V0;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(4)));
#else
        __attribute__((aligned(8)));
#endif
    public:
    //! Constructs the compute
    VolumeConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<MeshDefinition> meshdef,
                                       bool ignore_type);

    //! Destructor
    virtual ~VolumeConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar V0);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual pybind11::array_t<Scalar> getVolume()
        {
        return pybind11::array(m_mesh_data->getMeshTriangleData()->getNTypes(), m_volume);
        }

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
    GPUArray<Scalar2> m_params; //!< Parameters

    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing volume energy

    Scalar* m_volume; //! sum of the triangle areas within the mesh

    bool m_ignore_type; //! do we ignore type to calculate global area

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute volumes
    virtual void computeVolume();
    };

namespace detail
    {
//! Exports the VolumeConservationMeshForceCompute class to python
void export_VolumeConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
