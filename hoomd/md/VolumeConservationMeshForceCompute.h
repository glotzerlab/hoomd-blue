// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshForceCompute.h"
#include "VolumeConservationMeshParameters.h"

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
class PYBIND11_EXPORT VolumeConservationMeshForceCompute : public MeshForceCompute
    {
    public:
    //! Constructs the compute
    VolumeConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<MeshDefinition> meshdef,
                                       bool ignore_type);

    //! Destructor
    virtual ~VolumeConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, const volume_conservation_param_t& params);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual pybind11::array_t<Scalar> getVolume()
        {
        unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();
        if (m_ignore_type)
            n_types = 1;
        ArrayHandle<Scalar> h_volume(m_volume, access_location::host, access_mode::read);
        return pybind11::array(n_types, h_volume.data);
        }

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
    GPUArray<volume_conservation_param_t> m_params; //!< Parameters

    GPUArray<Scalar> m_volume; //!< memory space for volume

    Scalar m_volume_diff;

    bool m_ignore_type; //! do we ignore type to calculate global area

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;

    //! compute volumes
    void precomputeParameter() override;

    void postcomputeParameter(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id) override
        {
        ArrayHandle<Scalar> h_volume(m_volume, access_location::host, access_mode::readwrite);
        h_volume.data[type_id] += m_volume_diff;
        };

    virtual Scalar energyDiff(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id) override;
    };

namespace detail
    {
//! Exports the VolumeConservationMeshForceCompute class to python
void export_VolumeConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
