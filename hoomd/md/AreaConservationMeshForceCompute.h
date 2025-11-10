// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AreaConservationMeshParameters.h"
#include "MeshForceCompute.h"

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
//! Computes area constraint forces on the mesh
/*! Area constraint forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT AreaConservationMeshForceCompute : public MeshForceCompute
    {
    public:
    //! Constructs the compute
    AreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<MeshDefinition> meshdef,
                                     bool ignore_type);

    //! Destructor
    virtual ~AreaConservationMeshForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, const area_conservation_param_t& params);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual pybind11::array_t<Scalar> getArea()
        {
        unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();
        if (m_ignore_type)
            n_types = 1;
        ArrayHandle<Scalar> h_area(m_area, access_location::host, access_mode::read);
        return pybind11::array(n_types, h_area.data);
        };

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
    GPUArray<area_conservation_param_t> m_params; //!< Parameters
    GPUArray<Scalar> m_area;                      //!< memory space for area
    Scalar m_area_diff;
    bool m_ignore_type; //! ignore type to calculate global area if true

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;

    //! compute areas
    void precomputeParameter() override;

    void postcomputeParameter(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id) override
        {
        ArrayHandle<Scalar> h_area(m_area, access_location::host, access_mode::readwrite);
        h_area.data[type_id] += m_area_diff;
        };

    Scalar energyDiff(unsigned int idx_a,
                      unsigned int idx_b,
                      unsigned int idx_c,
                      unsigned int idx_d,
                      unsigned int type_id) override;
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceCompute class to python
void export_AreaConservationMeshForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
