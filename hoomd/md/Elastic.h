// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#pragma once

namespace hoomd
    {
namespace md
    {
struct ElasticCoefficients
    {
	Scalar elastic_coeff_1, elastic_coeff_2, elastic_coeff_3;	
    };

//! Computes elastic forces on the tetrahedral mesh
/*! Elastic forces are computed on every tetrahedron in a mesh.
 */
class PYBIND11_EXPORT Elastic : public ForceCompute
    {
    public:
    //! Constructs the compute
    Elastic(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<TetrahedronData> meshdef,
            pybind11::array_t<Scalar> reference_positions,
            pybind11::array_t<unsigned int> reference_tags);

    //! Destructor
    virtual ~Elastic(){};

    //! Set the parameters of the tetrahedra types
    virtual void setParams(unsigned int type, const ElasticCoefficients& params);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual void setReference(pybind11::array_t<Scalar> reference_positions,
                              pybind11::array_t<unsigned int> reference_tags);

    void computeForces(uint64_t timestep) override;

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
    GPUArray<ElasticCoefficients> m_params; //!< Parameters
    GPUArray<vec3<Scalar>>
        m_reference_vertex_displacements; //!< memory space for tetrahedra reference positions.
                                          //!< Indexed by (vertex, tetrahedron) see slide 33
    GPUArray<vec3<Scalar>> m_reference_inv_matrix;
    Index2D m_matrix_indexer;

    std::shared_ptr<TetrahedronData>
        m_tetrahedron_data; //!< Tetrahedron data to use in computing elastic forces
    };

namespace detail
    {
//! Exports the TriangleAreaConservationMeshForceCompute class to python
void export_Elastic(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
