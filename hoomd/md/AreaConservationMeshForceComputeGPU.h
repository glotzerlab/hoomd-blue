// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AreaConservationMeshForceCompute.h"
#include "AreaConservationMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file AreaConservationMeshForceComputeGPU.h
    \brief Declares a class for computing area constraint forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __AREACONSERVATIONMESHFORCECOMPUTE_GPU_H__
#define __AREACONSERVATIONMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes area energy forces on the mesh on the GPU
/*! Area conservation energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT AreaConservationMeshForceComputeGPU : public AreaConservationMeshForceCompute
    {
    public:
    //! Constructs the compute
    AreaConservationMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<MeshDefinition> meshdef,
					bool ignore_type);

    virtual pybind11::array_t<Scalar> getArea()
        {
        ArrayHandle<Scalar> h_area(m_area_GPU, access_location::host, access_mode::read);
        return pybind11::array(m_mesh_data->getMeshTriangleData()->getNTypes(), h_area.data);
        }

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory
    unsigned int m_num_blocks; //!< number of memory blocks reserved for partial sum memory

    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    GPUArray<Scalar> m_partial_sum; //!< memory space for partial sum over area
    GPUArray<Scalar> m_sum;         //!< memory space for sum over area

    GPUArray<Scalar> m_area_GPU; //!< memory space for area

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute areas
    virtual void precomputeParameter();
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceComputeGPU class to python
void export_AreaConservationMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
