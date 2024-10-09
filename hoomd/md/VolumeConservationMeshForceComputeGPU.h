// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "VolumeConservationMeshForceCompute.h"
#include "VolumeConservationMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file VolumeConservationMeshForceComputeGPU.h
    \brief Declares a class for computing volume constraint forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __VOLUMECONSERVATIONMESHFORCECOMPUTE_GPU_H__
#define __VOLUMECONSERVATIONMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes volume conservation energy forces on the mesh on the GPU
/*! Volume energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT VolumeConservationMeshForceComputeGPU
    : public VolumeConservationMeshForceCompute
    {
    public:
    //! Constructs the compute
    VolumeConservationMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<MeshDefinition> meshdef,
					  bool ignore_type);

    virtual pybind11::array_t<Scalar> getVolume()
        {
        ArrayHandle<Scalar> h_volume(m_volume_GPU, access_location::host, access_mode::read);
        return pybind11::array(m_mesh_data->getMeshTriangleData()->getNTypes(), h_volume.data);
        }

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory
    unsigned int m_num_blocks;       //!< number of memory blocks reserved for partial sum memory

    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    GPUArray<Scalar> m_partial_sum; //!< memory space for partial sum over volume
    GPUArray<Scalar> m_sum;          //!< memory space for sum over volume

    GPUArray<Scalar> m_volume_GPU;          //!< memory space for sum over volume

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute volumes
    virtual void computeVolume();

    };

namespace detail
    {
//! Exports the VolumeConservationMeshForceComputeGPU class to python
void export_VolumeConservationMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
