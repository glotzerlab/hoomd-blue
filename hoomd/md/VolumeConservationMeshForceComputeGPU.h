// Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

//! Computes helfrich energy forces on the mesh on the GPU
/*! Helfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT VolumeConservationMeshForceComputeGPU
    : public VolumeConservationMeshForceCompute
    {
    public:
    //! Constructs the compute
    VolumeConservationMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<MeshDefinition> meshdef);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        VolumeConservationMeshForceCompute::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar V0);

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory
    unsigned int m_num_blocks;       //!< number of memory blocks reserved for partial sum memory

    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size of force loop
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution
    GPUArray<Scalar2> m_params;          //!< Parameters stored on the GPU

    GPUArray<Scalar> m_partial_sum; //!< memory space for partial sum over volume
    GPUArray<Scalar> m_sum;          //!< memory space for sum over volume

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute volumes
    virtual void computeVolume();

    private:
    //! allocate the memory needed to store partial sums
    void resizePartialSumArrays();
    };

namespace detail
    {
//! Exports the VolumeConservationMeshForceComputeGPU class to python
void export_VolumeConservationMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
