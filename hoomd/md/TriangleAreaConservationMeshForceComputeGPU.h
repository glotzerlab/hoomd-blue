// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TriangleAreaConservationMeshForceCompute.h"
#include "TriangleAreaConservationMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file TriangleAreaConservationMeshForceComputeGPU.h
    \brief Declares a class for computing area conservation energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_GPU_H__
#define __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes area conservation energy forces on the mesh on the GPU
/*! TriangleAreaConservation energy forces are computed on every particle in a mesh.

    \ingroup computes

*/
class PYBIND11_EXPORT TriangleAreaConservationMeshForceComputeGPU
    : public TriangleAreaConservationMeshForceCompute
    {
    public:
    //! Constructs the compute
    TriangleAreaConservationMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                std::shared_ptr<MeshDefinition> meshdef);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        TriangleAreaConservationMeshForceCompute::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A_mesh);

    virtual Scalar getArea()
        {
        computeArea();
        return m_area;
        }

    protected:
    unsigned int m_block_size; //!< block size for partial sum memory
    unsigned int m_num_blocks; //!< number of memory blocks reserved for partial sum memory

    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size of force loop
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution
    GPUArray<Scalar2> m_params;         //!< Parameters stored on the GPU

    GPUArray<Scalar> m_partial_sum; //!< memory space for partial sum over volume
    GPUArray<Scalar> m_sum;         //!< memory space for sum over volume

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    virtual void computeArea();
    };

namespace detail
    {
//! Exports the TriangleAreaConservationMeshForceComputeGPU class to python
void export_TriangleAreaConservationMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
