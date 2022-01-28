// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "AreaConservationMeshForceCompute.h"
#include "AreaConservationMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file AreaConservationMeshForceComputeGPU.h
    \brief Declares a class for computing area conservation energy forces on the GPU
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

//! Computes area conservation energy forces on the mesh on the GPU
/*! AreaConservation energy forces are computed on every particle in a mesh.

    \ingroup computes

*/
class PYBIND11_EXPORT AreaConservationMeshForceComputeGPU : public AreaConservationMeshForceCompute
    {
    public:
    //! Constructs the compute
    AreaConservationMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<MeshDefinition> meshdef);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        AreaConservationMeshForceCompute::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar A0);

    protected:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size of force loop
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution
    GPUArray<Scalar> m_params;         //!< Parameters stored on the GPU

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the AreaConservationMeshForceComputeGPU class to python
void export_AreaConservationMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
