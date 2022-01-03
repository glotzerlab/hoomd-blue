// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "HelfrichMeshForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file HelfrichMeshForceComputeGPU.h
    \brief Declares a class for computing helfrich energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __HELFRICHMESHFORCECOMPUTE_GPU_H__
#define __HELFRICHMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes helfrich energy forces on the mesh on the GPU
/*! Helfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT HelfrichMeshForceComputeGPU : public HelfrichMeshForceCompute
    {
    public:
    //! Constructs the compute
    HelfrichMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<MeshDefinition> meshdef);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        HelfrichMeshForceCompute::setAutotunerParams(enable, period);
        m_tuner_force->setPeriod(period);
        m_tuner_force->setEnabled(enable);
        m_tuner_sigma->setPeriod(period);
        m_tuner_sigma->setEnabled(enable);
        }

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K);

    protected:
    std::unique_ptr<Autotuner> m_tuner_force; //!< Autotuner for block size of force loop
    std::unique_ptr<Autotuner> m_tuner_sigma; //!< Autotuner for block size of sigma loop
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    };

namespace detail
    {
//! Exports the HelfrichMeshForceComputeGPU class to python
void export_HelfrichMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
