// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichMeshForceCompute.h"
#include "HelfrichMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file HelfrichMeshForceComputeGPU.h
    \brief Declares a class for computing helfrich energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

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
    HelfrichMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<MeshDefinition> meshdef);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_force;
    std::shared_ptr<Autotuner<1>> m_tuner_sigma;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! compute sigmas
    virtual void computeSigma();
    };

namespace detail
    {
//! Exports the HelfrichMeshForceComputeGPU class to python
void export_HelfrichMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
