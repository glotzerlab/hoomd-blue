// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichGeneralMeshForceCompute.h"
#include "HelfrichGeneralMeshForceComputeGPU.cuh"
#include "HelfrichMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file HelfrichGeneralMeshForceComputeGPU.h
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
/*! HelfrichGeneral energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT HelfrichGeneralMeshForceComputeGPU : public HelfrichGeneralMeshForceCompute
    {
    public:
    //! Constructs the compute
    HelfrichGeneralMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<MeshDefinition> meshdef);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_force;
    std::shared_ptr<Autotuner<1>> m_tuner_sigma;

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;

    //! compute sigmas
    void precomputeParameter() override;
    };

namespace detail
    {
//! Exports the HelfrichGeneralMeshForceComputeGPU class to python
void export_HelfrichGeneralMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
