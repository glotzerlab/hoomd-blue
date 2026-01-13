// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CurvatureHelfrichMeshForceCompute.h"
#include "CurvatureHelfrichMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file CurvatureHelfrichMeshForceComputeGPU.h
    \brief Declares a class for computing helfrich energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __CURVATUREHELFRICHMESHFORCECOMPUTE_GPU_H__
#define __CURVATUREHELFRICHMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes helfrich energy forces on the mesh on the GPU
/*! CurvatureHelfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT CurvatureHelfrichMeshForceComputeGPU : public CurvatureHelfrichMeshForceCompute
    {
    public:
    //! Constructs the compute
    CurvatureHelfrichMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
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
//! Exports the CurvatureHelfrichMeshForceComputeGPU class to python
void export_CurvatureHelfrichMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
