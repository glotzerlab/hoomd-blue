// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "DEMEvaluator.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

/*! \file DEM2DForceGPU.cuh
  \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by
  DEM2DForceComputeGPU.
*/

#ifndef __DEM2DFORCEGPU_CUH__
#define __DEM2DFORCEGPU_CUH__

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>

namespace hoomd
    {
namespace dem
    {
namespace kernel
    {
//! Kernel driver that computes 2D DEM forces on the GPU for DEM2DForceComputeGPU
template<typename Real, typename Real2, typename Real4, typename Evaluator>
hipError_t gpu_compute_dem2d_forces(Scalar4* d_force,
                                    Scalar4* d_torque,
                                    Scalar* d_virial,
                                    const size_t virial_pitch,
                                    const unsigned int N,
                                    const unsigned int n_ghosts,
                                    const Scalar4* d_pos,
                                    const Scalar4* d_quat,
                                    const Real2* d_vertices,
                                    const unsigned int* d_num_shape_verts,
                                    const Scalar* d_diameter,
                                    const Scalar4* d_velocity,
                                    const unsigned int vertexCount,
                                    const BoxDim& box,
                                    const unsigned int* d_n_neigh,
                                    const unsigned int* d_nlist,
                                    const size_t* d_head_list,
                                    const Evaluator evaluator,
                                    const Real r_cutsq,
                                    const unsigned int n_shapes,
                                    const unsigned int particlesPerBlock,
                                    const unsigned int maxVerts);

    } // end namespace kernel
    } // end namespace dem
    } // end namespace hoomd

#endif

#endif
