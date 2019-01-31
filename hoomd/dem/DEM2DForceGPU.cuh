// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "DEMEvaluator.h"

/*! \file DEM2DForceGPU.cuh
  \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by DEM2DForceComputeGPU.
*/

#ifndef __DEM2DFORCEGPU_CUH__
#define __DEM2DFORCEGPU_CUH__

#ifdef ENABLE_CUDA

//! Kernel driver that computes 2D DEM forces on the GPU for DEM2DForceComputeGPU
template<typename Real, typename Real2, typename Real4, typename Evaluator>
cudaError_t gpu_compute_dem2d_forces(
        Scalar4* d_force,
        Scalar4* d_torque,
        Scalar* d_virial,
        const unsigned int virial_pitch,
        const unsigned int N,
        const unsigned int n_ghosts,
        const Scalar4 *d_pos,
        const Scalar4 *d_quat,
        const Real2 *d_vertices,
        const unsigned int *d_num_shape_verts,
        const Scalar *d_diameter,
        const Scalar4 *d_velocity,
        const unsigned int vertexCount,
        const BoxDim& box,
        const unsigned int *d_n_neigh,
        const unsigned int *d_nlist,
        const unsigned int *d_head_list,
        const Evaluator evaluator,
        const Real r_cutsq,
        const unsigned int n_shapes,
        const unsigned int particlesPerBlock,
        const unsigned int maxVerts);

#endif

#endif
