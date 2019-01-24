// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/Index1D.h"
#include "DEMEvaluator.h"

/*! \file DEM3DForceGPU.cuh
  \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by DEM3DForceComputeGPU.
*/

#ifndef __DEM3DFORCEGPU_CUH__
#define __DEM3DFORCEGPU_CUH__

#ifdef ENABLE_CUDA

//! Kernel driver that computes 3D DEM forces on the GPU for DEM3DForceComputeGPU
template<typename Real,  typename Real4, typename Evaluator>
cudaError_t gpu_compute_dem3d_forces(
    Scalar4* d_force,
    Scalar4* d_torque,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const unsigned int n_ghosts,
    const Scalar4 *d_pos,
    const Scalar4 *d_quat,
    const unsigned int *d_nextFaces,
    const unsigned int *d_firstFaceVertices,
    const unsigned int *d_nextVertices,
    const unsigned int *d_realVertices,
    const Real4 *d_vertices,
    const Scalar *d_diam,
    const Scalar4 *d_velocity,
    const unsigned int maxFeatures,
    const unsigned int maxVertices,
    const unsigned int numFaces,
    const unsigned int numDegenerateVerts,
    const unsigned int numVerts,
    const unsigned int numEdges,
    const unsigned int numTypes,
    const BoxDim& box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    const Evaluator evaluator,
    const Real r_cutsq,
    const unsigned int particlesPerBlock,
    const unsigned int *d_firstTypeVert,
    const unsigned int *d_numTypeVerts,
    const unsigned int *d_firstTypeEdge,
    const unsigned int *d_numTypeEdges,
    const unsigned int *d_numTypeFaces,
    const unsigned int *d_vertexConnectivity,
    const unsigned int *d_edges);

#endif

#endif
