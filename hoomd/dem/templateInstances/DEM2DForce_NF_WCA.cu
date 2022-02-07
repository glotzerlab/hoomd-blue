// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "../DEM2DForceGPU.cu"

namespace hoomd
    {
namespace dem
    {
namespace kernel
    {
typedef DEMEvaluator<Scalar, Scalar4, WCAPotential<Scalar, Scalar4, NoFriction<Scalar>>> WCADEM;

template hipError_t
gpu_compute_dem2d_forces<Scalar, Scalar2, Scalar4, WCADEM>(Scalar4* d_force,
                                                           Scalar4* d_torque,
                                                           Scalar* d_virial,
                                                           const size_t virial_pitch,
                                                           const unsigned int N,
                                                           const unsigned int n_ghosts,
                                                           const Scalar4* d_pos,
                                                           const Scalar4* d_quat,
                                                           const Scalar2* d_vertices,
                                                           const unsigned int* d_num_shape_verts,
                                                           const Scalar* d_diam,
                                                           const Scalar4* d_velocity,
                                                           const unsigned int vertexCount,
                                                           const BoxDim& box,
                                                           const unsigned int* d_n_neigh,
                                                           const unsigned int* d_nlist,
                                                           const size_t* d_head_list,
                                                           const WCADEM potential,
                                                           const Scalar r_cutsq,
                                                           const unsigned int n_shapes,
                                                           const unsigned int particlesPerBlock,
                                                           const unsigned int maxVerts);

    } // end namespace kernel
    } // end namespace dem
    } // end namespace hoomd
