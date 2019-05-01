// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"

// Maintainer: jglaser

/*! \file ForceComposite.cuh
    \brief Defines GPU driver functions for the composite particle integration on the GPU.
*/

cudaError_t gpu_rigid_force(Scalar4* d_force,
                 Scalar4* d_torque,
                 const unsigned int *d_molecule_len,
                 const unsigned int *d_molecule_list,
                 const unsigned int *d_molecule_idx,
                 const unsigned int *d_rigid_center,
                 Index2D molecule_indexer,
                 const Scalar4 *d_postype,
                 const Scalar4* d_orientation,
                 Index2D body_indexer,
                 Scalar3* d_body_pos,
                 Scalar4* d_body_orientation,
                 const unsigned int *d_body_len,
                 const unsigned int *d_body,
                 const unsigned int *d_tag,
                 uint2 *d_flag,
                 Scalar4* d_net_force,
                 Scalar4* d_net_torque,
                 unsigned int n_mol,
                 unsigned int N,
                 unsigned int n_bodies_per_block,
                 unsigned int block_size,
                 const cudaDeviceProp& dev_prop,
                 bool zero_force,
                 const GPUPartition &gpu_partition);

cudaError_t gpu_rigid_virial(Scalar* d_virial,
                 const unsigned int *d_molecule_len,
                 const unsigned int *d_molecule_list,
                 const unsigned int *d_molecule_idx,
                 const unsigned int *d_rigid_center,
                 Index2D molecule_indexer,
                 const Scalar4 *d_postype,
                 const Scalar4* d_orientation,
                 Index2D body_indexer,
                 Scalar3* d_body_pos,
                 Scalar4* d_body_orientation,
                 Scalar4* d_net_force,
                 Scalar *d_net_virial,
                 const unsigned int *d_body,
                 const unsigned int *d_tag,
                 unsigned int n_mol,
                 unsigned int N,
                 unsigned int n_bodies_per_block,
                 unsigned int net_virial_pitch,
                 unsigned int virial_pitch,
                 unsigned int block_size,
                 const cudaDeviceProp& dev_prop,
                 const GPUPartition& gpu_partition);


void gpu_update_composite(unsigned int N,
    unsigned int n_ghost,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    Index2D body_indexer,
    const unsigned int *d_lookup_center,
    const Scalar3 *d_body_pos,
    const Scalar4 *d_body_orientation,
    const unsigned int *d_body_len,
    const unsigned int *d_molecule_order,
    const unsigned int *d_molecule_len,
    const unsigned int *d_molecule_idx,
    int3 *d_image,
    const BoxDim box,
    const BoxDim global_box,
    unsigned int block_size,
    uint2 *d_flag,
    const GPUPartition &gpu_partition);


cudaError_t gpu_find_rigid_centers(const unsigned int *d_body,
                                const unsigned int *d_tag,
                                const unsigned int *d_rtag,
                                const unsigned int N,
                                const unsigned int nghost,
                                unsigned int *d_rigid_center,
                                unsigned int *d_lookup_center,
                                unsigned int &n_rigid);
