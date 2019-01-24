// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SFCPackUpdaterGPU.cu
    \brief Defines GPU kernel code for generating the space-filling curve sorted order on the GPU. Used by SFCPackUpdaterGPU.
*/

#include "SFCPackUpdaterGPU.cuh"
#include "hoomd/extern/kernels/mergesort.cuh"

//! Kernel to bin particles
template<bool twod>
__global__ void gpu_sfc_bin_particles_kernel(unsigned int N,
    const Scalar4 *d_pos,
    unsigned int *d_particle_bins,
    const unsigned int *d_traversal_order,
    unsigned int n_grid,
    unsigned int *d_sorted_order,
    const BoxDim box)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    // fetch particle position
    Scalar4 postype = d_pos[idx];
    Scalar3 p = make_scalar3(postype.x, postype.y, postype.z);

    Scalar3 f = box.makeFraction(p);
    int ib = (unsigned int)(f.x * n_grid) % n_grid;
    int jb = (unsigned int)(f.y * n_grid) % n_grid;
    int kb = (unsigned int)(f.z * n_grid) % n_grid;

    // if the particle is slightly outside, move back into grid
    if (ib < 0) ib = 0;
    if (ib >= n_grid) ib = n_grid - 1;

    if (jb < 0) jb = 0;
    if (jb >= n_grid) jb = n_grid - 1;

    if (kb < 0) kb = 0;
    if (kb >= n_grid) kb = n_grid - 1;

    // record its bin
    unsigned int bin;
    if (twod)
        {
        // do not use Hilbert curve in 2D
        bin = ib*n_grid + jb;
        d_particle_bins[idx] = bin;
        }
    else
        {
        bin = ib*(n_grid*n_grid) + jb * n_grid + kb;
        d_particle_bins[idx] = d_traversal_order[bin];
        }

    // store index of ptl
    d_sorted_order[idx] = idx;
    }

/*! \param N number of local particles
    \param d_pos Device array of positions
    \param d_particle_bins Device array of particle bins
    \param d_traversal_order Device array of Hilbert-curve bins
    \param n_grid Number of grid elements along one edge
    \param d_sorted_order Sorted order of particles
    \param box Box dimensions
    \param twod If true, bin particles in two dimensions
    */
void gpu_generate_sorted_order(unsigned int N,
        const Scalar4 *d_pos,
        unsigned int *d_particle_bins,
        unsigned int *d_traversal_order,
        unsigned int n_grid,
        unsigned int *d_sorted_order,
        const BoxDim& box,
        bool twod,
        mgpu::ContextPtr mgpu_context)
    {
    // maybe need to autotune, but SFCPackUpdater is called infrequently
    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;

    if (twod)
        gpu_sfc_bin_particles_kernel<true><<<n_blocks, block_size>>>(N, d_pos, d_particle_bins, d_traversal_order, n_grid, d_sorted_order, box);
    else
        gpu_sfc_bin_particles_kernel<false><<<n_blocks, block_size>>>(N, d_pos, d_particle_bins, d_traversal_order, n_grid, d_sorted_order, box);

    // Sort particles
    if (N)
        mgpu::MergesortPairs(d_particle_bins, d_sorted_order, N, *mgpu_context);
    }

//! Kernel to apply sorted order
__global__ void gpu_apply_sorted_order_kernel(
        unsigned int N,
        unsigned int n_ghost,
        const unsigned int *d_sorted_order,
        const Scalar4 *d_pos,
        Scalar4 *d_pos_alt,
        const Scalar4 *d_vel,
        Scalar4 *d_vel_alt,
        const Scalar3 *d_accel,
        Scalar3 *d_accel_alt,
        const Scalar *d_charge,
        Scalar *d_charge_alt,
        const Scalar *d_diameter,
        Scalar *d_diameter_alt,
        const int3 *d_image,
        int3 *d_image_alt,
        const unsigned int *d_body,
        unsigned int *d_body_alt,
        const unsigned int *d_tag,
        unsigned int *d_tag_alt,
        const Scalar4 *d_orientation,
        Scalar4 *d_orientation_alt,
        const Scalar4 *d_angmom,
        Scalar4 *d_angmom_alt,
        const Scalar3 *d_inertia,
        Scalar3 *d_inertia_alt,
        const Scalar *d_net_virial,
        Scalar *d_net_virial_alt,
        unsigned int virial_pitch,
        const Scalar4 *d_net_force,
        Scalar4 *d_net_force_alt,
        const Scalar4 *d_net_torque,
        Scalar4 *d_net_torque_alt,
        unsigned int *d_rtag)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N+n_ghost) return;

    // apply sorted order only for local ptls
    unsigned int old_idx = (idx < N ? d_sorted_order[idx] : idx);

    // permute and copy over particle data
    d_pos_alt[idx] = d_pos[old_idx];
    d_vel_alt[idx] = d_vel[old_idx];
    d_accel_alt[idx] = d_accel[old_idx];
    d_charge_alt[idx] = d_charge[old_idx];
    d_diameter_alt[idx] = d_diameter[old_idx];
    d_image_alt[idx] = d_image[old_idx];
    d_body_alt[idx] = d_body[old_idx];
    unsigned int tag = d_tag[old_idx];
    d_tag_alt[idx] = tag;
    d_orientation_alt[idx] = d_orientation[old_idx];
    d_angmom_alt[idx] = d_angmom[old_idx];
    d_inertia_alt[idx] = d_inertia[old_idx];
    d_net_virial_alt[0*virial_pitch+idx] = d_net_virial[0*virial_pitch+old_idx];
    d_net_virial_alt[1*virial_pitch+idx] = d_net_virial[1*virial_pitch+old_idx];
    d_net_virial_alt[2*virial_pitch+idx] = d_net_virial[2*virial_pitch+old_idx];
    d_net_virial_alt[3*virial_pitch+idx] = d_net_virial[3*virial_pitch+old_idx];
    d_net_virial_alt[4*virial_pitch+idx] = d_net_virial[4*virial_pitch+old_idx];
    d_net_virial_alt[5*virial_pitch+idx] = d_net_virial[5*virial_pitch+old_idx];
    d_net_force_alt[idx] = d_net_force[old_idx];
    d_net_torque_alt[idx] = d_net_torque[old_idx];

    if (idx < N)
        {
        // update rtag to point to particle position in new arrays
        d_rtag[tag] = idx;
        }
    }

void gpu_apply_sorted_order(
        unsigned int N,
        unsigned int n_ghost,
        const unsigned int *d_sorted_order,
        const Scalar4 *d_pos,
        Scalar4 *d_pos_alt,
        const Scalar4 *d_vel,
        Scalar4 *d_vel_alt,
        const Scalar3 *d_accel,
        Scalar3 *d_accel_alt,
        const Scalar *d_charge,
        Scalar *d_charge_alt,
        const Scalar *d_diameter,
        Scalar *d_diameter_alt,
        const int3 *d_image,
        int3 *d_image_alt,
        const unsigned int *d_body,
        unsigned int *d_body_alt,
        const unsigned int *d_tag,
        unsigned int *d_tag_alt,
        const Scalar4 *d_orientation,
        Scalar4 *d_orientation_alt,
        const Scalar4 *d_angmom,
        Scalar4 *d_angmom_alt,
        const Scalar3 *d_inertia,
        Scalar3 *d_inertia_alt,
        const Scalar *d_net_virial,
        Scalar *d_net_virial_alt,
        unsigned int virial_pitch,
        const Scalar4 *d_net_force,
        Scalar4 *d_net_force_alt,
        const Scalar4 *d_net_torque,
        Scalar4 *d_net_torque_alt,
        unsigned int *d_rtag
        )
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = (N+n_ghost)/block_size + 1;

    gpu_apply_sorted_order_kernel<<<n_blocks, block_size>>>(N,
        n_ghost,
        d_sorted_order,
        d_pos,
        d_pos_alt,
        d_vel,
        d_vel_alt,
        d_accel,
        d_accel_alt,
        d_charge,
        d_charge_alt,
        d_diameter,
        d_diameter_alt,
        d_image,
        d_image_alt,
        d_body,
        d_body_alt,
        d_tag,
        d_tag_alt,
        d_orientation,
        d_orientation_alt,
        d_angmom,
        d_angmom_alt,
        d_inertia,
        d_inertia_alt,
        d_net_virial,
        d_net_virial_alt,
        virial_pitch,
        d_net_force,
        d_net_force_alt,
        d_net_torque,
        d_net_torque_alt,
        d_rtag);
    }
