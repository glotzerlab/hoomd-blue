/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

/*! \file SFCPackUpdaterGPU.cu
    \brief Defines GPU kernel code for generating the space-filling curve sorted order on the GPU. Used by SFCPackUpdaterGPU.
*/

#include "SFCPackUpdaterGPU.cuh"
#include "kernels/mergesort.cuh"

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
    unsigned int ib = (unsigned int)(f.x * n_grid) % n_grid;
    unsigned int jb = (unsigned int)(f.y * n_grid) % n_grid;
    unsigned int kb = (unsigned int)(f.z * n_grid) % n_grid;

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
        const Scalar *d_net_virial,
        Scalar *d_net_virial_alt,
        const Scalar4 *d_net_force,
        Scalar4 *d_net_force_alt,
        const Scalar4 *d_net_torque,
        Scalar4 *d_net_torque_alt,
        unsigned int *d_rtag)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    unsigned int old_idx = d_sorted_order[idx];

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
    d_net_virial_alt[idx] = d_net_virial[old_idx];
    d_net_force_alt[idx] = d_net_force[old_idx];
    d_net_torque_alt[idx] = d_net_torque[old_idx];

    // update rtag to point to particle position in new arrays
    d_rtag[tag] = idx;
    }

void gpu_apply_sorted_order(
        unsigned int N,
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
        const Scalar *d_net_virial,
        Scalar *d_net_virial_alt,
        const Scalar4 *d_net_force,
        Scalar4 *d_net_force_alt,
        const Scalar4 *d_net_torque,
        Scalar4 *d_net_torque_alt,
        unsigned int *d_rtag
        )
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;

    gpu_apply_sorted_order_kernel<<<n_blocks, block_size>>>(N,
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
        d_net_virial,
        d_net_virial_alt,
        d_net_force,
        d_net_force_alt,
        d_net_torque,
        d_net_torque_alt,
        d_rtag);
    }
