// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MeshVolumeConservationGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file MeshVolumeConservationGPU.cu
    \brief Defines GPU kernel code for calculating the volume_constraint forces. Used by
   MeshVolumeConservationComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for calculating volume_constraint sigmas on the GPU
/*! \param d_sigma Device memory to write per paricle sigma
    \param d_sigma_dash Device memory to write per particle sigma_dash
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param blist List of mesh bonds stored on the GPU
    \param d_triangles device array of mesh triangles
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
*/
__global__ void gpu_compute_volume_constraint_volume_kernel(Scalar volume,
                                                            const unsigned int N,
                                                            const Scalar4* d_pos,
                                                            const unsigned int* d_rtag,
                                                            const BoxDim& box,
                                                            const group_storage<6>* tlist,
                                                            const unsigned int* tpos_list,
                                                            const Index2D tlist_idx,
                                                            const unsigned int* n_triangles_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds = n_bonds_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // initialize the force to 0
    Scalar3 sigma_dash = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar sigma = 0.0;

    // loop over all angles
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        group_storage<4> cur_bond = blist[blist_idx(idx, bond_idx)];

        int cur_bond_idx = cur_bond.idx[0];
        int cur_tr1_idx = cur_bond.idx[1];
        int cur_tr2_idx = cur_bond.idx[2];

        if (cur_tr1_idx == cur_tr2_idx)
            continue;

        const group_storage<6>& triangle1 = d_triangles[cur_tr1_idx];

        unsigned int cur_idx_c = d_rtag[triangle1.tag[0]];

        unsigned int iterator = 1;
        while (idx == cur_idx_c || cur_bond_idx == cur_idx_c)
            {
            cur_idx_c = d_rtag[triangle1.tag[iterator]];
            iterator++;
            }

        const group_storage<6>& triangle2 = d_triangles[cur_tr2_idx];

        unsigned int cur_idx_d = d_rtag[triangle2.tag[0]];

        iterator = 1;
        while (idx == cur_idx_d || cur_bond_idx == cur_idx_d)
            {
            cur_idx_d = d_rtag[triangle2.tag[iterator]];
            iterator++;
            }

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_bond_idx];
        Scalar3 bb_pos = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_idx_c];
        Scalar3 cc_pos = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 dd_postype = d_pos[cur_idx_d];
        Scalar3 dd_pos = make_scalar3(dd_postype.x, dd_postype.y, dd_postype.z);

        Scalar3 dab = pos - bb_pos;
        Scalar3 dac = pos - cc_pos;
        Scalar3 dad = pos - dd_pos;
        Scalar3 dbc = bb_pos - cc_pos;
        Scalar3 dbd = bb_pos - dd_pos;

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        rac = sqrt(rac);
        Scalar rad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
        rad = sqrt(rad);

        Scalar rbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
        rbc = sqrt(rbc);
        Scalar rbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
        rbd = sqrt(rbd);

        Scalar3 nab, nac, nad, nbc, nbd;
        nab = dab / rab;
        nac = dac / rac;
        nad = dad / rad;
        nbc = dbc / rbc;
        nbd = dbd / rbd;

        Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;

        if (c_accb > 1.0)
            c_accb = 1.0;
        if (c_accb < -1.0)
            c_accb = -1.0;

        Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;

        if (c_addb > 1.0)
            c_addb = 1.0;
        if (c_addb < -1.0)
            c_addb = -1.0;

        vec3<Scalar> nbac
            = cross(vec3<Scalar>(nab.x, nab.y, nab.z), vec3<Scalar>(nac.x, nac.y, nac.z));

        Scalar inv_nbac = 1.0 / sqrt(dot(nbac, nbac));

        vec3<Scalar> nbad
            = cross(vec3<Scalar>(nab.x, nab.y, nab.z), vec3<Scalar>(nad.x, nad.y, nad.z));

        Scalar inv_nbad = 1.0 / sqrt(dot(nbad, nbad));

        if (dot(nbac, nbad) * inv_nbad * inv_nbac > 0.9)
            {
            this->m_exec_conf->msg->error() << "volume_constraint calculations : triangles "
                                            << tr_idx1 << " " << tr_idx2 << " overlap." << std::endl
                                            << std::endl;
            throw std::runtime_error("Error in bending energy calculation");
            }

        Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
        if (inv_s_accb < SMALL)
            inv_s_accb = SMALL;
        inv_s_accb = 1.0 / inv_s_accb;

        Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
        if (inv_s_addb < SMALL)
            inv_s_addb = SMALL;
        inv_s_addb = 1.0 / inv_s_addb;

        Scalar cot_accb = c_accb * inv_s_accb;
        Scalar cot_addb = c_addb * inv_s_addb;

        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;

        Scalar sigma_a = sigma_hat_ab * rsqab * 0.25;

        Scalar3 sigma_dash_a = sigma_hat_ab * dab;

        sigma += sigma_a;
        sigma_dash += sigma_dash_a;
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_sigma[idx] = sigma;
    d_sigma_dash[idx] = sigma_dash;
    }

/*! \param d_sigma Device memory to write per paricle sigma
    \param d_sigma_dash Device memory to write per particle sigma_dash
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param blist List of mesh bonds stored on the GPU
    \param d_triangles device array of mesh triangles
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_volume_constraint_volume(Scalar volume,
                                                const unsigned int N,
                                                const Scalar4* d_pos,
                                                const int3* d_image,
                                                const BoxDim& box,
                                                const group_storage<6>* tlist,
                                                const unsigned int* tpos_list,
                                                const Index2D tlist_idx,
                                                const unsigned int* n_triangles_list,
                                                int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_volume_constraint_volume_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_volume_constraint_volume_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       volume,
                       N,
                       d_pos,
                       d_image,
                       box,
                       tlist,
                       tpos_list,
                       tlist_idx,
                       n_triangles_list);

    return hipSuccess;
    }

//! Kernel for calculating volume_constraint sigmas on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param d_sigma Device memory to write per paricle sigma
    \param d_sigma_dash Device memory to write per particle sigma_dash
    \param blist List of mesh bonds stored on the GPU
    \param d_triangles device array of mesh triangles
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param d_params K params packed as Scalar variables
    \param n_bond_type number of mesh bond types
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
*/
__global__ void gpu_compute_volume_constraint_force_kernel(Scalar4* d_force,
                                                           Scalar* d_virial,
                                                           const size_t virial_pitch,
                                                           const unsigned int N,
                                                           const Scalar4* d_pos,
                                                           const int3* d_image,
                                                           const BoxDim& box,
                                                           const Scalar volume,
                                                           const group_storage<6>* tlist,
                                                           const unsigned int* tpos_list,
                                                           const Index2D tlist_idx,
                                                           const unsigned int* n_triangles_list,
                                                           Scalar* d_params,
                                                           const unsigned int n_triangle_type,
                                                           unsigned int* d_flags);
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds = n_bonds_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    int3 image_a = __ldg(d_image + idx);

    vec3<Scalar> pos_a = box.shift(pos, image_a);

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // loop over all angles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<6> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_b = cur_triangle.idx[0];
        int cur_triangle_c = cur_triangle.idx[1];
        int cur_triangle_type = cur_triangle.idx[5];

        // get the angle parameters (MEM TRANSFER: 8 bytes)
        Scalar2 params = __ldg(d_params + cur_triangle_type);
        Scalar K = params.x;
        Scalar V0 = params.y;

        Scalar VolDiff = volume - V0;

        Scalar energy = K * VolDiff * VolDiff / (2 * V0 * N);

        VolDiff = -K / V0 * VolDiff / 6.0;

        int cur_triangle_abc = tpos_list[tlist_idx(idx, triangle_idx)];

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_triangle_b];
        Scalar3 bb_pos = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
        int3 image_b = d_image[cur_triangle_b] vec3<Scalar> pos_b = box.shift(bb_pos, image_b);

        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_triangle_c];
        Scalar3 cc_pos = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);
        int3 image_c = d_image[cur_triangle_c] vec3<Scalar> pos_c = box.shift(cc_pos, image_c);

        vec3<Scalar> dVol;
        if (cur_triangle_abc == 1)
            {
            dVol = cross(pos_b, pos_c);
            }
        else
            {
            dVol = cross(pos_c, pos_b);
            }

        Scalar3 Fa;

        Fa.x = VolDiff * dVol.x;
        Fa.y = VolDiff * dVol.y;
        Fa.z = VolDiff * dVol.z;

        force.x += Fa.x;
        force.y += Fa.y;
        force.z += Fa.z;
        force.w = energy;

        virial[0] += Scalar(1. / 2.) * pos.x * Fa.x; // xx
        virial[1] += Scalar(1. / 2.) * pos.y * Fa.x; // xy
        virial[2] += Scalar(1. / 2.) * pos.z * Fa.x; // xz
        virial[3] += Scalar(1. / 2.) * pos.y * Fa.y; // yy
        virial[4] += Scalar(1. / 2.) * pos.z * Fa.y; // yz
        virial[5] += Scalar(1. / 2.) * pos.z * Fa.z; // zz
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param d_sigma Device memory to write per paricle sigma
    \param d_sigma_dash Device memory to write per particle sigma_dash
    \param blist List of mesh bonds stored on the GPU
    \param d_triangles device array of mesh triangles
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param d_params K params packed as Scalar variables
    \param n_bond_type number of mesh bond types
    \param block_size Block size to use when performing calculations
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_volume_constraint_force(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const Scalar4* d_pos,
                                               const int3* d_image,
                                               const unsigned int* d_rtag,
                                               const BoxDim& box,
                                               const Scalar volume,
                                               const group_storage<6>* tlist,
                                               const unsigned int* tpos_list,
                                               const Index2D tlist_idx,
                                               const unsigned int* n_triangles_list,
                                               Scalar* d_params,
                                               const unsigned int n_triangle_type,
                                               int block_size,
                                               unsigned int* d_flags);
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_volume_constraint_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_volume_constraint_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_image,
                       box,
                       volume,
                       tlist,
                       tpos_list,
                       tlist_idx,
                       n_triangles_list,
                       d_params,
                       n_triangle_type,
                       d_flags);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
