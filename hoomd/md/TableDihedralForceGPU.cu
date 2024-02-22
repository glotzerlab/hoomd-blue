// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TableDihedralForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/VectorMath.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL 0.001f

/*! \file TableDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating the table dihedral forces. Used by
   TableDihedralForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
/*!  This kernel is called to calculate the table dihedral forces on all triples this is defined or

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param device_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param dlist List of dihedrals stored on the GPU
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals stored on the GPU
    \param n_dihedral_type number of dihedral types
    \param d_tables Tables of the potential and force
    \param table_value index helper function
    \param delta_phi dihedral delta of the table

    See TableDihedralForceCompute for information on the memory layout.
*/
__global__ void gpu_compute_table_dihedral_forces_kernel(Scalar4* d_force,
                                                         Scalar* d_virial,
                                                         const size_t virial_pitch,
                                                         const unsigned int N,
                                                         const Scalar4* device_pos,
                                                         const BoxDim box,
                                                         const group_storage<4>* dlist,
                                                         const unsigned int* dihedral_ABCD,
                                                         const unsigned int pitch,
                                                         const unsigned int* n_dihedrals_list,
                                                         const Scalar2* d_tables,
                                                         const Index2D table_value,
                                                         const Scalar delta_phi)
    {
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals = n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = device_pos[idx]; // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a, pos_b, pos_c,
        pos_d; // allocate space for the a,b,c, and d atom in the a-b-c-d set

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);

    // initialize the virial tensor to 0
    Scalar virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = 0;

    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = dlist[pitch * dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch * dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];
        int cur_dihedral_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = device_pos[cur_dihedral_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = device_pos[cur_dihedral_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);
        // get the d-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 z_postype = device_pos[cur_dihedral_z_idx];
        Scalar3 z_pos = make_scalar3(z_postype.x, z_postype.y, z_postype.z);

        if (cur_dihedral_abcd == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 3)
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = pos_a - pos_b;
        Scalar3 dcb = pos_c - pos_b;
        Scalar3 ddc = pos_d - pos_c;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        Scalar3 dcbm = -dcb;
        dcbm = box.minImage(dcbm);

        // c0 calculation
        Scalar sb1 = Scalar(1.0) / (dab.x * dab.x + dab.y * dab.y + dab.z * dab.z);
        Scalar sb3 = Scalar(1.0) / (ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z);

        Scalar rb1 = fast::sqrt(sb1);
        Scalar rb3 = fast::sqrt(sb3);

        Scalar c0 = (dab.x * ddc.x + dab.y * ddc.y + dab.z * ddc.z) * rb1 * rb3;

        // 1st and 2nd angle

        Scalar b1mag2 = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar b1mag = fast::sqrt(b1mag2);
        Scalar b2mag2 = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z;
        Scalar b2mag = fast::sqrt(b2mag2);
        Scalar b3mag2 = ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z;
        Scalar b3mag = fast::sqrt(b3mag2);

        Scalar ctmp = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z;
        Scalar r12c1 = Scalar(1.0) / (b1mag * b2mag);
        Scalar c1mag = ctmp * r12c1;

        ctmp = dcbm.x * ddc.x + dcbm.y * ddc.y + dcbm.z * ddc.z;
        Scalar r12c2 = Scalar(1.0) / (b2mag * b3mag);
        Scalar c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        Scalar sin2 = Scalar(1.0) - c1mag * c1mag;
        if (sin2 < 0.0f)
            sin2 = 0.0f;
        Scalar sc1 = fast::sqrt(sin2);
        if (sc1 < SMALL)
            sc1 = SMALL;
        sc1 = Scalar(1.0) / sc1;

        sin2 = Scalar(1.0) - c2mag * c2mag;
        if (sin2 < 0.0f)
            sin2 = 0.0f;
        Scalar sc2 = fast::sqrt(sin2);
        if (sc2 < SMALL)
            sc2 = SMALL;
        sc2 = Scalar(1.0) / sc2;

        Scalar s12 = sc1 * sc2;
        Scalar c = (c0 + c1mag * c2mag) * s12;

        if (c > Scalar(1.0))
            c = Scalar(1.0);
        if (c < -Scalar(1.0))
            c = -Scalar(1.0);

        // determinant
        Scalar det = dot(dab,
                         make_scalar3(ddc.y * dcb.z - ddc.z * dcb.y,
                                      ddc.z * dcb.x - ddc.x * dcb.z,
                                      ddc.x * dcb.y - ddc.y * dcb.x));

        // phi
        Scalar phi = acosf(c);

        if (det < 0)
            phi = -phi;

        // precomputed term
        Scalar value_f = (Scalar(M_PI) + phi) / delta_phi;

        // compute index into the table and read in values
        unsigned int value_i = value_f;
        Scalar2 VT0 = __ldg(d_tables + table_value(value_i, cur_dihedral_type));
        Scalar2 VT1 = __ldg(d_tables + table_value(value_i + 1, cur_dihedral_type));
        // unpack the data
        Scalar V0 = VT0.x;
        Scalar V1 = VT1.x;
        Scalar T0 = VT0.y;
        Scalar T1 = VT1.y;

        // compute the linear interpolation coefficient
        Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and T;
        Scalar V = V0 + f * (V1 - V0);
        Scalar T = T0 + f * (T1 - T0);

        // from Blondel and Karplus 1995
        vec3<Scalar> A = cross(vec3<Scalar>(dab), vec3<Scalar>(dcbm));
        Scalar Asq = dot(A, A);

        vec3<Scalar> B = cross(vec3<Scalar>(ddc), vec3<Scalar>(dcbm));
        Scalar Bsq = dot(B, B);

        Scalar3 f_a = -T * vec_to_scalar3(b2mag / Asq * A);
        Scalar3 f_b
            = -f_a
              + T / b2mag * vec_to_scalar3(dot(dab, dcbm) / Asq * A - dot(ddc, dcbm) / Bsq * B);
        Scalar3 f_c = T
                      * vec_to_scalar3(dot(ddc, dcbm) / Bsq / b2mag * B
                                       - dot(dab, dcbm) / Asq / b2mag * A - b2mag / Bsq * B);
        Scalar3 f_d = T * b2mag / Bsq * vec_to_scalar3(B);

        // Now, apply the force to each individual atom a,b,c,d
        // and accumulate the energy/virial

        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        Scalar dihedral_eng = V * Scalar(1.0 / 4.0);

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        Scalar dihedral_virial[6];

        dihedral_virial[0] = (1. / 4.) * (dab.x * f_a.x + dcb.x * f_c.x + (ddc.x + dcb.x) * f_d.x);
        dihedral_virial[1] = (1. / 4.) * (dab.y * f_a.x + dcb.y * f_c.x + (ddc.y + dcb.y) * f_d.x);
        dihedral_virial[2] = (1. / 4.) * (dab.z * f_a.x + dcb.z * f_c.x + (ddc.z + dcb.z) * f_d.x);
        dihedral_virial[3] = (1. / 4.) * (dab.y * f_a.y + dcb.y * f_c.y + (ddc.y + dcb.y) * f_d.y);
        dihedral_virial[4] = (1. / 4.) * (dab.z * f_a.y + dcb.z * f_c.y + (ddc.z + dcb.z) * f_d.y);
        dihedral_virial[5] = (1. / 4.) * (dab.z * f_a.z + dcb.z * f_c.z + (ddc.z + dcb.z) * f_d.z);

        if (cur_dihedral_abcd == 0)
            {
            force_idx.x += f_a.x;
            force_idx.y += f_a.y;
            force_idx.z += f_a.z;
            }
        if (cur_dihedral_abcd == 1)
            {
            force_idx.x += f_b.x;
            force_idx.y += f_b.y;
            force_idx.z += f_b.z;
            }
        if (cur_dihedral_abcd == 2)
            {
            force_idx.x += f_c.x;
            force_idx.y += f_c.y;
            force_idx.z += f_c.z;
            }
        if (cur_dihedral_abcd == 3)
            {
            force_idx.x += f_d.x;
            force_idx.y += f_d.y;
            force_idx.z += f_d.z;
            }

        force_idx.w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            virial_idx[k] += dihedral_virial[k];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int k = 0; k < 6; k++)
        d_virial[k * virial_pitch + idx] = virial_idx[k];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param device_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param dlist List of dihedrals stored on the GPU
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals stored on the GPU
    \param n_dihedral_type number of dihedral types
    \param d_tables Tables of the potential and force
    \param table_width Number of points in each table
    \param table_value indexer helper
    \param block_size Block size at which to run the kernel
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \note This is just a kernel driver. See gpu_compute_table_dihedral_forces_kernel for full
   documentation.
*/
hipError_t gpu_compute_table_dihedral_forces(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* device_pos,
                                             const BoxDim& box,
                                             const group_storage<4>* dlist,
                                             const unsigned int* dihedral_ABCD,
                                             const unsigned int pitch,
                                             const unsigned int* n_dihedrals_list,
                                             const Scalar2* d_tables,
                                             const unsigned int table_width,
                                             const Index2D& table_value,
                                             const unsigned int block_size)
    {
    assert(d_tables);
    assert(table_width > 1);

    if (N == 0)
        return hipSuccess;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_table_dihedral_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    Scalar delta_phi = Scalar(2.0 * M_PI) / (Scalar)(table_width - 1);

    hipLaunchKernelGGL((gpu_compute_table_dihedral_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       device_pos,
                       box,
                       dlist,
                       dihedral_ABCD,
                       pitch,
                       n_dihedrals_list,
                       d_tables,
                       table_value,
                       delta_phi);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
