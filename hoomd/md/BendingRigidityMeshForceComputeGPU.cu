// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BendingRigidityMeshForceComputeGPU.cuh"
#include "hip/hip_runtime.h"
#include "hoomd/TextureTools.h"

/*! \file BendingRigidityMeshForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating the bending rigidity forces. Used by
   BendingRigidityMeshForceComputeComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel for calculating helfrich sigmas on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param blist List of mesh bonds stored on the GPU
    \param bpos_list Position of current index in list of mesh bonds stored on the GPU
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param d_params K params packed as Scalar variables
    \param n_bond_type number of mesh bond types
*/
__global__ void gpu_compute_bending_rigidity_force_kernel(Scalar4* d_force,
                                                          Scalar* d_virial,
                                                          const size_t virial_pitch,
                                                          const unsigned int N,
                                                          const Scalar4* d_pos,
                                                          const unsigned int* d_rtag,
                                                          BoxDim box,
                                                          const group_storage<4>* blist,
                                                          const Index2D blist_idx,
                                                          const unsigned int* bpos_list,
                                                          const unsigned int* n_bonds_list,
                                                          Scalar* d_params,
                                                          const unsigned int n_bond_type)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int n_bonds = n_bonds_list[idx];

    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // loop over all bonds
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        group_storage<4> cur_bond = blist[blist_idx(idx, bond_idx)];

        unsigned int cur_idx_a, cur_idx_b, cur_idx_c, cur_idx_d;

        int cur_bond_pos = bpos_list[blist_idx(idx, bond_idx)];

        if (cur_bond_pos < 2)
            {
            cur_idx_a = idx;
            cur_idx_b = cur_bond.idx[0];
            cur_idx_c = cur_bond.idx[1];
            cur_idx_d = cur_bond.idx[2];
            }
        else
            {
            cur_idx_a = cur_bond.idx[0];
            cur_idx_b = cur_bond.idx[1];
            cur_idx_c = idx;
            cur_idx_d = cur_bond.idx[2];
            }

        int cur_bond_type = cur_bond.idx[3];

        if (cur_idx_c == cur_idx_d)
            continue;

        Scalar K = __ldg(d_params + cur_bond_type);

        Scalar4 bb_postype = d_pos[cur_idx_b];
        Scalar3 bb_pos = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
        Scalar4 dd_postype = d_pos[cur_idx_d];
        Scalar3 dd_pos = make_scalar3(dd_postype.x, dd_postype.y, dd_postype.z);
        Scalar3 aa_pos, cc_pos;

        if (cur_bond_pos < 2)
            {
            aa_pos = pos;
            Scalar4 cc_postype = d_pos[cur_idx_c];
            cc_pos = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);
            }
        else
            {
            cc_pos = pos;
            Scalar4 aa_postype = d_pos[cur_idx_a];
            aa_pos = make_scalar3(aa_postype.x, aa_postype.y, aa_postype.z);
            }

        Scalar3 dab = aa_pos - bb_pos;
        Scalar3 dac = aa_pos - cc_pos;
        Scalar3 dad = aa_pos - dd_pos;
        Scalar3 dbc = bb_pos - cc_pos;
        Scalar3 dbd = bb_pos - dd_pos;

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);

        Scalar3 z1;
        z1.x = dab.y * dac.z - dab.z * dac.y;
        z1.y = dab.z * dac.x - dab.x * dac.z;
        z1.z = dab.x * dac.y - dab.y * dac.x;

        Scalar3 z2;
        z2.x = dad.y * dab.z - dad.z * dab.y;
        z2.y = dad.z * dab.x - dad.x * dab.z;
        z2.z = dad.x * dab.y - dad.y * dab.x;

        Scalar n1 = fast::rsqrt(z1.x * z1.x + z1.y * z1.y + z1.z * z1.z);
        Scalar n2 = fast::rsqrt(z2.x * z2.x + z2.y * z2.y + z2.z * z2.z);
        Scalar z1z2 = z1.x * z2.x + z1.y * z2.y + z1.z * z2.z;

        Scalar cosinus = z1z2 * n1 * n2;

        Scalar3 A1 = n1 * n2 * z2 - cosinus * n1 * n1 * z1;
        Scalar3 A2 = n1 * n2 * z1 - cosinus * n2 * n2 * z2;

        Scalar3 Fac;

        Fac.x = A1.y * dab.z - A1.z * dab.y;
        Fac.y = -A1.x * dab.z + A1.z * dab.x;
        Fac.z = A1.x * dab.y - A1.y * dab.x;

        Fac *= 0.5 * K;

        virial[0] += Scalar(1. / 2.) * dac.x * Fac.x; // xx

        virial[1] += Scalar(1. / 2.) * dac.y * Fac.x; // xy
        virial[2] += Scalar(1. / 2.) * dac.z * Fac.x; // xz
        virial[3] += Scalar(1. / 2.) * dac.y * Fac.y; // yy
        virial[4] += Scalar(1. / 2.) * dac.z * Fac.y; // yz
        virial[5] += Scalar(1. / 2.) * dac.z * Fac.z; // zz

        if (cur_bond_pos < 2)
            {
            Scalar3 Fab, Fad;
            Fab.x = -A1.y * dac.z + A1.z * dac.y + A2.y * dad.z - A2.z * dad.y;
            Fab.y = A1.x * dac.z - A1.z * dac.x - A2.x * dad.z + A2.z * dad.x;
            Fab.z = -A1.x * dac.y + A1.y * dac.x + A2.x * dad.y - A2.y * dad.x;

            Fad.x = -A2.y * dab.z + A2.z * dab.y;
            Fad.y = A2.x * dab.z - A2.z * dab.x;
            Fad.z = -A2.x * dab.y + A2.y * dab.x;

            Fab *= 0.5 * K;
            Fad *= 0.5 * K;

            virial[0] += Scalar(1. / 2.) * (dab.x * Fab.x + dad.x * Fad.x); // xx
            virial[1] += Scalar(1. / 2.) * (dab.y * Fab.x + dad.y * Fad.x); // xy
            virial[2] += Scalar(1. / 2.) * (dab.z * Fab.x + dad.z * Fad.x); // xz
            virial[3] += Scalar(1. / 2.) * (dab.y * Fab.y + dad.y * Fad.y); // yy
            virial[4] += Scalar(1. / 2.) * (dab.z * Fab.y + dad.z * Fad.y); // yz
            virial[5] += Scalar(1. / 2.) * (dab.z * Fab.z + dad.z * Fad.z); // zz

            Fac += (Fab + Fad);
            }
        else
            {
            Fac *= -1;
            }

        force.x += Fac.x;
        force.y += Fac.y;
        force.z += Fac.z;
        force.w += K / 8.0 * (1 - cosinus);
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
    \param blist List of mesh bonds stored on the GPU
    \param bpos_list Position of current index in list of mesh bonds stored on the GPU
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param d_params K params packed as Scalar variables
    \param n_bond_type number of mesh bond types
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_bending_rigidity_force(Scalar4* d_force,
                                              Scalar* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const Scalar4* d_pos,
                                              const unsigned int* d_rtag,
                                              const BoxDim& box,
                                              const group_storage<4>* blist,
                                              const Index2D blist_idx,
                                              const unsigned int* bpos_list,
                                              const unsigned int* n_bonds_list,
                                              Scalar* d_params,
                                              const unsigned int n_bond_type,
                                              int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_bending_rigidity_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_bending_rigidity_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_rtag,
                       box,
                       blist,
                       blist_idx,
                       bpos_list,
                       n_bonds_list,
                       d_params,
                       n_bond_type);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
