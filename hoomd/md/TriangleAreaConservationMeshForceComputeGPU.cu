// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TriangleAreaConservationMeshForceComputeGPU.cuh"
#include "hip/hip_runtime.h"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file TriangleAreaConservationMeshForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating the triangle area conservation forces. Used by
   TriangleAreaConservationMeshForceComputeComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel for calculating area conservation force on the GPU
/*!
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangle indices stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params K,A0 params packed as Scalar variables
    \param n_triangle_type number of mesh triangle types
*/
__global__ void
gpu_compute_TriangleAreaConservation_force_kernel(Scalar4* d_force,
                                                  Scalar* d_virial,
                                                  const size_t virial_pitch,
                                                  const unsigned int N,
                                                  const Scalar4* d_pos,
                                                  BoxDim box,
                                                  const group_storage<3>* tlist,
                                                  const unsigned int* tpos_list,
                                                  const Index2D tlist_idx,
                                                  const unsigned int* n_triangles_list,
                                                  Scalar2* d_params,
                                                  const unsigned int n_triangle_type)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_triangles = n_triangles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // Scalar area = 0.0;
    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_abc = tpos_list[tlist_idx(idx, triangle_idx)];

        int cur_mem2_idx = cur_triangle.idx[0];
        int cur_mem3_idx = cur_triangle.idx[1];
        int cur_triangle_type = cur_triangle.idx[2];

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_mem2_idx];
        Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);

        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_mem3_idx];
        Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);

        Scalar3 dab, dac;

        if (cur_triangle_abc == 0)
            {
            dab = pos_a - pos_b;
            dac = pos_a - pos_c;
            }
        else
            {
            dab = pos_b - pos_a;
            dac = pos_b - pos_c;
            }

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        rab = sqrt(rab);
        Scalar rac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        rac = sqrt(rac);

        Scalar3 nab, nac;
        nab = dab / rab;
        nac = dac / rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        Scalar inv_s_baac = 1.0 / s_baac;

        Scalar2 params = __ldg(d_params + cur_triangle_type);
        Scalar K = params.x;
        Scalar At = params.y;

        Scalar3 dc_drab = -nac / rab + c_baac / rab * nab;
        Scalar3 ds_drab = -c_baac * inv_s_baac * dc_drab;

        Scalar tri_area = rab * rac * s_baac / 6; // triangle area/3
        Scalar Ut = 3 * tri_area - At;

        Scalar prefactor = K / (2 * At) * Ut;

        Scalar3 Fab = prefactor * (-nab * rac * s_baac + ds_drab * rab * rac);

        if (cur_triangle_abc == 0)
            {
            Scalar3 dc_drac = -nab / rac + c_baac / rac * nac;
            Scalar3 ds_drac = -c_baac * inv_s_baac * dc_drac;
            Scalar3 Fac = prefactor * (-nac * rab * s_baac + ds_drac * rab * rac);

            force.x += (Fab.x + Fac.x);
            force.y += (Fab.y + Fac.y);
            force.z += (Fab.z + Fac.z);

            virial[0] += Scalar(1. / 2.) * (dab.x * Fab.x + dac.x * Fac.x); // xx
            virial[1] += Scalar(1. / 2.) * (dab.y * Fab.x + dac.y * Fac.x); // xy
            virial[2] += Scalar(1. / 2.) * (dab.z * Fab.x + dac.z * Fac.x); // xz
            virial[3] += Scalar(1. / 2.) * (dab.y * Fab.y + dac.y * Fac.y); // yy
            virial[4] += Scalar(1. / 2.) * (dab.z * Fab.y + dac.z * Fac.y); // yz
            virial[5] += Scalar(1. / 2.) * (dab.z * Fab.z + dac.z * Fac.z); // zz
            }
        else
            {
            force.x -= Fab.x;
            force.y -= Fab.y;
            force.z -= Fab.z;

            virial[0] += Scalar(1. / 2.) * dab.x * Fab.x; // xx
            virial[1] += Scalar(1. / 2.) * dab.y * Fab.x; // xy
            virial[2] += Scalar(1. / 2.) * dab.z * Fab.x; // xz
            virial[3] += Scalar(1. / 2.) * dab.y * Fab.y; // yy
            virial[4] += Scalar(1. / 2.) * dab.z * Fab.y; // yz
            virial[5] += Scalar(1. / 2.) * dab.z * Fab.z; // zz
            }

        force.w += K / (6.0 * At) * Ut * Ut; // divided by 3 because of three
                                             // particles sharing the energy
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

/*!
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangles stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params K, A0 params packed as Scalar variables
    \param n_triangle_type number of mesh triangle types
    \param block_size Block size to use when performing calculations
    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_TriangleAreaConservation_force(Scalar4* d_force,
                                                      Scalar* d_virial,
                                                      const size_t virial_pitch,
                                                      const unsigned int N,
                                                      const Scalar4* d_pos,
                                                      const BoxDim& box,
                                                      const group_storage<3>* tlist,
                                                      const unsigned int* tpos_list,
                                                      const Index2D tlist_idx,
                                                      const unsigned int* n_triangles_list,
                                                      Scalar2* d_params,
                                                      const unsigned int n_triangle_type,
                                                      int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_TriangleAreaConservation_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_TriangleAreaConservation_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       box,
                       tlist,
                       tpos_list,
                       tlist_idx,
                       n_triangles_list,
                       d_params,
                       n_triangle_type);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
