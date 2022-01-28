#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "AreaConservationMeshForceComputeGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file AreaConservationMeshForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating the area conservation forces. Used by
   AreaConservationMeshForceComputeComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel for calculating area conservation force on the GPU
/*! 
    \param d_area Device memory to write total surface area
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params K,A0 params packed as Scalar variables
    \param n_triangle_type number of mesh triangle types
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
*/
__global__ void gpu_compute_AreaConservation_force_kernel(Scalar4* d_force,
                                                          Scalar* d_virial,
                                                          const size_t virial_pitch,
                                                          const unsigned int N,
                                                          const Scalar4* d_pos,
                                                          BoxDim box,
                                                          const group_storage<6>* tlist,
                                                          const Index2D tlist_idx,
                                                          const unsigned int* n_triangles_list,
                                                          Scalar* d_params,
                                                          const unsigned int n_triangle_type,
                                                          unsigned int* d_flags)
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
    
    Scalar area = 0.0;
    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<6> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        // int cur_mem1_idx = cur_triangle.idx[0];
        int cur_mem2_idx = cur_triangle.idx[0];
        int cur_mem3_idx = cur_triangle.idx[1];

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_mem2_idx];
        Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_mem3_idx];
        Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);

        Scalar3 dab = pos_b - pos_a;
        Scalar3 dac = pos_c - pos_a;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

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

        Scalar2 params = __ldg(d_params); // Todo: + cur_angle_type
        Scalar K = params.x;
        Scalar A0 = params.y;
        Scalar At = A0 / N;

        Scalar3 dc_dra, dc_drb, dc_drc; // dcos_baac / dr_a 
        dc_dra = - nac / rab - nab /rac + c_baac / rab * nab + c_baac / rac * nac;
        dc_drb = nac / rab - c_baac / rab * nab;
        dc_drc = nac / rac - c_baac / rac * nac;

        Scalar3 ds_dra, ds_drb, ds_drc; // dsin_baac / dr_a 
        ds_dra = - 1.0 * c_baac * inv_s_baac * dc_dra;
        ds_drb = - 1.0 * c_baac * inv_s_baac * dc_drb;
        ds_drc = - 1.0 * c_baac * inv_s_baac * dc_drc;

        Scalar numerator_base;
        area += rab * rac * s_baac / 2;
        numerator_base = rab * rac * s_baac / 2 - At;

        Scalar3 Fa = - K / (2 * At) * numerator_base * (-nab * rac * s_baac - nac * rab * s_baac + ds_dra * rab * rac);

        force.x += Fa.x;
        force.y += Fa.y;
        force.z += Fa.z;
        force.w += K/(6.0*At)*numerator_base*numerator_base; // divided by 3 because of three 
                                                            // particles sharing the energy

        virial[0] += Scalar(1. / 2.) * pos_a.x * Fa.x; // xx
        virial[1] += Scalar(1. / 2.) * pos_a.y * Fa.x; // xy
        virial[2] += Scalar(1. / 2.) * pos_a.z * Fa.x; // xz
        virial[3] += Scalar(1. / 2.) * pos_a.y * Fa.y; // yy
        virial[4] += Scalar(1. / 2.) * pos_a.z * Fa.y; // yz
        virial[5] += Scalar(1. / 2.) * pos_a.z * Fa.z; // zz
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;
    d_area[idx] = area;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

/*!
    \param d_area Device memory to write total surface area
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params K, A0 params packed as Scalar variables
    \param n_triangle_type number of mesh triangle types
    \param block_size Block size to use when performing calculations
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
    \param compute_capability Device compute capability (200, 300, 350, ...)
    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_AreaConservation_force(Scalar4* d_force,
                                              Scalar* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const Scalar4* d_pos,
                                              const BoxDim& box,
                                              const group_storage<6>* tlist,
                                              const Index2D tlist_idx,
                                              const unsigned int* n_triangles_list,
                                              Scalar* d_params,
                                              const unsigned int n_triangle_type,
                                              int block_size,
                                              unsigned int* d_flags)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_AreaConservation_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_AreaConservation_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_area,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       box,
                       tlist,
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
