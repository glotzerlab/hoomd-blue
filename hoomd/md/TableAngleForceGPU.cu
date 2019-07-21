// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

#include "TableAngleForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL 0.001f

/*! \file TableAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the table angle forces. Used by TableAngleForceComputeGPU.
*/

/*!  This kernel is called to calculate the table angle forces on all triples this is defined or

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param alist List of angles stored on the GPU
    \param apos_list List of particle position in angle stored on the GPU
    \param pitch Pitch of 2D angle list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param n_angle_type number of angle types
    \param d_tables Tables of the potential and force
    \param table_value index helper function
    \param delta_th angle delta of the table

    See TableAngleForceCompute for information on the memory layout.
*/
__global__ void gpu_compute_table_angle_forces_kernel(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim box,
                                     const group_storage<3> *alist,
                                     const unsigned int *apos_list,
                                     const unsigned int pitch,
                                     const unsigned int *n_angles_list,
                                     const Scalar2 *d_tables,
                                     const Index2D table_value,
                                     const Scalar delta_th)
    {


    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles =n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet


    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(0.0, 0.0, 0.0, 0.0);

    Scalar fab[3], fcb[3];

    // initialize the virial tensor to 0
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];
        int cur_angle_type = cur_angle.idx[2];
        int cur_angle_abc = apos_list[pitch*angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = a_pos - b_pos;
        Scalar3 dcb = c_pos - b_pos;
        Scalar3 dac = a_pos - c_pos;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        Scalar rsqab = dot(dab, dab);
        Scalar rab = fast::sqrt(rsqab);
        Scalar rsqcb = dot(dcb, dcb);
        Scalar rcb = fast::sqrt(rsqcb);

        Scalar c_abbc = dot(dab, dcb);
        c_abbc /= rab*rcb;

        if (c_abbc > Scalar(1.0)) c_abbc = Scalar(1.0);
        if (c_abbc < -Scalar(1.0)) c_abbc = -Scalar(1.0);

        Scalar s_abbc = fast::sqrt(Scalar(1.0)- c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = Scalar(1.0)/s_abbc;

        // actually calculate the force
        Scalar theta = acosf(c_abbc);


        // precomputed term
        Scalar value_f = theta / delta_th;

        // compute index into the table and read in values
        unsigned int value_i = value_f;
        Scalar2 VT0 = __ldg(d_tables + table_value(value_i, cur_angle_type));
        Scalar2 VT1 = __ldg(d_tables + table_value(value_i+1, cur_angle_type));
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


        Scalar a = T * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;

        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;

        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = V*Scalar(1.0/3.0);

        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1./3.) * ( dab.x*fab[0] + dcb.x*fcb[0] );
        angle_virial[1] = Scalar(1./3.) * ( dab.y*fab[0] + dcb.y*fcb[0] );
        angle_virial[2] = Scalar(1./3.) * ( dab.z*fab[0] + dcb.z*fcb[0] );
        angle_virial[3] = Scalar(1./3.) * ( dab.y*fab[1] + dcb.y*fcb[1] );
        angle_virial[4] = Scalar(1./3.) * ( dab.z*fab[1] + dcb.z*fcb[1] );
        angle_virial[5] = Scalar(1./3.) * ( dab.z*fab[2] + dcb.z*fcb[2] );

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force_idx;
    for (unsigned int i = 0; i < 6 ; i++)
        d_virial[i*virial_pitch + idx] = virial[i];
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param alist List of angles stored on the GPU
    \param pitch Pitch of 2D angle list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param n_angle_type number of angle types
    \param d_tables Tables of the potential and force
    \param table_width Number of points in each table
    \param table_value indexer helper
    \param block_size Block size at which to run the kernel
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \note This is just a kernel driver. See gpu_compute_table_angle_forces_kernel for full documentation.
*/
cudaError_t gpu_compute_table_angle_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim &box,
                                     const group_storage<3> *alist,
                                     const unsigned int *apos_list,
                                     const unsigned int pitch,
                                     const unsigned int *n_angles_list,
                                     const Scalar2 *d_tables,
                                     const unsigned int table_width,
                                     const Index2D &table_value,
                                     const unsigned int block_size)
    {
    assert(d_tables);
    assert(table_width > 1);

    if (N == 0)
        return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_table_angle_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    Scalar delta_th = Scalar(M_PI)/(Scalar)(table_width - 1);

    gpu_compute_table_angle_forces_kernel<<< grid, threads >>>
            (d_force,
             d_virial,
             virial_pitch,
             N,
             d_pos,
             box,
             alist,
             apos_list,
             pitch,
             n_angles_list,
             d_tables,
             table_value,
             delta_th);

    return cudaSuccess;
    }

// vim:syntax=cpp
