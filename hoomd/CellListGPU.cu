// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "CellListGPU.cuh"

#include "hoomd/extern/util/mgpucontext.h"
#include "hoomd/extern/kernels/localitysort.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

/*! \file CellListGPU.cu
    \brief Defines GPU kernel code for cell list generation on the GPU
*/

//! Kernel that computes the cell list on the GPU
/*! \param d_cell_size Number of particles in each cell
    \param d_xyzf Cell XYZF data array
    \param d_tdb Cell TDB data array
    \param d_cell_orientation Particle orientation in cell list
    \param d_cell_idx Particle index in cell list
    \param d_conditions Conditions flags for detecting overflow and other error conditions
    \param d_pos Particle position array
    \param d_orientation Particle orientation array
    \param d_charge Particle charge array
    \param d_diameter Particle diameter array
    \param d_body Particle body array
    \param N Number of particles
    \param n_ghost Number of ghost particles
    \param Nmax Maximum number of particles that can be placed in a single cell
    \param flag_charge Set to true to store charge in the flag position in \a d_xyzf
    \param flag_type Set to true to store type in the flag position in \a d_xyzf
    \param box Box dimensions
    \param ci Indexer to compute cell id from cell grid coords
    \param cli Indexer to index into \a d_xyzf and \a d_tdb
    \param ghost_width Width of ghost layer

    \note Optimized for Fermi
*/
__global__ void gpu_compute_cell_list_kernel(unsigned int *d_cell_size,
                                             Scalar4 *d_xyzf,
                                             Scalar4 *d_tdb,
                                             Scalar4 *d_cell_orientation,
                                             unsigned int *d_cell_idx,
                                             uint3 *d_conditions,
                                             const Scalar4 *d_pos,
                                             const Scalar4 *d_orientation,
                                             const Scalar *d_charge,
                                             const Scalar *d_diameter,
                                             const unsigned int *d_body,
                                             const unsigned int N,
                                             const unsigned int n_ghost,
                                             const unsigned int Nmax,
                                             const bool flag_charge,
                                             const bool flag_type,
                                             const BoxDim box,
                                             const Index3D ci,
                                             const Index2D cli,
                                             const Scalar3 ghost_width)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N + n_ghost)
        return;

    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar flag = 0;
    Scalar diameter = 0;
    Scalar body = 0;
    Scalar type = postype.w;
    Scalar4 orientation = make_scalar4(0,0,0,0);
    if (d_tdb != NULL)
        {
        diameter = d_diameter[idx];
        body = __int_as_scalar(d_body[idx]);
        }
    if (d_cell_orientation != NULL)
        {
        orientation = d_orientation[idx];
        }

    if (flag_charge)
        flag = d_charge[idx];
    else if (flag_type)
        flag = type;
    else
        flag = __int_as_scalar(idx);

    // check for nan pos
    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z))
        {
        (*d_conditions).y = idx+1;
        return;
        }

    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos,ghost_width);

    // check if the particle is inside the unit cell + ghost layer in all dimensions
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)) )
        {
        // if a ghost particle is out of bounds, silently ignore it
        if (idx < N)
            (*d_conditions).z = idx+1;
        return;
        }

    // find the bin each particle belongs in
    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW() && periodic.x)
        ib = 0;
    if (jb == ci.getH() && periodic.y)
        jb = 0;
    if (kb == ci.getD() && periodic.z)
        kb = 0;

    unsigned int bin = ci(ib, jb, kb);

    // all particles should be in a valid cell
    // all particles should be in a valid cell
    if (ib < 0 || ib >= (int)ci.getW() ||
        jb < 0 || jb >= (int)ci.getH() ||
        kb < 0 || kb >= (int)ci.getD())
        {
        // but ghost particles that are out of range should not produce an error
        if (idx < N)
            (*d_conditions).z = idx+1;
        return;
        }

    unsigned int size = atomicInc(&d_cell_size[bin], 0xffffffff);
    if (size < Nmax)
        {
        unsigned int write_pos = cli(size, bin);
        d_xyzf[write_pos] = make_scalar4(pos.x, pos.y, pos.z, flag);
        if (d_tdb != NULL)
            d_tdb[write_pos] = make_scalar4(type, diameter, body, 0);
        if (d_cell_orientation != NULL)
            d_cell_orientation[write_pos] = orientation;
        if (d_cell_idx != NULL)
            d_cell_idx[write_pos] = idx;
        }
    else
        {
        // handle overflow
        atomicMax(&(*d_conditions).x, size+1);
        }
    }

cudaError_t gpu_compute_cell_list(unsigned int *d_cell_size,
                                  Scalar4 *d_xyzf,
                                  Scalar4 *d_tdb,
                                  Scalar4 *d_cell_orientation,
                                  unsigned int *d_cell_idx,
                                  uint3 *d_conditions,
                                  const Scalar4 *d_pos,
                                  const Scalar4 *d_orientation,
                                  const Scalar *d_charge,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_body,
                                  const unsigned int N,
                                  const unsigned int n_ghost,
                                  const unsigned int Nmax,
                                  const bool flag_charge,
                                  const bool flag_type,
                                  const BoxDim& box,
                                  const Index3D& ci,
                                  const Index2D& cli,
                                  const Scalar3& ghost_width,
                                  const unsigned int block_size)
    {
    cudaError_t err;
    err = cudaMemsetAsync(d_cell_size, 0, sizeof(unsigned int)*ci.getNumElements(),0);

    if (err != cudaSuccess)
        return err;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)gpu_compute_cell_list_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    int n_blocks = (N+n_ghost)/run_block_size + 1;

    gpu_compute_cell_list_kernel<<<n_blocks, run_block_size>>>(d_cell_size,
                                                               d_xyzf,
                                                               d_tdb,
                                                               d_cell_orientation,
                                                               d_cell_idx,
                                                               d_conditions,
                                                               d_pos,
                                                               d_orientation,
                                                               d_charge,
                                                               d_diameter,
                                                               d_body,
                                                               N,
                                                               n_ghost,
                                                               Nmax,
                                                               flag_charge,
                                                               flag_type,
                                                               box,
                                                               ci,
                                                               cli,
                                                               ghost_width);

    return cudaSuccess;
    }

__global__ void gpu_fill_indices_kernel(
    unsigned int cl_size,
    uint2 *d_idx,
    unsigned int *d_sort_permutation,
    unsigned int *d_cell_idx,
    unsigned int *d_cell_size,
    Index3D ci,
    Index2D cli
    )
    {
    unsigned int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cell_idx >= cl_size) return;

    unsigned int icell = cell_idx / cli.getW();
    unsigned int pidx = UINT_MAX;

    if (icell < ci.getNumElements())
        {
        unsigned int my_cell_size = d_cell_size[icell];
        unsigned int ilocal = cell_idx % cli.getW();
        if (ilocal < my_cell_size)
            {
            pidx = d_cell_idx[cell_idx];
            }
        }

    // pack cell idx and particle idx into uint2
    uint2 result;
    result.x = icell;
    result.y = pidx;

    // write out result
    d_idx[cell_idx] = result;

    // write identity permutation
    d_sort_permutation[cell_idx] = cell_idx;
    }

//! Lexicographic comparison operator on uint2
struct comp_less_uint2
    {
    __device__ bool operator()(const uint2& a, const uint2& b)
        {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
        }
    };

__global__ void gpu_apply_sorted_cell_list_order(
    unsigned int cl_size,
    unsigned int *d_cell_idx,
    unsigned int *d_cell_idx_new,
    Scalar4 *d_xyzf,
    Scalar4 *d_xyzf_new,
    Scalar4 *d_tdb,
    Scalar4 *d_tdb_new,
    Scalar4 *d_cell_orientation,
    Scalar4 *d_cell_orientation_new,
    unsigned int *d_sort_permutation,
    Index2D cli)
    {
    unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= cl_size)
        return;

    unsigned int perm_idx = d_sort_permutation[cell_idx];

    d_xyzf_new[cell_idx] = d_xyzf[perm_idx];
    if (d_cell_idx) d_cell_idx_new[cell_idx] = d_cell_idx[perm_idx];
    if (d_tdb) d_tdb_new[cell_idx] = d_tdb[perm_idx];
    if (d_cell_orientation) d_cell_orientation_new[cell_idx] = d_cell_orientation[perm_idx];
    }

/*! Driver function to sort the cell list on the GPU

   This applies lexicographical order to cell idx, particle idx pairs
   \param d_cell_size List of cell sizes
   \param d_xyzf List of coordinates and flag
   \param d_tdb List type diameter and body index
   \param d_sort_idx Temporary array for storing the cell/particle indices to be sorted
   \param d_sort_permutation Temporary array for storing the permuted cell list indices
   \param ci Cell indexer
   \param cli Cell list indexer
   \param mgpu_context ModernGPU context
 */
cudaError_t gpu_sort_cell_list(unsigned int *d_cell_size,
                        Scalar4 *d_xyzf,
                        Scalar4 *d_xyzf_new,
                        Scalar4 *d_tdb,
                        Scalar4 *d_tdb_new,
                        Scalar4 *d_cell_orientation,
                        Scalar4 *d_cell_orientation_new,
                        unsigned int *d_cell_idx,
                        unsigned int *d_cell_idx_new,
                        uint2 *d_sort_idx,
                        unsigned int *d_sort_permutation,
                        const Index3D ci,
                        const Index2D cli)
    {
    unsigned int block_size = 256;

    // fill indices table with cell idx/particle idx pairs
    dim3 threads(block_size);
    dim3 grid(cli.getNumElements()/block_size + 1);

    gpu_fill_indices_kernel<<<grid, threads>>>
        (
        cli.getNumElements(),
        d_sort_idx,
        d_sort_permutation,
        d_cell_idx,
        d_cell_size,
        ci,
        cli);

    // locality sort on those pairs
    thrust::device_ptr<uint2> d_sort_idx_thrust(d_sort_idx);
    thrust::device_ptr<unsigned int> d_sort_permutation_thrust(d_sort_permutation);
    thrust::sort_by_key(d_sort_idx_thrust, d_sort_idx_thrust + cli.getNumElements(), d_sort_permutation_thrust, comp_less_uint2());

    // apply sorted order
    gpu_apply_sorted_cell_list_order<<<grid, threads>>>(
        cli.getNumElements(),
        d_cell_idx,
        d_cell_idx_new,
        d_xyzf,
        d_xyzf_new,
        d_tdb,
        d_tdb_new,
        d_cell_orientation,
        d_cell_orientation_new,
        d_sort_permutation,
        cli);

    // copy back permuted arrays to original ones
    cudaMemcpy(d_xyzf, d_xyzf_new, sizeof(Scalar4)*cli.getNumElements(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cell_idx, d_cell_idx_new, sizeof(unsigned int)*cli.getNumElements(), cudaMemcpyDeviceToDevice);
    if (d_tdb)
        {
        cudaMemcpy(d_tdb, d_tdb_new, sizeof(Scalar4)*cli.getNumElements(), cudaMemcpyDeviceToDevice);
        }
    if (d_cell_orientation)
        {
        cudaMemcpy(d_cell_orientation, d_cell_orientation_new, sizeof(Scalar4)*cli.getNumElements(), cudaMemcpyDeviceToDevice);
        }

    return cudaSuccess;
    }
