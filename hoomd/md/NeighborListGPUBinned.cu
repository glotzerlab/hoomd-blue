// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "NeighborListGPUBinned.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/WarpTools.cuh"

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for O(N) neighbor list generation on the GPU
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel call for generating neighbor list on the GPU (Kepler optimized version)
/*! \tparam filter_body true when body filtering is enabled.
    \tparam enable_shared_cache true when the shared memory cache should be used.
    \param d_nlist Neighbor list data structure to write
    \param d_n_neigh Number of neighbors to write
    \param d_last_updated_pos Particle positions at this update are written to this array
    \param d_conditions Conditions array for writing overflow condition
    \param d_Nmax Maximum number of neighbors per type
    \param d_head_list List of indexes to access \a d_nlist
    \param d_pos Particle positions
    \param d_body Particle body indices
    \param N Number of particles
    \param d_cell_size Number of particles in each cell
    \param d_cell_xyzf Cell contents (xyzf array from CellList with flag=type)
    \param d_cell_idx Cell contents (particle indices)
    \param d_cell_type_body Cell contents (TypeBody array from CellList with)
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer for indexing cells
    \param cli Cell list indexer for indexing into d_cell_xyzf
    \param cadji Adjacent cell indexer listing the 27 neighboring cells
    \param box Simulation box dimensions
    \param d_r_cut Cutoff radius stored by pair type r_cut(i,j)
    \param r_buff The maximum radius for which to include particles as neighbors
    \param ntypes Number of particle types
    \param ghost_width Width of ghost cell layer
    \param offset Starting particle index
    \param nwork Number of particles to process

    \note optimized for Kepler
*/
template<unsigned char filter_body,
         unsigned char enable_shared_cache,
         int use_index,
         int threads_per_particle>
__global__ void gpu_compute_nlist_binned_kernel(unsigned int* d_nlist,
                                                unsigned int* d_n_neigh,
                                                Scalar4* d_last_updated_pos,
                                                unsigned int* d_conditions,
                                                const unsigned int* d_Nmax,
                                                const size_t* d_head_list,
                                                const Scalar4* d_pos,
                                                const unsigned int* d_body,
                                                const unsigned int N,
                                                const unsigned int* d_cell_size,
                                                const Scalar4* d_cell_xyzf,
                                                const unsigned int* d_cell_idx,
                                                const uint2* d_cell_type_body,
                                                const unsigned int* d_cell_adj,
                                                const Index3D ci,
                                                const Index2D cli,
                                                const Index2D cadji,
                                                const BoxDim box,
                                                const Scalar* d_r_cut,
                                                const Scalar r_buff,
                                                const unsigned int ntypes,
                                                const Scalar3 ghost_width,
                                                const unsigned int offset,
                                                const unsigned int nwork,
                                                const unsigned int ngpu)
    {
    // cache the r_listsq parameters into shared memory
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for per type pair parameters
    HIP_DYNAMIC_SHARED(unsigned char, s_data)

    // pointer for the r_listsq data
    Scalar* s_r_list = (Scalar*)(&s_data[0]);

    if (enable_shared_cache)
        {
        // load in the per type pair r_list
        for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < num_typ_parameters)
                {
                Scalar r_cut = d_r_cut[cur_offset + threadIdx.x];
                // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
                s_r_list[cur_offset + threadIdx.x]
                    = (r_cut > Scalar(0.0)) ? r_cut + r_buff : Scalar(-1.0);
                }
            }
        __syncthreads();
        }

    // each set of threads_per_particle threads is going to compute the neighbor list for a single
    // particle
    int my_pidx
        = blockIdx.x * (blockDim.x / threads_per_particle) + threadIdx.x / threads_per_particle;

    // one thread per particle
    if (my_pidx >= nwork)
        return;

    // get particle index
    my_pidx += offset;

    Scalar4 my_postype = d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_type = __scalar_as_int(my_postype.w);
    unsigned int my_body = d_body[my_pidx];
    size_t my_head = d_head_list[my_pidx];

    Scalar3 f = box.makeFraction(my_pos, ghost_width);

    // find the bin each particle belongs in
    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    uchar3 periodic = box.getPeriodic();

    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW() && periodic.x)
        ib = 0;
    if (jb == ci.getH() && periodic.y)
        jb = 0;
    if (kb == ci.getD() && periodic.z)
        kb = 0;

    int my_cell = ci(ib, jb, kb);

    // index of current neighbor
    unsigned int cur_adj = 0;

    // current device portion in cell list
    unsigned int igpu = 0;

    // current cell
    unsigned int neigh_cell = d_cell_adj[cadji(cur_adj, my_cell)];

    // size of current cell
    unsigned int neigh_size = d_cell_size[neigh_cell];

    // current index in cell
    int cur_offset = threadIdx.x % threads_per_particle;

    bool done = false;

    // total number of neighbors
    unsigned int nneigh = 0;

    unsigned int my_n_max = __ldg(d_Nmax + my_type);

    while (!done)
        {
        // initialize with default
        unsigned int neighbor;
        unsigned char has_neighbor = 0;

        // advance neighbor cell
        while (cur_offset >= neigh_size && !done)
            {
            cur_offset -= neigh_size;
            cur_adj++;
            if (cur_adj >= cadji.getW())
                {
                if (++igpu < ngpu)
                    {
                    cur_adj = 0;
                    }
                else
                    {
                    // we are past the end of the cell neighbors
                    done = true;
                    neigh_size = 0;
                    }
                }
            if (!done)
                {
                neigh_cell = __ldg(d_cell_adj + cadji(cur_adj, my_cell));
                neigh_size = __ldg(d_cell_size + neigh_cell + igpu * ci.getNumElements());
                }
            }
        // check for a neighbor if thread is still working
        if (!done)
            {
            Scalar4 cur_xyzf;
            unsigned int j;
            Scalar4 postype_j;
            if (!use_index)
                cur_xyzf = __ldg(d_cell_xyzf + cli(cur_offset, neigh_cell));
            else
                {
                j = __ldg(d_cell_idx + cli(cur_offset, neigh_cell) + igpu * cli.getNumElements());
                postype_j = d_pos[j];
                cur_xyzf = make_scalar4(postype_j.x, postype_j.y, postype_j.z, __int_as_scalar(j));
                }

            uint2 cur_type_body;
            if (!use_index)
                cur_type_body = __ldg(d_cell_type_body + cli(cur_offset, neigh_cell));
            else
                cur_type_body = make_uint2(__scalar_as_int(postype_j.w), __ldg(d_body + j));

            // advance cur_offset
            cur_offset += threads_per_particle;

            unsigned int neigh_type = cur_type_body.x;

            // Only do the hard work if the particle should be included by r_cut(i,j)
            Scalar r_list;

            if (enable_shared_cache)
                {
                r_list = s_r_list[typpair_idx(my_type, neigh_type)];
                }
            else
                {
                Scalar r_cut = d_r_cut[typpair_idx(my_type, neigh_type)];
                // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
                r_list = (r_cut > Scalar(0.0)) ? r_cut + r_buff : Scalar(-1.0);
                }

            if (r_list > Scalar(0.0))
                {
                unsigned int neigh_body = cur_type_body.y;

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                int cur_neigh = __scalar_as_int(cur_xyzf.w);

                // compute the distance between the two particles
                Scalar3 dx = my_pos - neigh_pos;

                // wrap the periodic boundary conditions
                dx = box.minImage(dx);

                // compute dr squared
                Scalar drsq = dot(dx, dx);

                bool excluded = (my_pidx == cur_neigh);

                if (filter_body && my_body != 0xffffffff)
                    excluded = excluded | (my_body == neigh_body);

                // store result in shared memory
                if (drsq <= r_list * r_list && !excluded)
                    {
                    neighbor = cur_neigh;
                    has_neighbor = 1;
                    }
                }
            }

        // now that possible neighbor checks are finished, done (for the cta) depends only on first
        // thread neighbor list only needs to get written into if thread 0 is not done
        done = hoomd::detail::WarpScan<bool, threads_per_particle>().Broadcast(done, 0);
        if (!done)
            {
            // scan over flags
            unsigned char k(0), n(0);
            hoomd::detail::WarpScan<unsigned char, threads_per_particle>().ExclusiveSum(
                has_neighbor,
                k,
                n);

            // write neighbor if it fits in list
            if (has_neighbor && (nneigh + k) < my_n_max)
                d_nlist[my_head + nneigh + k] = neighbor;

            // increment total neighbor count
            nneigh += n;
            }
        } // end while

    if (threadIdx.x % threads_per_particle == 0)
        {
        // flag if we need to grow the neighbor list
        if (nneigh >= my_n_max)
            atomicMax(&d_conditions[my_type], nneigh);

        d_n_neigh[my_pidx] = nneigh;
        d_last_updated_pos[my_pidx] = my_postype;
        }
    }

//! determine maximum possible block size
template<typename T> int get_max_block_size(T func)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % max_threads_per_particle;
    return max_threads;
    }

//! recursive template to launch neighborlist with given template parameters
/* \tparam cur_tpp Number of threads per particle (assumed to be power of two) */
template<int cur_tpp>
inline void launcher(unsigned int* d_nlist,
                     unsigned int* d_n_neigh,
                     Scalar4* d_last_updated_pos,
                     unsigned int* d_conditions,
                     const unsigned int* d_Nmax,
                     const size_t* d_head_list,
                     const Scalar4* d_pos,
                     const unsigned int* d_body,
                     const unsigned int N,
                     const unsigned int* d_cell_size,
                     const Scalar4* d_cell_xyzf,
                     const unsigned int* d_cell_idx,
                     const uint2* d_cell_type_body,
                     const unsigned int* d_cell_adj,
                     const Index3D ci,
                     const Index2D cli,
                     const Index2D cadji,
                     const BoxDim box,
                     const Scalar* d_r_cut,
                     const Scalar r_buff,
                     const unsigned int ntypes,
                     const Scalar3 ghost_width,
                     unsigned int tpp,
                     bool filter_body,
                     unsigned int block_size,
                     std::pair<unsigned int, unsigned int> range,
                     bool use_index,
                     const unsigned int ngpu,
                     const hipDeviceProp_t& devprop)
    {
    // shared memory = r_listsq + Nmax + stuff needed for neighborlist (computed below)
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = (unsigned int)(sizeof(Scalar) * typpair_idx.getNumElements());

    bool enable_shared = true;

    if (shared_size > devprop.sharedMemPerBlock)
        {
        enable_shared = false;
        shared_size = 0;
        }

    unsigned int offset = range.first;
    unsigned int nwork = range.second - range.first;

    if (tpp == cur_tpp && cur_tpp != 0)
        {
        if (!use_index)
            {
            if (!filter_body && !enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<0, 0, 0, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<0, 0, 0, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (filter_body && !enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<1, 0, 0, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<1, 0, 0, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (!filter_body && enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<0, 1, 0, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<0, 1, 0, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (filter_body && enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<1, 1, 0, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<1, 1, 0, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            }
        else // use_index
            {
            if (!filter_body && !enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<0, 0, 1, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<0, 0, 1, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (filter_body && !enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<1, 0, 1, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<1, 0, 1, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (!filter_body && enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<0, 1, 1, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<0, 1, 1, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            else if (filter_body && enable_shared)
                {
                unsigned int max_block_size;
                max_block_size
                    = get_max_block_size(gpu_compute_nlist_binned_kernel<1, 1, 1, cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size / tpp) + 1);

                hipLaunchKernelGGL((gpu_compute_nlist_binned_kernel<1, 1, 1, cur_tpp>),
                                   dim3(grid),
                                   dim3(block_size),
                                   shared_size,
                                   0,
                                   d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   d_Nmax,
                                   d_head_list,
                                   d_pos,
                                   d_body,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_idx,
                                   d_cell_type_body,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   d_r_cut,
                                   r_buff,
                                   ntypes,
                                   ghost_width,
                                   offset,
                                   nwork,
                                   ngpu);
                }
            }
        }
    else
        {
        launcher<cur_tpp / 2>(d_nlist,
                              d_n_neigh,
                              d_last_updated_pos,
                              d_conditions,
                              d_Nmax,
                              d_head_list,
                              d_pos,
                              d_body,
                              N,
                              d_cell_size,
                              d_cell_xyzf,
                              d_cell_idx,
                              d_cell_type_body,
                              d_cell_adj,
                              ci,
                              cli,
                              cadji,
                              box,
                              d_r_cut,
                              r_buff,
                              ntypes,
                              ghost_width,
                              tpp,
                              filter_body,
                              block_size,
                              range,
                              use_index,
                              ngpu,
                              devprop);
        }
    }

//! template specialization to terminate recursion
template<>
inline void launcher<min_threads_per_particle / 2>(unsigned int* d_nlist,
                                                   unsigned int* d_n_neigh,
                                                   Scalar4* d_last_updated_pos,
                                                   unsigned int* d_conditions,
                                                   const unsigned int* d_Nmax,
                                                   const size_t* d_head_list,
                                                   const Scalar4* d_pos,
                                                   const unsigned int* d_body,
                                                   const unsigned int N,
                                                   const unsigned int* d_cell_size,
                                                   const Scalar4* d_cell_xyzf,
                                                   const unsigned int* d_cell_idx,
                                                   const uint2* d_cell_type_body,
                                                   const unsigned int* d_cell_adj,
                                                   const Index3D ci,
                                                   const Index2D cli,
                                                   const Index2D cadji,
                                                   const BoxDim box,
                                                   const Scalar* d_r_cut,
                                                   const Scalar r_buff,
                                                   const unsigned int ntypes,
                                                   const Scalar3 ghost_width,
                                                   unsigned int tpp,
                                                   bool filter_body,
                                                   unsigned int block_size,
                                                   std::pair<unsigned int, unsigned int> range,
                                                   bool use_index,
                                                   const unsigned int ngpu,
                                                   const hipDeviceProp_t& devprop)
    {
    }

hipError_t gpu_compute_nlist_binned(unsigned int* d_nlist,
                                    unsigned int* d_n_neigh,
                                    Scalar4* d_last_updated_pos,
                                    unsigned int* d_conditions,
                                    const unsigned int* d_Nmax,
                                    const size_t* d_head_list,
                                    const Scalar4* d_pos,
                                    const unsigned int* d_body,
                                    const unsigned int N,
                                    const unsigned int* d_cell_size,
                                    const Scalar4* d_cell_xyzf,
                                    const unsigned int* d_cell_idx,
                                    const uint2* d_cell_type_body,
                                    const unsigned int* d_cell_adj,
                                    const Index3D& ci,
                                    const Index2D& cli,
                                    const Index2D& cadji,
                                    const BoxDim& box,
                                    const Scalar* d_r_cut,
                                    const Scalar r_buff,
                                    const unsigned int ntypes,
                                    const unsigned int threads_per_particle,
                                    const unsigned int block_size,
                                    bool filter_body,
                                    const Scalar3& ghost_width,
                                    const GPUPartition& gpu_partition,
                                    bool use_index,
                                    const hipDeviceProp_t& devprop)
    {
    unsigned int ngpu = gpu_partition.getNumActiveGPUs();

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        launcher<max_threads_per_particle>(d_nlist,
                                           d_n_neigh,
                                           d_last_updated_pos,
                                           d_conditions,
                                           d_Nmax,
                                           d_head_list,
                                           d_pos,
                                           d_body,
                                           N,
                                           d_cell_size,
                                           d_cell_xyzf,
                                           d_cell_idx,
                                           d_cell_type_body,
                                           d_cell_adj,
                                           ci,
                                           cli,
                                           cadji,
                                           box,
                                           d_r_cut,
                                           r_buff,
                                           ntypes,
                                           ghost_width,
                                           threads_per_particle,
                                           filter_body,
                                           block_size,
                                           range,
                                           use_index,
                                           ngpu,
                                           devprop);
        }
    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
