// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "NeighborListGPUBinned.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/WarpTools.cuh"

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for O(N) neighbor list generation on the GPU
*/

//! Kernel call for generating neighbor list on the GPU (Kepler optimized version)
/*! \tparam flags Set bit 1 to enable body filtering. Set bit 2 to enable diameter filtering.
    \param d_nlist Neighbor list data structure to write
    \param d_n_neigh Number of neighbors to write
    \param d_last_updated_pos Particle positions at this update are written to this array
    \param d_conditions Conditions array for writing overflow condition
    \param d_Nmax Maximum number of neighbors per type
    \param d_head_list List of indexes to access \a d_nlist
    \param d_pos Particle positions
    \param d_body Particle body indices
    \param d_diameter Particle diameters
    \param N Number of particles
    \param d_cell_size Number of particles in each cell
    \param d_cell_xyzf Cell contents (xyzf array from CellList with flag=type)
    \param d_cell_idx Cell contents (particle indices)
    \param d_cell_tdb Cell contents (tdb array from CellList with)
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
template<unsigned char flags, int use_index, int threads_per_particle>
__global__ void gpu_compute_nlist_binned_kernel(unsigned int *d_nlist,
                                                    unsigned int *d_n_neigh,
                                                    Scalar4 *d_last_updated_pos,
                                                    unsigned int *d_conditions,
                                                    const unsigned int *d_Nmax,
                                                    const unsigned int *d_head_list,
                                                    const Scalar4 *d_pos,
                                                    const unsigned int *d_body,
                                                    const Scalar *d_diameter,
                                                    const unsigned int N,
                                                    const unsigned int *d_cell_size,
                                                    const Scalar4 *d_cell_xyzf,
                                                    const unsigned int *d_cell_idx,
                                                    const Scalar4 *d_cell_tdb,
                                                    const unsigned int *d_cell_adj,
                                                    const Index3D ci,
                                                    const Index2D cli,
                                                    const Index2D cadji,
                                                    const BoxDim box,
                                                    const Scalar *d_r_cut,
                                                    const Scalar r_buff,
                                                    const unsigned int ntypes,
                                                    const Scalar3 ghost_width,
                                                    const unsigned int offset,
                                                    const unsigned int nwork,
                                                    const unsigned int ngpu)
    {
    bool filter_body = flags & 1;
    bool diameter_shift = flags & 2;

    // cache the r_listsq parameters into shared memory
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for per type pair parameters
    extern __shared__ unsigned char s_data[];

    // pointer for the r_listsq data
    Scalar *s_r_list = (Scalar *)(&s_data[0]);
    unsigned int *s_Nmax = (unsigned int *)(&s_data[sizeof(Scalar)*num_typ_parameters]);

    // load in the per type pair r_list
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            Scalar r_cut = d_r_cut[cur_offset + threadIdx.x];
            // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
            s_r_list[cur_offset + threadIdx.x] = (r_cut > Scalar(0.0)) ? r_cut+r_buff : Scalar(-1.0);
            }
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = d_Nmax[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // each set of threads_per_particle threads is going to compute the neighbor list for a single particle
    int my_pidx = blockIdx.x * (blockDim.x/threads_per_particle) + threadIdx.x/threads_per_particle;

    // one thread per particle
    if (my_pidx >= nwork) return;

    // get particle index
    my_pidx += offset;

    Scalar4 my_postype = d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_type = __scalar_as_int(my_postype.w);
    unsigned int my_body = d_body[my_pidx];
    Scalar my_diam = d_diameter[my_pidx];
    unsigned int my_head = d_head_list[my_pidx];

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

    int my_cell = ci(ib,jb,kb);

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

    while (! done)
        {
        // initialize with default
        unsigned int neighbor;
        unsigned char has_neighbor = 0;

        // advance neighbor cell
        while (cur_offset >= neigh_size && !done )
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
            if (! done)
                {
                neigh_cell = d_cell_adj[cadji(cur_adj, my_cell)];
                neigh_size = d_cell_size[neigh_cell+igpu*ci.getNumElements()];
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
                j = d_cell_idx[cli(cur_offset, neigh_cell)+igpu*cli.getNumElements()];
                postype_j = d_pos[j];
                cur_xyzf = make_scalar4(postype_j.x, postype_j.y, postype_j.z, __int_as_scalar(j));
                }

            Scalar4 cur_tdb;
            if (!use_index)
                cur_tdb = d_cell_tdb[cli(cur_offset, neigh_cell)];
            else
                cur_tdb = make_scalar4(postype_j.w, d_diameter[j], __int_as_scalar(d_body[j]),0);

            // advance cur_offset
            cur_offset += threads_per_particle;

            unsigned int neigh_type = __scalar_as_int(cur_tdb.x);

            // Only do the hard work if the particle should be included by r_cut(i,j)
            Scalar r_list = s_r_list[typpair_idx(my_type,neigh_type)];
            if (r_list > Scalar(0.0))
                {
                Scalar neigh_diam = cur_tdb.y;
                unsigned int neigh_body = __scalar_as_int(cur_tdb.z);

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x,
                                               cur_xyzf.y,
                                               cur_xyzf.z);
                int cur_neigh = __scalar_as_int(cur_xyzf.w);

                // compute the distance between the two particles
                Scalar3 dx = my_pos - neigh_pos;

                // wrap the periodic boundary conditions
                dx = box.minImage(dx);

                // compute dr squared
                Scalar drsq = dot(dx,dx);

                bool excluded = (my_pidx == cur_neigh);

                if (filter_body && my_body != 0xffffffff)
                    excluded = excluded | (my_body == neigh_body);

                Scalar sqshift = Scalar(0.0);
                if (diameter_shift)
                    {
                    const Scalar delta = (my_diam + neigh_diam) * Scalar(0.5) - Scalar(1.0);
                    // r^2 < (r_list + delta)^2
                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                    sqshift = (delta + Scalar(2.0) * r_list) * delta;
                    }

                // store result in shared memory
                if (drsq <= (r_list*r_list + sqshift) && !excluded)
                    {
                    neighbor = cur_neigh;
                    has_neighbor = 1;
                    }
                }
            }

        // now that possible neighbor checks are finished, done (for the cta) depends only on first thread
        // neighbor list only needs to get written into if thread 0 is not done
        done = hoomd::detail::WarpScan<bool, threads_per_particle>().Broadcast(done, 0);
        if (!done)
            {
            // scan over flags
            unsigned char k(0), n(0);
            hoomd::detail::WarpScan<unsigned char, threads_per_particle>().ExclusiveSum(has_neighbor, k, n);

            // write neighbor if it fits in list
            if (has_neighbor && (nneigh + k) < s_Nmax[my_type])
                d_nlist[my_head + nneigh + k] = neighbor;

            // increment total neighbor count
            nneigh += n;
            }
        } // end while

    if (threadIdx.x % threads_per_particle == 0)
        {
        // flag if we need to grow the neighbor list
        if (nneigh >= s_Nmax[my_type])
            atomicMax(&d_conditions[my_type], nneigh);

        d_n_neigh[my_pidx] = nneigh;
        d_last_updated_pos[my_pidx] = my_postype;
        }
    }

//! determine maximum possible block size
template<typename T>
int get_max_block_size(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % max_threads_per_particle;
    return max_threads;
    }

//! recursive template to launch neighborlist with given template parameters
/* \tparam cur_tpp Number of threads per particle (assumed to be power of two) */
template<int cur_tpp>
inline void launcher(unsigned int *d_nlist,
              unsigned int *d_n_neigh,
              Scalar4 *d_last_updated_pos,
              unsigned int *d_conditions,
              const unsigned int *d_Nmax,
              const unsigned int *d_head_list,
              const Scalar4 *d_pos,
              const unsigned int *d_body,
              const Scalar *d_diameter,
              const unsigned int N,
              const unsigned int *d_cell_size,
              const Scalar4 *d_cell_xyzf,
              const unsigned int *d_cell_idx,
              const Scalar4 *d_cell_tdb,
              const unsigned int *d_cell_adj,
              const Index3D ci,
              const Index2D cli,
              const Index2D cadji,
              const BoxDim box,
              const Scalar *d_r_cut,
              const Scalar r_buff,
              const unsigned int ntypes,
              const Scalar3 ghost_width,
              const unsigned int compute_capability,
              unsigned int tpp,
              bool filter_body,
              bool diameter_shift,
              unsigned int block_size,
              std::pair<unsigned int, unsigned int> range,
              bool use_index,
              const unsigned int ngpu)
    {
    // shared memory = r_listsq + Nmax + stuff needed for neighborlist (computed below)
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = sizeof(Scalar)*typpair_idx.getNumElements() + sizeof(unsigned int)*ntypes;

    unsigned int offset = range.first;
    unsigned int nwork = range.second - range.first;

    if (tpp == cur_tpp && cur_tpp != 0)
        {
        if (!use_index)
            {
            if (!diameter_shift && !filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<0,0,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<0,0,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (!diameter_shift && filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<1,0,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<1,0,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (diameter_shift && !filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<2,0,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<2,0,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (diameter_shift && filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<3,0,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<3,0,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            if (!diameter_shift && !filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<0,1,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<0,1,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (!diameter_shift && filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<1,1,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<1,1,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (diameter_shift && !filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<2,1,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<2,1,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
            else if (diameter_shift && filter_body)
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_nlist_binned_kernel<3,1,cur_tpp>);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(nwork / (block_size/tpp) + 1);

                gpu_compute_nlist_binned_kernel<3,1,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                                             d_n_neigh,
                                                                                             d_last_updated_pos,
                                                                                             d_conditions,
                                                                                             d_Nmax,
                                                                                             d_head_list,
                                                                                             d_pos,
                                                                                             d_body,
                                                                                             d_diameter,
                                                                                             N,
                                                                                             d_cell_size,
                                                                                             d_cell_xyzf,
                                                                                             d_cell_idx,
                                                                                             d_cell_tdb,
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
        launcher<cur_tpp/2>(d_nlist,
                     d_n_neigh,
                     d_last_updated_pos,
                     d_conditions,
                     d_Nmax,
                     d_head_list,
                     d_pos,
                     d_body,
                     d_diameter,
                     N,
                     d_cell_size,
                     d_cell_xyzf,
                     d_cell_idx,
                     d_cell_tdb,
                     d_cell_adj,
                     ci,
                     cli,
                     cadji,
                     box,
                     d_r_cut,
                     r_buff,
                     ntypes,
                     ghost_width,
                     compute_capability,
                     tpp,
                     filter_body,
                     diameter_shift,
                     block_size,
                     range,
                     use_index,
                     ngpu
                     );
        }
    }

//! template specialization to terminate recursion
template<>
inline void launcher<min_threads_per_particle/2>(unsigned int *d_nlist,
              unsigned int *d_n_neigh,
              Scalar4 *d_last_updated_pos,
              unsigned int *d_conditions,
              const unsigned int *d_Nmax,
              const unsigned int *d_head_list,
              const Scalar4 *d_pos,
              const unsigned int *d_body,
              const Scalar *d_diameter,
              const unsigned int N,
              const unsigned int *d_cell_size,
              const Scalar4 *d_cell_xyzf,
              const unsigned int *d_cell_idx,
              const Scalar4 *d_cell_tdb,
              const unsigned int *d_cell_adj,
              const Index3D ci,
              const Index2D cli,
              const Index2D cadji,
              const BoxDim box,
              const Scalar *d_r_cut,
              const Scalar r_buff,
              const unsigned int ntypes,
              const Scalar3 ghost_width,
              const unsigned int compute_capability,
              unsigned int tpp,
              bool filter_body,
              bool diameter_shift,
              unsigned int block_size,
              std::pair<unsigned int, unsigned int> range,
              bool use_index,
              const unsigned int ngpu)
    { }

cudaError_t gpu_compute_nlist_binned(unsigned int *d_nlist,
                                     unsigned int *d_n_neigh,
                                     Scalar4 *d_last_updated_pos,
                                     unsigned int *d_conditions,
                                     const unsigned int *d_Nmax,
                                     const unsigned int *d_head_list,
                                     const Scalar4 *d_pos,
                                     const unsigned int *d_body,
                                     const Scalar *d_diameter,
                                     const unsigned int N,
                                     const unsigned int *d_cell_size,
                                     const Scalar4 *d_cell_xyzf,
                                     const unsigned int *d_cell_idx,
                                     const Scalar4 *d_cell_tdb,
                                     const unsigned int *d_cell_adj,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Index2D& cadji,
                                     const BoxDim& box,
                                     const Scalar *d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     const unsigned int threads_per_particle,
                                     const unsigned int block_size,
                                     bool filter_body,
                                     bool diameter_shift,
                                     const Scalar3& ghost_width,
                                     const unsigned int compute_capability,
                                     const GPUPartition& gpu_partition,
                                     bool use_index)
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
                                       d_diameter,
                                       N,
                                       d_cell_size,
                                       d_cell_xyzf,
                                       d_cell_idx,
                                       d_cell_tdb,
                                       d_cell_adj,
                                       ci,
                                       cli,
                                       cadji,
                                       box,
                                       d_r_cut,
                                       r_buff,
                                       ntypes,
                                       ghost_width,
                                       compute_capability,
                                       threads_per_particle,
                                       filter_body,
                                       diameter_shift,
                                       block_size,
                                       range,
                                       use_index,
                                       ngpu
                                       );
        }
    return cudaSuccess;
    }
