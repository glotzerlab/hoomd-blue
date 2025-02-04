// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CommunicatorGPU.cu
    \brief Implementation of communication algorithms on the GPU
*/

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"
#include "ParticleData.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hipcub/hipcub.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#pragma GCC diagnostic pop

#include <cassert>

namespace hoomd
    {
//! Select a particle for migration
__global__ void gpu_select_particle_migrate(unsigned int N,
                                            const Scalar4* d_postype,
                                            unsigned int* d_flags,
                                            unsigned int comm_mask,
                                            const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    Scalar3 f = box.makeFraction(pos);

    unsigned int flags = 0;
    if (f.x >= Scalar(1.0))
        flags |= send_east;
    if (f.x < Scalar(0.0))
        flags |= send_west;
    if (f.y >= Scalar(1.0))
        flags |= send_north;
    if (f.y < Scalar(0.0))
        flags |= send_south;
    if (f.z >= Scalar(1.0))
        flags |= send_up;
    if (f.z < Scalar(0.0))
        flags |= send_down;

    // filter allowed directions
    flags &= comm_mask;

    d_flags[idx] = flags;
    }

//! Select a particle for migration
struct get_migrate_key_gpu : public thrust::unary_function<const unsigned int, unsigned int>
    {
    const uint3 my_pos;               //!< My domain decomposition position
    const Index3D di;                 //!< Domain indexer
    const unsigned int mask;          //!< Mask of allowed directions
    const unsigned int* d_cart_ranks; //!< Rank lookup table

    //! Constructor
    /*!
     */
    __host__ __device__ get_migrate_key_gpu(const uint3 _my_pos,
                                            const Index3D _di,
                                            const unsigned int _mask,
                                            const unsigned int* _d_cart_ranks)
        : my_pos(_my_pos), di(_di), mask(_mask), d_cart_ranks(_d_cart_ranks)
        {
        }

    //! Generate key for a sent particle
    __device__ unsigned int operator()(const unsigned int flags)
        {
        int ix, iy, iz;
        ix = iy = iz = 0;

        if ((flags & send_east) && (mask & send_east))
            ix = 1;
        else if ((flags & send_west) && (mask & send_west))
            ix = -1;

        if ((flags & send_north) && (mask & send_north))
            iy = 1;
        else if ((flags & send_south) && (mask & send_south))
            iy = -1;

        if ((flags & send_up) && (mask & send_up))
            iz = 1;
        else if ((flags & send_down) && (mask & send_down))
            iz = -1;

        int i = my_pos.x;
        int j = my_pos.y;
        int k = my_pos.z;

        i += ix;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        j += iy;
        if (j == (int)di.getH())
            j = 0;
        else if (j < 0)
            j += di.getH();

        k += iz;
        if (k == (int)di.getD())
            k = 0;
        else if (k < 0)
            k += di.getD();

        return d_cart_ranks[di(i, j, k)];
        }
    };

/*! \param N Number of local particles
    \param d_pos Device array of particle positions
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param box Local box
    \param comm_mask Mask of allowed communication directions
    \param alloc Caching allocator
 */
void gpu_stage_particles(const unsigned int N,
                         const Scalar4* d_pos,
                         unsigned int* d_comm_flag,
                         const BoxDim& box,
                         const unsigned int comm_mask)
    {
    assert(d_pos);
    assert(d_comm_flag);

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_select_particle_migrate),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_pos,
                       d_comm_flag,
                       comm_mask,
                       box);
    }

/*! \param nsend Number of particles in buffer
    \param d_in Send buf (in-place sort)
    \param d_comm_flags Buffer of communication flags
    \param di Domain indexer
    \param box Local box
    \param d_keys Output array (target domains)
    \param d_begin Output array (start indices per key in send buf)
    \param d_end Output array (end indices per key in send buf)
    \param d_neighbors List of neighbor ranks
    \param mask Mask of communicating directions
    \param alloc Caching allocator
 */
void gpu_sort_migrating_particles(const size_t nsend,
                                  detail::pdata_element* d_in,
                                  const unsigned int* d_comm_flags,
                                  const Index3D& di,
                                  const uint3 my_pos,
                                  const unsigned int* d_cart_ranks,
                                  unsigned int* d_keys,
                                  unsigned int* d_begin,
                                  unsigned int* d_end,
                                  const unsigned int* d_neighbors,
                                  const unsigned int nneigh,
                                  const unsigned int mask,
                                  unsigned int* d_tmp,
                                  detail::pdata_element* d_in_copy,
                                  CachedAllocator& alloc)
    {
    assert(d_in);
    assert(d_comm_flags);
    assert(d_cart_ranks);
    assert(d_keys);
    assert(d_begin);
    assert(d_end);
    assert(d_neighbors);

    // Wrap input & output
    thrust::device_ptr<detail::pdata_element> in_ptr(d_in);
    thrust::device_ptr<const unsigned int> comm_flags_ptr(d_comm_flags);
    thrust::device_ptr<unsigned int> keys_ptr(d_keys);
    thrust::device_ptr<const unsigned int> neighbors_ptr(d_neighbors);

    // generate keys
    thrust::transform(comm_flags_ptr,
                      comm_flags_ptr + nsend,
                      keys_ptr,
                      get_migrate_key_gpu(my_pos, di, mask, d_cart_ranks));

    // allocate temp arrays
    thrust::device_ptr<unsigned int> tmp_ptr(d_tmp);
    thrust::device_ptr<detail::pdata_element> in_copy_ptr(d_in_copy);

    // copy and fill with ascending integer sequence
    thrust::counting_iterator<unsigned int> count_it(0);
    thrust::copy(make_zip_iterator(thrust::make_tuple(count_it, in_ptr)),
                 thrust::make_zip_iterator(thrust::make_tuple(count_it + nsend, in_ptr + nsend)),
                 thrust::make_zip_iterator(thrust::make_tuple(tmp_ptr, in_copy_ptr)));

    // sort buffer by neighbors
#ifdef __HIP_PLATFORM_HCC__
    thrust::sort_by_key(thrust::hip::par(alloc),
#else
    thrust::sort_by_key(thrust::cuda::par(alloc),
#endif
                        keys_ptr,
                        keys_ptr + nsend,
                        tmp_ptr);

    // reorder send buf
    thrust::gather(tmp_ptr, tmp_ptr + nsend, in_copy_ptr, in_ptr);

    thrust::device_ptr<unsigned int> begin_ptr(d_begin);
    thrust::device_ptr<unsigned int> end_ptr(d_end);

    thrust::lower_bound(keys_ptr,
                        keys_ptr + nsend,
                        neighbors_ptr,
                        neighbors_ptr + nneigh,
                        begin_ptr);

    thrust::upper_bound(keys_ptr, keys_ptr + nsend, neighbors_ptr, neighbors_ptr + nneigh, end_ptr);
    }

__global__ void gpu_wrap_particles_kernel(const unsigned int n_recv,
                                          detail::pdata_element* d_recv,
                                          const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_recv)
        return;

    detail::pdata_element p = d_recv[idx];
    box.wrap(p.pos, p.image);
    d_recv[idx] = p;
    }

/*! \param n_recv Number of particles in buffer
    \param d_in Buffer of particle data elements
    \param box Box for which to apply boundary conditions
 */
void gpu_wrap_particles(const unsigned int n_recv, detail::pdata_element* d_in, const BoxDim& box)
    {
    assert(d_in);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;

    hipLaunchKernelGGL(gpu_wrap_particles_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_recv,
                       d_in,
                       box);
    }

//! Reset reverse lookup tags of particles we are removing
/* \param n_delete_ptls Number of particles to delete
 * \param d_delete_tags Array of particle tags to delete
 * \param d_rtag Array for tag->idx lookup
 */
void gpu_reset_rtags(unsigned int n_delete_ptls, unsigned int* d_delete_tags, unsigned int* d_rtag)
    {
    assert(d_delete_tags);
    assert(d_rtag);

    thrust::device_ptr<unsigned int> delete_tags_ptr(d_delete_tags);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::constant_iterator<unsigned int> not_local(NOT_LOCAL);
    thrust::scatter(not_local, not_local + n_delete_ptls, delete_tags_ptr, rtag_ptr);
    }

//! Kernel to select ghost atoms due to non-bonded interactions
__global__ void gpu_make_ghost_exchange_plan_kernel(unsigned int N,
                                                    const Scalar4* d_postype,
                                                    const unsigned int* d_body,
                                                    unsigned int* d_plan,
                                                    const BoxDim box,
                                                    const Scalar* d_r_ghost,
                                                    const Scalar* d_r_ghost_body,
                                                    Scalar r_ghost_max,
                                                    unsigned int ntypes,
                                                    unsigned int mask)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    Scalar3 npd = box.getNearestPlaneDistance();

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);
    Scalar ghost_width = __ldg(d_r_ghost + type);

    if (d_body[idx] < MIN_FLOPPY)
        {
        ghost_width = max(ghost_width, __ldg(d_r_ghost_body + type));
        }

    Scalar3 ghost_fraction = ghost_width / npd;

    Scalar3 f = box.makeFraction(pos);

    unsigned int plan = 0;

    // is particle inside ghost layer? set plan accordingly.
    if (f.x >= Scalar(1.0) - ghost_fraction.x)
        plan |= send_east;
    if (f.x < ghost_fraction.x)
        plan |= send_west;
    if (f.y >= Scalar(1.0) - ghost_fraction.y)
        plan |= send_north;
    if (f.y < ghost_fraction.y)
        plan |= send_south;
    if (f.z >= Scalar(1.0) - ghost_fraction.z)
        plan |= send_up;
    if (f.z < ghost_fraction.z)
        plan |= send_down;

    // filter out non-communicating directions
    plan &= mask;

    d_plan[idx] = plan;
    };

//! Construct plans for sending non-bonded ghost particles
/*! \param d_plan Array of ghost particle plans
 * \param N number of particles to check
 * \param d_pos Array of particle positions
 * \param box Dimensions of local simulation box
 * \param r_ghost Width of boundary layer
 */
void gpu_make_ghost_exchange_plan(unsigned int* d_plan,
                                  unsigned int N,
                                  const Scalar4* d_pos,
                                  const unsigned int* d_body,
                                  const BoxDim& box,
                                  const Scalar* d_r_ghost,
                                  const Scalar* d_r_ghost_body,
                                  Scalar r_ghost_max,
                                  unsigned int ntypes,
                                  unsigned int mask)
    {
    assert(d_plan);
    assert(d_pos);
    assert(d_body);
    assert(d_r_ghost);
    assert(d_r_ghost_body);

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    hipLaunchKernelGGL(gpu_make_ghost_exchange_plan_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_pos,
                       d_body,
                       d_plan,
                       box,
                       d_r_ghost,
                       d_r_ghost_body,
                       r_ghost_max,
                       ntypes,
                       mask);
    }

__device__ unsigned int get_direction_mask(unsigned int plan)
    {
    unsigned int mask = 0;
    for (int ix = -1; ix <= 1; ix++)
        for (int iy = -1; iy <= 1; iy++)
            for (int iz = -1; iz <= 1; iz++)
                {
                unsigned int flags = 0;
                if (ix == 1)
                    flags |= send_east;
                if (ix == -1)
                    flags |= send_west;
                if (iy == 1)
                    flags |= send_north;
                if (iy == -1)
                    flags |= send_south;
                if (iz == 1)
                    flags |= send_up;
                if (iz == -1)
                    flags |= send_down;

                unsigned int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
                if ((flags & plan) == flags)
                    mask |= (1 << dir);
                }

    return mask;
    }

//! Kernel to select ghost atoms due to non-bonded interactions
template<unsigned int group_size, typename members_t>
__global__ void gpu_make_ghost_group_exchange_plan_kernel(unsigned int N,
                                                          const members_t* d_groups,
                                                          unsigned int* d_group_plan,
                                                          const unsigned int* d_rtag,
                                                          const unsigned int* d_plans,
                                                          unsigned int n_local)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    unsigned int plan = 0;
    members_t members = d_groups[idx];

    unsigned int gs = group_size;

    for (unsigned int i = 0; i < gs; ++i)
        {
        unsigned int tag = members.tag[i];
        unsigned int pidx = d_rtag[tag];

        if (i == 0 && pidx >= n_local)
            {
            // only the rank that owns the first ptl of a group sends it as a ghost
            plan = 0;
            break;
            }

        if (pidx < n_local)
            {
            plan |= d_plans[pidx];
            }
        }

    d_group_plan[idx] = plan;
    };

template<unsigned int group_size, typename members_t>
void gpu_make_ghost_group_exchange_plan(unsigned int* d_ghost_group_plan,
                                        const members_t* d_groups,
                                        unsigned int N,
                                        const unsigned int* d_rtag,
                                        const unsigned int* d_plans,
                                        unsigned int n_local)
    {
    assert(d_ghost_group_plan);
    assert(d_groups);
    assert(d_rtag);
    assert(d_plans);

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_make_ghost_group_exchange_plan_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_groups,
                       d_ghost_group_plan,
                       d_rtag,
                       d_plans,
                       n_local);
    }

//! Apply adjacency masks to plan and return number of matching neighbors
__global__ void gpu_ghost_neighbor_counts(unsigned int N,
                                          const unsigned int* d_ghost_plan,
                                          unsigned int* d_counts,
                                          const unsigned int* d_adj,
                                          unsigned int nneigh)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    unsigned int plan = d_ghost_plan[idx];
    unsigned int count = 0;
    unsigned int mask = get_direction_mask(plan);

    for (unsigned int i = 0; i < nneigh; i++)
        {
        unsigned int adj = d_adj[i];

        if (adj & mask)
            count++;
        }

    d_counts[idx] = count;
    };

//! Apply adjacency masks to plan and integer and return nth matching neighbor rank
struct get_neighbor_rank_n
    {
    const unsigned int* d_adj;
    const unsigned int* d_neighbor;
    const unsigned int nneigh;

    __host__ __device__ get_neighbor_rank_n(const unsigned int* _d_adj,
                                            const unsigned int* _d_neighbor,
                                            unsigned int _nneigh)
        : d_adj(_d_adj), d_neighbor(_d_neighbor), nneigh(_nneigh)
        {
        }

    __device__ uint2 operator()(unsigned int plan, unsigned int n)
        {
        unsigned int count = 0;
        unsigned int ineigh;
        unsigned int mask = get_direction_mask(plan);

        for (ineigh = 0; ineigh < nneigh; ineigh++)
            {
            unsigned int adj = d_adj[ineigh];
            if (adj & mask)
                {
                if (count == n)
                    break;
                count++;
                }
            }
        return make_uint2(d_neighbor[ineigh], d_adj[ineigh] & mask);
        }
    };

unsigned int gpu_exchange_ghosts_count_neighbors(unsigned int N,
                                                 const unsigned int* d_ghost_plan,
                                                 const unsigned int* d_adj,
                                                 unsigned int* d_counts,
                                                 unsigned int* d_scan,
                                                 unsigned int nneigh,
                                                 CachedAllocator& alloc)
    {
    if (!N)
        return 0;

    assert(d_ghost_plan);
    assert(d_adj);
    assert(d_counts);

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    // compute neighbor counts
    hipLaunchKernelGGL(gpu_ghost_neighbor_counts,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_ghost_plan,
                       d_counts,
                       d_adj,
                       nneigh);

    // determine output size
    unsigned int total;
    assert(d_scan);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_scan, N);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_scan, N);
    alloc.deallocate((char*)d_temp_storage);

    // determine total number of ghosts
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_total = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_total);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts, d_total, N);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_counts, d_total, N);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&total, d_total, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_total);

    return total;
    }

__global__ void gpu_expand_neighbors_kernel(const unsigned int n_out,
                                            const unsigned int* d_output_indices,
                                            const unsigned int* d_scan,
                                            const unsigned int* d_plan,
                                            uint2* d_idx_adj_out,
                                            const unsigned int* d_neighbors,
                                            const unsigned int* d_adj,
                                            const unsigned int nneigh,
                                            unsigned int* d_neighbors_out)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_out)
        return;

    // Now use the segmented scan to fetch nth neighbor
    get_neighbor_rank_n getn(d_adj, d_neighbors, nneigh);

    unsigned int pidx = d_output_indices[idx];
    int n = idx - d_scan[pidx];
    int plan = d_plan[pidx];
    uint2 neighbor_adj = getn(plan, n);

    // write out values to global mem
    d_neighbors_out[idx] = neighbor_adj.x;
    d_idx_adj_out[idx] = make_uint2(pidx, neighbor_adj.y);
    }

void gpu_exchange_ghosts_make_indices(unsigned int N,
                                      const unsigned int* d_ghost_plan,
                                      const unsigned int* d_tag,
                                      const unsigned int* d_adj,
                                      const unsigned int* d_unique_neighbors,
                                      const unsigned int* d_counts,
                                      const unsigned int* d_scan,
                                      uint2* d_ghost_idx_adj,
                                      unsigned int* d_ghost_neigh,
                                      unsigned int* d_ghost_begin,
                                      unsigned int* d_ghost_end,
                                      unsigned int n_unique_neigh,
                                      unsigned int n_out,
                                      unsigned int mask,
                                      CachedAllocator& alloc)
    {
    assert(d_ghost_plan);
    assert(d_tag);
    assert(d_adj);
    assert(d_unique_neighbors);
    assert(d_counts);
    assert(d_scan);
    assert(d_ghost_idx_adj);
    assert(d_ghost_neigh);
    assert(d_ghost_begin);
    assert(d_ghost_end);

    /*
     * expand each tag by the number of neighbors to send the corresponding ptl to
     * and assign each copy to a different neighbor
     */

    if (n_out)
        {
        // scatter the nonzero counts into their corresponding output positions
        thrust::device_ptr<const unsigned int> counts(d_counts);
        thrust::device_ptr<const unsigned int> scan(d_scan);
        unsigned int* d_output_indices = alloc.getTemporaryBuffer<unsigned int>(n_out);
        assert(d_output_indices);

        thrust::device_ptr<unsigned int> output_indices(d_output_indices);
#ifdef __HIP_PLATFORM_HCC__
        thrust::fill(thrust::hip::par(alloc),
#else
        thrust::fill(thrust::cuda::par(alloc),
#endif
                     output_indices,
                     output_indices + n_out,
                     0);

#ifdef __HIP_PLATFORM_HCC__
        thrust::scatter_if(thrust::hip::par(alloc),
#else
        thrust::scatter_if(thrust::cuda::par(alloc),
#endif
                           thrust::counting_iterator<unsigned int>(0),
                           thrust::counting_iterator<unsigned int>(N),
                           scan,
                           counts,
                           output_indices);

// compute max-scan over the output indices, filling in the holes
#ifdef __HIP_PLATFORM_HCC__
        thrust::inclusive_scan(thrust::hip::par(alloc),
#else
        thrust::inclusive_scan(thrust::cuda::par(alloc),
#endif
                               output_indices,
                               output_indices + n_out,
                               output_indices,
                               thrust::maximum<unsigned int>());
        // output_indices now contains the source particle index, replicated by the number of ghosts

        // fill destination arrays
        unsigned int block_size = 256;
        unsigned int n_blocks = n_out / block_size + 1;

        hipLaunchKernelGGL(gpu_expand_neighbors_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_output_indices,
                           d_scan,
                           d_ghost_plan,
                           d_ghost_idx_adj,
                           d_unique_neighbors,
                           d_adj,
                           n_unique_neigh,
                           d_ghost_neigh);
        alloc.deallocate((char*)d_output_indices);

        // sort by neighbor
        thrust::device_ptr<unsigned int> ghost_neigh(d_ghost_neigh);
        thrust::device_ptr<uint2> ghost_idx_adj(d_ghost_idx_adj);
#ifdef __HIP_PLATFORM_HCC__
        thrust::sort_by_key(thrust::hip::par(alloc),
#else
        thrust::sort_by_key(thrust::cuda::par(alloc),
#endif
                            ghost_neigh,
                            ghost_neigh + n_out,
                            ghost_idx_adj);

        thrust::device_ptr<const unsigned int> unique_neighbors(d_unique_neighbors);
        thrust::device_ptr<unsigned int> ghost_begin(d_ghost_begin);
        thrust::device_ptr<unsigned int> ghost_end(d_ghost_end);

        thrust::lower_bound(ghost_neigh,
                            ghost_neigh + n_out,
                            unique_neighbors,
                            unique_neighbors + n_unique_neigh,
                            ghost_begin);

        thrust::upper_bound(ghost_neigh,
                            ghost_neigh + n_out,
                            unique_neighbors,
                            unique_neighbors + n_unique_neigh,
                            ghost_end);
        }
    else
        {
        hipMemset(d_ghost_begin, 0, sizeof(unsigned int) * n_unique_neigh);
        hipMemset(d_ghost_end, 0, sizeof(unsigned int) * n_unique_neigh);
        }
    }

template<typename T>
__global__ void
gpu_pack_kernel(unsigned int n_out, const uint2* d_ghost_idx_adj, const T* in, T* out)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_out)
        return;
    unsigned int idx = d_ghost_idx_adj[buf_idx].x;
    out[buf_idx] = in[idx];
    }

__global__ void gpu_pack_wrap_kernel(unsigned int n_out,
                                     const uint2* d_ghost_idx_adj,
                                     const Scalar4* d_postype,
                                     const int3* d_img,
                                     Scalar4* out_pos,
                                     int3* out_img,
                                     Index3D di,
                                     uint3 my_pos,
                                     BoxDim box)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_out)
        return;

    uint2 idx_adj = d_ghost_idx_adj[buf_idx];
    unsigned int idx = idx_adj.x;
    unsigned int adj = idx_adj.y;

    // get direction triple from adjacency element
    // wrap
    uchar3 periodic = make_uchar3(0, 0, 0);
    char3 wrap = make_char3(0, 0, 0);

    const unsigned int mask_east
        = 1 << 2 | 1 << 5 | 1 << 8 | 1 << 11 | 1 << 14 | 1 << 17 | 1 << 20 | 1 << 23 | 1 << 26;
    const unsigned int mask_west = mask_east >> 2;
    const unsigned int mask_north
        = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 15 | 1 << 16 | 1 << 17 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_south = mask_north >> 6;
    const unsigned int mask_up
        = 1 << 18 | 1 << 19 | 1 << 20 | 1 << 21 | 1 << 22 | 1 << 23 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_down = mask_up >> 18;

    if (di.getW() > 1)
        {
        if (my_pos.x == di.getW() - 1 && (adj & mask_east))
            {
            wrap.x = 1;
            periodic.x = 1;
            }
        else if (my_pos.x == 0 && (adj & mask_west))
            {
            wrap.x = -1;
            periodic.x = 1;
            }
        }
    if (di.getH() > 1)
        {
        if (my_pos.y == di.getH() - 1 && (adj & mask_north))
            {
            wrap.y = 1;
            periodic.y = 1;
            }
        else if (my_pos.y == 0 && (adj & mask_south))
            {
            wrap.y = -1;
            periodic.y = 1;
            }
        }
    if (di.getD() > 1)
        {
        if (my_pos.z == di.getD() - 1 && (adj & mask_up))
            {
            wrap.z = 1;
            periodic.z = 1;
            }
        else if (my_pos.z == 0 && (adj & mask_down))
            {
            wrap.z = -1;
            periodic.z = 1;
            }
        }

    box.setPeriodic(periodic);
    int3 img = make_int3(0, 0, 0);
    if (d_img)
        img = d_img[idx];
    Scalar4 postype = d_postype[idx];
    box.wrap(postype, img, wrap);

    out_pos[buf_idx] = postype;

    if (out_img)
        {
        out_img[buf_idx] = img;
        }
    }

void gpu_exchange_ghosts_pack(unsigned int n_out,
                              const uint2* d_ghost_idx_adj,
                              const unsigned int* d_tag,
                              const Scalar4* d_pos,
                              const int3* d_img,
                              const Scalar4* d_vel,
                              const Scalar* d_charge,
                              const Scalar* d_diameter,
                              const unsigned int* d_body,
                              const Scalar4* d_orientation,
                              unsigned int* d_tag_sendbuf,
                              Scalar4* d_pos_sendbuf,
                              Scalar4* d_vel_sendbuf,
                              Scalar* d_charge_sendbuf,
                              Scalar* d_diameter_sendbuf,
                              unsigned int* d_body_sendbuf,
                              int3* d_img_sendbuf,
                              Scalar4* d_orientation_sendbuf,
                              bool send_tag,
                              bool send_pos,
                              bool send_vel,
                              bool send_charge,
                              bool send_diameter,
                              bool send_body,
                              bool send_image,
                              bool send_orientation,
                              const Index3D& di,
                              uint3 my_pos,
                              const BoxDim& box)
    {
    assert(d_ghost_idx_adj);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_out / block_size + 1;
    if (send_tag)
        {
        assert(d_tag);
        assert(d_tag_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_tag,
                           d_tag_sendbuf);
        }
    if (send_pos)
        {
        assert(d_pos);
        assert(d_pos_sendbuf);
        if (send_image)
            {
            assert(d_img);
            }
        hipLaunchKernelGGL(gpu_pack_wrap_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_pos,
                           d_img,
                           d_pos_sendbuf,
                           send_image ? d_img_sendbuf : 0,
                           di,
                           my_pos,
                           box);
        }
    if (send_vel)
        {
        assert(d_vel);
        assert(d_vel_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_vel,
                           d_vel_sendbuf);
        }
    if (send_charge)
        {
        assert(d_charge);
        assert(d_charge_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_charge,
                           d_charge_sendbuf);
        }
    if (send_diameter)
        {
        assert(d_diameter);
        assert(d_diameter_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_diameter,
                           d_diameter_sendbuf);
        }
    if (send_body)
        {
        assert(d_body);
        assert(d_body_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_body,
                           d_body_sendbuf);
        }
    if (send_orientation)
        {
        assert(d_orientation);
        assert(d_orientation_sendbuf);
        hipLaunchKernelGGL(gpu_pack_kernel,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_out,
                           d_ghost_idx_adj,
                           d_orientation,
                           d_orientation_sendbuf);
        }
    }

void gpu_exchange_ghosts_pack_netforce(unsigned int n_out,
                                       const uint2* d_ghost_idx_adj,
                                       const Scalar4* d_netforce,
                                       Scalar4* d_netforce_sendbuf)
    {
    assert(d_ghost_idx_adj);
    assert(d_netforce);
    assert(d_netforce_sendbuf);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_out / block_size + 1;
    hipLaunchKernelGGL(gpu_pack_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_out,
                       d_ghost_idx_adj,
                       d_netforce,
                       d_netforce_sendbuf);
    }

__global__ void gpu_pack_netvirial_kernel(unsigned int n_out,
                                          const uint2* d_ghost_idx_adj,
                                          const Scalar* in,
                                          Scalar* out,
                                          unsigned int pitch_in)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_out)
        return;
    unsigned int idx = d_ghost_idx_adj[buf_idx].x;
    out[6 * buf_idx + 0] = in[idx + 0 * pitch_in];
    out[6 * buf_idx + 1] = in[idx + 1 * pitch_in];
    out[6 * buf_idx + 2] = in[idx + 2 * pitch_in];
    out[6 * buf_idx + 3] = in[idx + 3 * pitch_in];
    out[6 * buf_idx + 4] = in[idx + 4 * pitch_in];
    out[6 * buf_idx + 5] = in[idx + 5 * pitch_in];
    }

void gpu_exchange_ghosts_pack_netvirial(unsigned int n_out,
                                        const uint2* d_ghost_idx_adj,
                                        const Scalar* d_netvirial,
                                        Scalar* d_netvirial_sendbuf,
                                        unsigned int pitch_in)
    {
    assert(d_ghost_idx_adj);
    assert(d_netvirial);
    assert(d_netvirial_sendbuf);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_out / block_size + 1;
    hipLaunchKernelGGL(gpu_pack_netvirial_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_out,
                       d_ghost_idx_adj,
                       d_netvirial,
                       d_netvirial_sendbuf,
                       pitch_in);
    }

template<class members_t, class ranks_t, class group_element_t>
__global__ void gpu_group_pack_kernel(unsigned int n_out,
                                      const uint2* d_ghost_idx_adj,
                                      const unsigned int* d_group_tag,
                                      const members_t* d_groups,
                                      const typeval_union* d_group_typeval,
                                      const ranks_t* d_group_ranks,
                                      group_element_t* d_groups_sendbuf)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_out)
        return;

    unsigned int idx = d_ghost_idx_adj[buf_idx].x;

    group_element_t el;
    el.tags = d_groups[idx];
    el.group_tag = d_group_tag[idx];
    el.typeval = d_group_typeval[idx];
    el.ranks = d_group_ranks[idx];

    d_groups_sendbuf[buf_idx] = el;
    }

template<class members_t, class ranks_t, class group_element_t>
void gpu_exchange_ghost_groups_pack(unsigned int n_out,
                                    const uint2* d_ghost_idx_adj,
                                    const unsigned int* d_group_tag,
                                    const members_t* d_groups,
                                    const typeval_union* d_group_typeval,
                                    const ranks_t* d_group_ranks,
                                    group_element_t* d_groups_sendbuf)
    {
    assert(d_ghost_idx_adj);
    assert(d_group_tag);
    assert(d_groups);
    assert(d_group_typeval);
    assert(d_group_ranks);
    assert(d_groups_sendbuf);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_out / block_size + 1;

    hipLaunchKernelGGL(gpu_group_pack_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_out,
                       d_ghost_idx_adj,
                       d_group_tag,
                       d_groups,
                       d_group_typeval,
                       d_group_ranks,
                       d_groups_sendbuf);
    }

template<typename T> __global__ void gpu_unpack_kernel(unsigned int n_in, const T* in, T* out)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_in)
        return;
    out[buf_idx] = in[buf_idx];
    }

void gpu_exchange_ghosts_copy_buf(unsigned int n_recv,
                                  const unsigned int* d_tag_recvbuf,
                                  const Scalar4* d_pos_recvbuf,
                                  const Scalar4* d_vel_recvbuf,
                                  const Scalar* d_charge_recvbuf,
                                  const Scalar* d_diameter_recvbuf,
                                  const unsigned int* d_body_recvbuf,
                                  const int3* d_image_recvbuf,
                                  const Scalar4* d_orientation_recvbuf,
                                  unsigned int* d_tag,
                                  Scalar4* d_pos,
                                  Scalar4* d_vel,
                                  Scalar* d_charge,
                                  Scalar* d_diameter,
                                  unsigned int* d_body,
                                  int3* d_image,
                                  Scalar4* d_orientation,
                                  bool send_tag,
                                  bool send_pos,
                                  bool send_vel,
                                  bool send_charge,
                                  bool send_diameter,
                                  bool send_body,
                                  bool send_image,
                                  bool send_orientation)
    {
    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;
    if (send_tag)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<unsigned int>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_tag_recvbuf,
                           d_tag);
        }
    if (send_pos)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar4>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_pos_recvbuf,
                           d_pos);
        }
    if (send_vel)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar4>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_vel_recvbuf,
                           d_vel);
        }
    if (send_charge)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_charge_recvbuf,
                           d_charge);
        }
    if (send_diameter)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_diameter_recvbuf,
                           d_diameter);
        }
    if (send_body)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<unsigned int>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_body_recvbuf,
                           d_body);
        }
    if (send_image)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<int3>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_image_recvbuf,
                           d_image);
        }
    if (send_orientation)
        {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar4>),
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_recv,
                           d_orientation_recvbuf,
                           d_orientation);
        }
    }

void gpu_exchange_ghosts_copy_netforce_buf(unsigned int n_recv,
                                           const Scalar4* d_netforce_recvbuf,
                                           Scalar4* d_netforce)
    {
    assert(d_netforce_recvbuf);
    assert(d_netforce);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_unpack_kernel<Scalar4>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_recv,
                       d_netforce_recvbuf,
                       d_netforce);
    }

__global__ void gpu_unpack_netvirial_kernel(unsigned int n_in,
                                            const Scalar* in,
                                            Scalar* out,
                                            unsigned int pitch_out)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= n_in)
        return;
    out[buf_idx + 0 * pitch_out] = in[6 * buf_idx + 0];
    out[buf_idx + 1 * pitch_out] = in[6 * buf_idx + 1];
    out[buf_idx + 2 * pitch_out] = in[6 * buf_idx + 2];
    out[buf_idx + 3 * pitch_out] = in[6 * buf_idx + 3];
    out[buf_idx + 4 * pitch_out] = in[6 * buf_idx + 4];
    out[buf_idx + 5 * pitch_out] = in[6 * buf_idx + 5];
    }

void gpu_exchange_ghosts_copy_netvirial_buf(unsigned int n_recv,
                                            const Scalar* d_netvirial_recvbuf,
                                            Scalar* d_netvirial,
                                            unsigned int pitch_out)
    {
    assert(d_netvirial_recvbuf);
    assert(d_netvirial);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;
    hipLaunchKernelGGL(gpu_unpack_netvirial_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_recv,
                       d_netvirial_recvbuf,
                       d_netvirial,
                       pitch_out);
    }

template<class members_t, class ranks_t, class group_element_t>
__global__ void gpu_unpack_groups_kernel(unsigned int nrecv,
                                         const group_element_t* d_groups_recvbuf,
                                         unsigned int* d_group_tag,
                                         members_t* d_groups,
                                         typeval_union* d_group_typeval,
                                         ranks_t* d_group_ranks,
                                         const unsigned int* d_keep,
                                         unsigned int* d_scan)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= nrecv)
        return;

    if (d_keep[buf_idx])
        {
        group_element_t el = d_groups_recvbuf[buf_idx];

        unsigned int out_idx = d_scan[buf_idx];

        d_group_tag[out_idx] = el.group_tag;
        d_groups[out_idx] = el.tags;
        d_group_typeval[out_idx] = el.typeval;
        d_group_ranks[out_idx] = el.ranks;
        }
    }

template<unsigned int size, class members_t, class group_element_t>
__global__ void gpu_mark_received_ghost_groups_kernel(unsigned int nrecv,
                                                      const group_element_t* d_groups_recvbuf,
                                                      unsigned int* d_group_tag,
                                                      members_t* d_groups,
                                                      unsigned int* d_keep,
                                                      const unsigned int* d_group_rtag,
                                                      const unsigned int* d_rtag,
                                                      unsigned int max_n_local)
    {
    unsigned int buf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (buf_idx >= nrecv)
        return;

    group_element_t el = d_groups_recvbuf[buf_idx];

    unsigned int keep = 0;

    unsigned int group_tag = el.group_tag;
    if (d_group_rtag[group_tag] == GROUP_NOT_LOCAL)
        {
        bool has_nonlocal_members = false;
        for (unsigned int j = 0; j < size; ++j)
            {
            unsigned int tag = el.tags.tag[j];
            if (d_rtag[tag] >= max_n_local)
                {
                has_nonlocal_members = true;
                break;
                }
            }

        if (!has_nonlocal_members)
            {
            keep = 1;
            }
        }

    d_keep[buf_idx] = keep;
    }

template<unsigned int size, class members_t, class ranks_t, class group_element_t>
void gpu_exchange_ghost_groups_copy_buf(unsigned int nrecv,
                                        const group_element_t* d_groups_recvbuf,
                                        unsigned int* d_group_tag,
                                        members_t* d_groups,
                                        typeval_union* d_group_typeval,
                                        ranks_t* d_group_ranks,
                                        unsigned int* d_keep,
                                        unsigned int* d_scan,
                                        const unsigned int* d_group_rtag,
                                        const unsigned int* d_rtag,
                                        unsigned int max_n_local,
                                        unsigned int& n_keep,
                                        CachedAllocator& alloc)
    {
    n_keep = 0;
    if (!nrecv)
        return;

    assert(d_groups_recvbuf);
    assert(d_group_tag);
    assert(d_groups);
    assert(d_keep);
    assert(d_group_rtag);
    assert(d_rtag);

    unsigned int block_size = 256;
    unsigned int n_blocks = nrecv / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_mark_received_ghost_groups_kernel<size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       nrecv,
                       d_groups_recvbuf,
                       d_group_tag,
                       d_groups,
                       d_keep,
                       d_group_rtag,
                       d_rtag,
                       max_n_local);

    assert(d_scan);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_keep, d_scan, nrecv);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_keep, d_scan, nrecv);
    alloc.deallocate((char*)d_temp_storage);

    // determine total number of received groups
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_n_keep = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_n_keep);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_keep, d_n_keep, nrecv);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_keep, d_n_keep, nrecv);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&n_keep, d_n_keep, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_n_keep);

    hipLaunchKernelGGL(gpu_unpack_groups_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       nrecv,
                       d_groups_recvbuf,
                       d_group_tag,
                       d_groups,
                       d_group_typeval,
                       d_group_ranks,
                       d_keep,
                       d_scan);
    }

void gpu_compute_ghost_rtags(unsigned int first_idx,
                             unsigned int n_ghost,
                             const unsigned int* d_tag,
                             unsigned int* d_rtag)
    {
    assert(d_tag);
    assert(d_rtag);

    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::counting_iterator<unsigned int> idx(first_idx);
    thrust::scatter(idx, idx + n_ghost, tag_ptr, rtag_ptr);
    }

/*!
 * Routines for communication of bonded groups
 */
template<unsigned int group_size, typename group_t, typename ranks_t>
__global__ void gpu_mark_groups_kernel(unsigned int N,
                                       const unsigned int* d_comm_flags,
                                       unsigned int n_groups,
                                       const group_t* d_members,
                                       ranks_t* d_group_ranks,
                                       unsigned int* d_rank_mask,
                                       unsigned int* d_scan,
                                       const unsigned int* d_rtag,
                                       const Index3D di,
                                       uint3 my_pos,
                                       const unsigned* d_cart_ranks,
                                       bool incomplete)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_groups)
        return;

    // Load group
    group_t g = d_members[group_idx];

    ranks_t r = d_group_ranks[group_idx];

    // initialize bit field
    unsigned int mask = 0;
    unsigned int my_rank = d_cart_ranks[di(my_pos.x, my_pos.y, my_pos.z)];

    bool update = false;

    unsigned int gs = group_size;

    // loop through members of group
    for (unsigned int i = 0; i < gs; ++i)
        {
        unsigned int tag = g.tag[i];
        unsigned int pidx = d_rtag[tag];

        if (pidx == NOT_LOCAL)
            {
            // if any ptl is non-local, send
            update = true;
            }
        else // local ptl
            {
            if (incomplete)
                {
                // initially, update rank information for all local ptls
                r.idx[i] = my_rank;
                mask |= (1 << i);
                }

            unsigned int flags = d_comm_flags[pidx];

            if (flags)
                {
                // the local particle is going to be sent to a different domain
                mask |= (1 << i);

                // parse communication flags
                int ix, iy, iz;
                ix = iy = iz = 0;

                if (flags & send_east)
                    ix = 1;
                else if (flags & send_west)
                    ix = -1;

                if (flags & send_north)
                    iy = 1;
                else if (flags & send_south)
                    iy = -1;

                if (flags & send_up)
                    iz = 1;
                else if (flags & send_down)
                    iz = -1;

                int ni = my_pos.x;
                int nj = my_pos.y;
                int nk = my_pos.z;

                ni += ix;
                if (ni == (int)di.getW())
                    ni = 0;
                else if (ni < 0)
                    ni += di.getW();

                nj += iy;
                if (nj == (int)di.getH())
                    nj = 0;
                else if (nj < 0)
                    nj += di.getH();

                nk += iz;
                if (nk == (int)di.getD())
                    nk = 0;
                else if (nk < 0)
                    nk += di.getD();

                // update ranks information
                r.idx[i] = d_cart_ranks[di(ni, nj, nk)];

                // a local ptl has changed place, send ranks information
                update = true;
                }
            }
        } // end for

    // write out ranks
    d_group_ranks[group_idx] = r;

    // if group is purely local do not send
    if (!update)
        mask = 0;

    // write out bitmask
    d_rank_mask[group_idx] = mask;

    // set zero-one input for scan
    d_scan[group_idx] = mask ? 1 : 0;
    }

/*! \param N Number of particles
    \param d_comm_flags Array of communication flags
    \param n_groups Number of local groups
    \param d_members Array of group member tags
    \param d_group_tag Array of group tags
    \param d_group_rtag Array of group rtags
    \param d_group_ranks Auxiliary array of group member ranks
    \param d_rtag Particle data reverse-lookup table for tags
    \param di Domain decomposition indexer
    \param my_pos Integer triple of domain coordinates
    \param incomplete If true, initially update auxiliary rank information
 */
template<unsigned int group_size, typename group_t, typename ranks_t>
void gpu_mark_groups(unsigned int N,
                     const unsigned int* d_comm_flags,
                     unsigned int n_groups,
                     const group_t* d_members,
                     ranks_t* d_group_ranks,
                     unsigned int* d_rank_mask,
                     const unsigned int* d_rtag,
                     unsigned int* d_marked_groups,
                     unsigned int* d_scan,
                     unsigned int& n_out,
                     const Index3D di,
                     uint3 my_pos,
                     const unsigned int* d_cart_ranks,
                     bool incomplete,
                     CachedAllocator& alloc)
    {
    n_out = 0;
    if (!n_groups)
        return;

    assert(d_comm_flags);
    assert(d_members);
    assert(d_group_ranks);
    assert(d_rank_mask);
    assert(d_rtag);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_groups / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_mark_groups_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_comm_flags,
                       n_groups,
                       d_members,
                       d_group_ranks,
                       d_rank_mask,
                       d_marked_groups,
                       d_rtag,
                       di,
                       my_pos,
                       d_cart_ranks,
                       incomplete);

    // scan over marked groups
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_scan,
                                     n_groups);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_scan,
                                     n_groups);
    alloc.deallocate((char*)d_temp_storage);

    // determine total number of sent groups
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_n_out = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_n_out);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_n_out,
                              n_groups);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_n_out,
                              n_groups);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&n_out, d_n_out, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_n_out);
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename rank_element_t>
__global__ void gpu_scatter_ranks_and_mark_send_groups_kernel(unsigned int n_groups,
                                                              const unsigned int* d_group_tag,
                                                              const ranks_t* d_group_ranks,
                                                              unsigned int* d_rank_mask,
                                                              const group_t* d_groups,
                                                              const unsigned int* d_rtag,
                                                              const unsigned int* d_comm_flags,
                                                              unsigned int* d_marked_send_groups,
                                                              const unsigned int* d_scan,
                                                              rank_element_t* d_out_ranks)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_groups)
        return;

    unsigned int mask = d_rank_mask[group_idx];

    // determine if rank information needs to be sent
    if (mask)
        {
        unsigned int out_idx = d_scan[group_idx];
        rank_element_t el;
        el.ranks = d_group_ranks[group_idx];
        el.mask = mask;
        el.tag = d_group_tag[group_idx];
        d_out_ranks[out_idx] = el;
        }

    // determine if whole group needs to be sent
    group_t members = d_groups[group_idx];

    unsigned int gs = group_size;

    mask = 0;
    for (unsigned int i = 0; i < gs; ++i)
        {
        unsigned int tag = members.tag[i];
        unsigned int pidx = d_rtag[tag];

        // are the communication flags set for this member particle?
        if (pidx != NOT_LOCAL && d_comm_flags[pidx])
            mask |= (1 << i);
        }

    // output to 0-1 array
    d_marked_send_groups[group_idx] = mask ? 1 : 0;
    d_rank_mask[group_idx] = mask;
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename rank_element_t>
void gpu_scatter_ranks_and_mark_send_groups(unsigned int n_groups,
                                            const unsigned int* d_group_tag,
                                            const ranks_t* d_group_ranks,
                                            unsigned int* d_rank_mask,
                                            const group_t* d_groups,
                                            const unsigned int* d_rtag,
                                            const unsigned int* d_comm_flags,
                                            unsigned int* d_marked_send_groups,
                                            unsigned int* d_scan,
                                            unsigned int& n_send,
                                            rank_element_t* d_out_ranks,
                                            CachedAllocator& alloc)
    {
    n_send = 0;
    if (!n_groups)
        return;

    assert(d_group_tag);
    assert(d_group_ranks);
    assert(d_rank_mask);
    assert(d_groups);
    assert(d_rtag);
    assert(d_comm_flags);
    assert(d_marked_send_groups);
    assert(d_scan);
    assert(d_out_ranks);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_groups / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_scatter_ranks_and_mark_send_groups_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_groups,
                       d_group_tag,
                       d_group_ranks,
                       d_rank_mask,
                       d_groups,
                       d_rtag,
                       d_comm_flags,
                       d_marked_send_groups,
                       d_scan,
                       d_out_ranks);

    // scan over groups marked for sending
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_send_groups,
                                     d_scan,
                                     n_groups);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_send_groups,
                                     d_scan,
                                     n_groups);
    alloc.deallocate((char*)d_temp_storage);

    // determine total number of sent groups
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_n_send = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_n_send);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_send_groups,
                              d_n_send,
                              n_groups);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_send_groups,
                              d_n_send,
                              n_groups);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&n_send, d_n_send, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_n_send);
    }

template<unsigned int group_size, typename ranks_t, typename rank_element_t>
__global__ void gpu_update_ranks_table_kernel(unsigned int n_groups,
                                              ranks_t* d_group_ranks,
                                              unsigned int* d_group_rtag,
                                              unsigned int n_recv,
                                              const rank_element_t* d_ranks_recvbuf)
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv)
        return;

    rank_element_t el = d_ranks_recvbuf[recv_idx];
    unsigned int tag = el.tag;
    unsigned int gidx = d_group_rtag[tag];

    if (gidx != GROUP_NOT_LOCAL)
        {
        ranks_t new_ranks = el.ranks;
        unsigned int mask = el.mask;

        unsigned int gs = group_size;

        for (unsigned int i = 0; i < gs; ++i)
            {
            bool update = mask & (1 << i);

            if (update)
                //                d_group_ranks[gidx].idx[i] = new_ranks.idx[i];
                atomicExch(&d_group_ranks[gidx].idx[i], new_ranks.idx[i]);
            }
        }
    }

template<unsigned int group_size, typename ranks_t, typename rank_element_t>
void gpu_update_ranks_table(unsigned int n_groups,
                            ranks_t* d_group_ranks,
                            unsigned int* d_group_rtag,
                            unsigned int n_recv,
                            const rank_element_t* d_ranks_recvbuf)
    {
    assert(d_group_ranks);
    assert(d_group_rtag);
    assert(d_ranks_recvbuf);
    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_update_ranks_table_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_groups,
                       d_group_ranks,
                       d_group_rtag,
                       n_recv,
                       d_ranks_recvbuf);
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename packed_t>
__global__ void gpu_scatter_and_mark_groups_for_removal_kernel(unsigned int n_groups,
                                                               const group_t* d_groups,
                                                               const typeval_union* d_group_typeval,
                                                               const unsigned int* d_group_tag,
                                                               unsigned int* d_group_rtag,
                                                               const ranks_t* d_group_ranks,
                                                               unsigned int* d_rank_mask,
                                                               const unsigned int* d_rtag,
                                                               const unsigned int* d_comm_flags,
                                                               unsigned int my_rank,
                                                               const unsigned int* d_scan,
                                                               unsigned int* d_marked_groups,
                                                               packed_t* d_out_groups,
                                                               unsigned int* d_out_rank_mask,
                                                               bool local_multiple)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_groups)
        return;

    unsigned int mask = d_rank_mask[group_idx];
    unsigned int flag = 1;

    // are we sending this group?
    if (mask)
        {
        unsigned int out_idx = d_scan[group_idx];

        packed_t el;
        el.tags = d_groups[group_idx];
        el.typeval = d_group_typeval[group_idx];
        el.group_tag = d_group_tag[group_idx];
        el.ranks = d_group_ranks[group_idx];
        d_out_groups[out_idx] = el;
        d_out_rank_mask[out_idx] = mask;

        unsigned int gs = group_size;

        // determine if the group still has any local ptls
        bool is_local = false;
        for (unsigned int i = 0; i < gs; ++i)
            {
            unsigned int tag = el.tags.tag[i];
            unsigned int pidx = d_rtag[tag];
            if (pidx != NOT_LOCAL && !d_comm_flags[pidx])
                {
                if (local_multiple || i == 0)
                    {
                    is_local = true;
                    }
                }
            }

        // if group is no longer local, flag for removal
        if (!is_local)
            {
            d_group_rtag[el.group_tag] = GROUP_NOT_LOCAL;
            flag = 0;
            }
        }

    // update zero-one array (zero == remove group, one == retain group)
    d_marked_groups[group_idx] = flag;
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename packed_t>
void gpu_scatter_and_mark_groups_for_removal(unsigned int n_groups,
                                             const group_t* d_groups,
                                             const typeval_union* d_group_typeval,
                                             const unsigned int* d_group_tag,
                                             unsigned int* d_group_rtag,
                                             const ranks_t* d_group_ranks,
                                             unsigned int* d_rank_mask,
                                             const unsigned int* d_rtag,
                                             const unsigned int* d_comm_flags,
                                             unsigned int my_rank,
                                             const unsigned int* d_scan,
                                             unsigned int* d_marked_groups,
                                             packed_t* d_out_groups,
                                             unsigned int* d_out_rank_mask,
                                             bool local_multiple)
    {
    assert(d_groups);
    assert(d_group_typeval);
    assert(d_group_tag);
    assert(d_group_rtag);
    assert(d_group_ranks);
    assert(d_rank_mask);
    assert(d_rtag);
    assert(d_comm_flags);
    assert(d_scan);
    assert(d_marked_groups);
    assert(d_out_groups);
    assert(d_out_rank_mask);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_groups / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_scatter_and_mark_groups_for_removal_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_groups,
                       d_groups,
                       d_group_typeval,
                       d_group_tag,
                       d_group_rtag,
                       d_group_ranks,
                       d_rank_mask,
                       d_rtag,
                       d_comm_flags,
                       my_rank,
                       d_scan,
                       d_marked_groups,
                       d_out_groups,
                       d_out_rank_mask,
                       local_multiple);
    }

template<typename group_t, typename ranks_t>
__global__ void gpu_remove_groups_kernel(unsigned int n_groups,
                                         const group_t* d_groups,
                                         group_t* d_groups_alt,
                                         const typeval_union* d_group_typeval,
                                         typeval_union* d_group_typeval_alt,
                                         const unsigned int* d_group_tag,
                                         unsigned int* d_group_tag_alt,
                                         const ranks_t* d_group_ranks,
                                         ranks_t* d_group_ranks_alt,
                                         unsigned int* d_group_rtag,
                                         const unsigned int* d_scan)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_groups)
        return;

    unsigned int group_tag = d_group_tag[group_idx];
    bool keep = (d_group_rtag[group_tag] != GROUP_NOT_LOCAL);

    unsigned int out_idx = d_scan[group_idx];

    if (keep)
        {
        // scatter into output array
        d_groups_alt[out_idx] = d_groups[group_idx];
        d_group_typeval_alt[out_idx] = d_group_typeval[group_idx];
        d_group_tag_alt[out_idx] = group_tag;
        d_group_ranks_alt[out_idx] = d_group_ranks[group_idx];

        // rebuild rtags
        d_group_rtag[group_tag] = out_idx;
        }
    }

template<typename group_t, typename ranks_t>
void gpu_remove_groups(unsigned int n_groups,
                       const group_t* d_groups,
                       group_t* d_groups_alt,
                       const typeval_union* d_group_typeval,
                       typeval_union* d_group_typeval_alt,
                       const unsigned int* d_group_tag,
                       unsigned int* d_group_tag_alt,
                       const ranks_t* d_group_ranks,
                       ranks_t* d_group_ranks_alt,
                       unsigned int* d_group_rtag,
                       unsigned int& new_ngroups,
                       const unsigned int* d_marked_groups,
                       unsigned int* d_scan,
                       CachedAllocator& alloc)
    {
    new_ngroups = 0;
    if (!n_groups)
        return;

    assert(d_groups);
    assert(d_groups_alt);
    assert(d_group_typeval);
    assert(d_group_typeval_alt);
    assert(d_group_tag);
    assert(d_group_tag_alt);
    assert(d_group_ranks);
    assert(d_group_ranks_alt);
    assert(d_group_rtag);
    assert(d_marked_groups);
    assert(d_scan);

    // scan over marked groups
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_scan,
                                     n_groups);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_scan,
                                     n_groups);
    alloc.deallocate((char*)d_temp_storage);

    // determine new_ngroups number of ghosts
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_new_ngroups = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_new_ngroups);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_new_ngroups,
                              n_groups);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_new_ngroups,
                              n_groups);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&new_ngroups, d_new_ngroups, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_new_ngroups);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_groups / block_size + 1;

    hipLaunchKernelGGL(gpu_remove_groups_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_groups,
                       d_groups,
                       d_groups_alt,
                       d_group_typeval,
                       d_group_typeval_alt,
                       d_group_tag,
                       d_group_tag_alt,
                       d_group_ranks,
                       d_group_ranks_alt,
                       d_group_rtag,
                       d_scan);
    }

template<typename packed_t>
__global__ void gpu_count_unique_groups_kernel(unsigned int n_recv,
                                               const packed_t* d_groups_in,
                                               const unsigned int* d_group_rtag,
                                               unsigned int* d_scan,
                                               bool local_multiple,
                                               unsigned int myrank)
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv)
        return;

    packed_t el = d_groups_in[recv_idx];

    unsigned int rtag = d_group_rtag[el.group_tag];

    bool remove = false;
    if (!local_multiple)
        {
        // only add if we own the first particle
        if (el.ranks.idx[0] != myrank)
            {
            remove = true;
            }
        }

    // write out zero-one array
    d_scan[recv_idx] = (!remove && rtag == GROUP_NOT_LOCAL) ? 1 : 0;
    }

template<typename packed_t, typename group_t, typename ranks_t>
__global__ void gpu_add_groups_kernel(unsigned int n_recv,
                                      unsigned int n_groups,
                                      const packed_t* d_groups_in,
                                      const unsigned int* d_scan,
                                      group_t* d_groups,
                                      typeval_union* d_group_typeval,
                                      unsigned int* d_group_tag,
                                      ranks_t* d_group_ranks,
                                      unsigned int* d_group_rtag,
                                      bool local_multiple,
                                      unsigned int myrank)
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv)
        return;

    packed_t el = d_groups_in[recv_idx];

    bool remove = false;
    if (!local_multiple)
        {
        // only add if we own the first particle
        if (el.ranks.idx[0] != myrank)
            {
            remove = true;
            }
        }

    unsigned int tag = el.group_tag;
    unsigned int rtag = d_group_rtag[tag];
    if (!remove && rtag == GROUP_NOT_LOCAL)
        {
        unsigned int add_idx = n_groups + d_scan[recv_idx];

        d_groups[add_idx] = el.tags;
        d_group_typeval[add_idx] = el.typeval;
        d_group_tag[add_idx] = tag;
        d_group_ranks[add_idx] = el.ranks;

        // update reverse-lookup table
        d_group_rtag[tag] = add_idx;
        }
    }

template<typename packed_t, typename group_t, typename ranks_t>
void gpu_add_groups(unsigned int n_groups,
                    unsigned int n_recv,
                    const packed_t* d_groups_in,
                    group_t* d_groups,
                    typeval_union* d_group_typeval,
                    unsigned int* d_group_tag,
                    ranks_t* d_group_ranks,
                    unsigned int* d_group_rtag,
                    unsigned int& new_ngroups,
                    unsigned int* d_marked_groups,
                    unsigned int* d_tmp,
                    bool local_multiple,
                    unsigned int myrank,
                    CachedAllocator& alloc)
    {
    new_ngroups = n_groups;
    if (!n_recv)
        return;

    assert(d_groups_in);
    assert(d_groups);
    assert(d_group_typeval);
    assert(d_group_tag);
    assert(d_group_rtag);
    assert(d_marked_groups);
    assert(d_tmp);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_recv / block_size + 1;

    // update locally existing groups
    hipLaunchKernelGGL(gpu_count_unique_groups_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_recv,
                       d_groups_in,
                       d_group_rtag,
                       d_marked_groups,
                       local_multiple,
                       myrank);

    unsigned int n_unique;

    // scan over input groups, select those which are not already local
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_tmp,
                                     n_recv);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_marked_groups,
                                     d_tmp,
                                     n_recv);
    alloc.deallocate((char*)d_temp_storage);

    // determine n_unique number of ghosts
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_n_unique = alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_n_unique);

    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_n_unique,
                              n_recv);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage,
                              temp_storage_bytes,
                              d_marked_groups,
                              d_n_unique,
                              n_recv);
    alloc.deallocate((char*)d_temp_storage);

    hipMemcpy(&n_unique, d_n_unique, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_n_unique);

    new_ngroups += n_unique;

    // add new groups at the end
    hipLaunchKernelGGL(gpu_add_groups_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_recv,
                       n_groups,
                       d_groups_in,
                       d_tmp,
                       d_groups,
                       d_group_typeval,
                       d_group_tag,
                       d_group_ranks,
                       d_group_rtag,
                       local_multiple,
                       myrank);
    }

template<unsigned int group_size, typename members_t, typename ranks_t>
__global__ void gpu_mark_bonded_ghosts_kernel(unsigned int n_groups,
                                              members_t* d_groups,
                                              ranks_t* d_ranks,
                                              const Scalar4* d_postype,
                                              const BoxDim box,
                                              const unsigned int* d_rtag,
                                              unsigned int* d_plan,
                                              Index3D di,
                                              uint3 my_pos,
                                              const unsigned int* d_cart_ranks_inv,
                                              unsigned int my_rank,
                                              unsigned int mask)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_groups)
        return;

    // load group member tags
    members_t g = d_groups[group_idx];

    // load group member ranks
    ranks_t r = d_ranks[group_idx];

    unsigned int gs = group_size;

    for (unsigned int i = 0; i < gs; ++i)
        {
        unsigned int rank = r.idx[i];

        if (rank != my_rank)
            {
            // incomplete group

            // send group to neighbor rank stored for that member
            uint3 neigh_pos = di.getTriple(d_cart_ranks_inv[rank]);

            // only neighbors are considered for communication
            unsigned int flags = 0;
            if (neigh_pos.x == my_pos.x + 1 || (my_pos.x == di.getW() - 1 && neigh_pos.x == 0))
                flags |= send_east;
            if (neigh_pos.x == my_pos.x - 1 || (my_pos.x == 0 && neigh_pos.x == di.getW() - 1))
                flags |= send_west;
            if (neigh_pos.y == my_pos.y + 1 || (my_pos.y == di.getH() - 1 && neigh_pos.y == 0))
                flags |= send_north;
            if (neigh_pos.y == my_pos.y - 1 || (my_pos.y == 0 && neigh_pos.y == di.getH() - 1))
                flags |= send_south;
            if (neigh_pos.z == my_pos.z + 1 || (my_pos.z == di.getD() - 1 && neigh_pos.z == 0))
                flags |= send_up;
            if (neigh_pos.z == my_pos.z - 1 || (my_pos.z == 0 && neigh_pos.z == di.getD() - 1))
                flags |= send_down;

            flags &= mask;

            // Send all local members of the group to this neighbor
            for (unsigned int j = 0; j < gs; ++j)
                {
                unsigned int tag_j = g.tag[j];
                unsigned int rtag_j = d_rtag[tag_j];

                if (rtag_j != NOT_LOCAL)
                    {
                    // disambiguate between positive and negative directions
                    // based on position (this is necessary for boundary conditions
                    // to be applied correctly)
                    if (flags & send_east && flags & send_west)
                        {
                        Scalar4 postype = d_postype[rtag_j];
                        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                        Scalar3 f = box.makeFraction(pos);
                        // remove one of the flags
                        flags &= ~(f.x > Scalar(0.5) ? send_west : send_east);
                        }
                    if (flags & send_north && flags & send_south)
                        {
                        Scalar4 postype = d_postype[rtag_j];
                        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                        Scalar3 f = box.makeFraction(pos);
                        // remove one of the flags
                        flags &= ~(f.y > Scalar(0.5) ? send_south : send_north);
                        }
                    if (flags & send_up && flags & send_down)
                        {
                        Scalar4 postype = d_postype[rtag_j];
                        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                        Scalar3 f = box.makeFraction(pos);
                        // remove one of the flags
                        flags &= ~(f.z > Scalar(0.5) ? send_down : send_up);
                        }

                    // set ghost plans
                    atomicOr(&d_plan[rtag_j], flags);
                    }
                }
            }
        }
    }

template<unsigned int group_size, typename members_t, typename ranks_t>
void gpu_mark_bonded_ghosts(unsigned int n_groups,
                            members_t* d_groups,
                            ranks_t* d_ranks,
                            const Scalar4* d_postype,
                            const BoxDim& box,
                            const unsigned int* d_rtag,
                            unsigned int* d_plan,
                            Index3D& di,
                            uint3 my_pos,
                            const unsigned int* d_cart_ranks_inv,
                            unsigned int my_rank,
                            unsigned int mask)
    {
    assert(d_groups);
    assert(d_ranks);
    assert(d_postype);
    assert(d_rtag);
    assert(d_plan);
    assert(d_cart_ranks_inv);

    unsigned int block_size = 256;
    unsigned int n_blocks = n_groups / block_size + 1;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_mark_bonded_ghosts_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       n_groups,
                       d_groups,
                       d_ranks,
                       d_postype,
                       box,
                       d_rtag,
                       d_plan,
                       di,
                       my_pos,
                       d_cart_ranks_inv,
                       my_rank,
                       mask);
    }

void gpu_reset_exchange_plan(unsigned int N, unsigned int* d_plan)
    {
    hipMemsetAsync(d_plan, 0, sizeof(unsigned int) * N);
    }
/*
 *! Explicit template instantiations for BondData (n=2)
 */

template void
gpu_mark_groups<2, group_storage<2>, group_storage<2>>(unsigned int N,
                                                       const unsigned int* d_comm_flags,
                                                       unsigned int n_groups,
                                                       const group_storage<2>* d_members,
                                                       group_storage<2>* d_group_ranks,
                                                       unsigned int* d_rank_mask,
                                                       const unsigned int* d_rtag,
                                                       unsigned int* d_marked_groups,
                                                       unsigned int* d_scan,
                                                       unsigned int& n_out,
                                                       const Index3D di,
                                                       uint3 my_pos,
                                                       const unsigned int* d_cart_ranks,
                                                       bool incomplete,
                                                       CachedAllocator& alloc);

template void gpu_scatter_ranks_and_mark_send_groups<2>(unsigned int n_groups,
                                                        const unsigned int* d_group_tag,
                                                        const group_storage<2>* d_group_ranks,
                                                        unsigned int* d_rank_mask,
                                                        const group_storage<2>* d_groups,
                                                        const unsigned int* d_rtag,
                                                        const unsigned int* d_comm_flags,
                                                        unsigned int* d_marked_send_groups,
                                                        unsigned int* d_scan,
                                                        unsigned int& n_send,
                                                        rank_element<group_storage<2>>* d_out_ranks,
                                                        CachedAllocator& alloc);

template void gpu_update_ranks_table<2>(unsigned int n_groups,
                                        group_storage<2>* d_group_ranks,
                                        unsigned int* d_group_rtag,
                                        unsigned int n_recv,
                                        const rank_element<group_storage<2>>* d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<2>(unsigned int n_groups,
                                                         const group_storage<2>* d_groups,
                                                         const typeval_union* d_group_typeval,
                                                         const unsigned int* d_group_tag,
                                                         unsigned int* d_group_rtag,
                                                         const group_storage<2>* d_group_ranks,
                                                         unsigned int* d_rank_mask,
                                                         const unsigned int* d_rtag,
                                                         const unsigned int* d_comm_flags,
                                                         unsigned int my_rank,
                                                         const unsigned int* d_scan,
                                                         unsigned int* d_marked_groups,
                                                         packed_storage<2>* d_out_groups,
                                                         unsigned int* d_out_rank_mask,
                                                         bool local_multiple);

template void gpu_remove_groups(unsigned int n_groups,
                                const group_storage<2>* d_groups,
                                group_storage<2>* d_groups_alt,
                                const typeval_union* d_group_typeval,
                                typeval_union* d_group_typeval_alt,
                                const unsigned int* d_group_tag,
                                unsigned int* d_group_tag_alt,
                                const group_storage<2>* d_group_ranks,
                                group_storage<2>* d_group_ranks_alt,
                                unsigned int* d_group_rtag,
                                unsigned int& new_ngroups,
                                const unsigned int* d_marked_groups,
                                unsigned int* d_scan,
                                CachedAllocator& alloc);

template void gpu_add_groups(unsigned int n_groups,
                             unsigned int n_recv,
                             const packed_storage<2>* d_groups_in,
                             group_storage<2>* d_groups,
                             typeval_union* d_group_typeval,
                             unsigned int* d_group_tag,
                             group_storage<2>* d_group_ranks,
                             unsigned int* d_group_rtag,
                             unsigned int& new_ngroups,
                             unsigned int* d_marked_groups,
                             unsigned int* d_tmp,
                             bool local_multiple,
                             unsigned int myrank,
                             CachedAllocator& alloc);

template void gpu_mark_bonded_ghosts<2>(unsigned int n_groups,
                                        group_storage<2>* d_groups,
                                        group_storage<2>* d_ranks,
                                        const Scalar4* d_postype,
                                        const BoxDim& box,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_plan,
                                        Index3D& di,
                                        uint3 my_pos,
                                        const unsigned int* d_cart_ranks,
                                        unsigned int my_rank,
                                        unsigned int mask);

/*
 *! Explicit template instantiations for BondData (n=3)
 */

template void
gpu_mark_groups<3, group_storage<3>, group_storage<3>>(unsigned int N,
                                                       const unsigned int* d_comm_flags,
                                                       unsigned int n_groups,
                                                       const group_storage<3>* d_members,
                                                       group_storage<3>* d_group_ranks,
                                                       unsigned int* d_rank_mask,
                                                       const unsigned int* d_rtag,
                                                       unsigned int* d_marked_groups,
                                                       unsigned int* d_scan,
                                                       unsigned int& n_out,
                                                       const Index3D di,
                                                       uint3 my_pos,
                                                       const unsigned int* d_cart_ranks,
                                                       bool incomplete,
                                                       CachedAllocator& alloc);

template void gpu_scatter_ranks_and_mark_send_groups<3>(unsigned int n_groups,
                                                        const unsigned int* d_group_tag,
                                                        const group_storage<3>* d_group_ranks,
                                                        unsigned int* d_rank_mask,
                                                        const group_storage<3>* d_groups,
                                                        const unsigned int* d_rtag,
                                                        const unsigned int* d_comm_flags,
                                                        unsigned int* d_marked_send_groups,
                                                        unsigned int* d_scan,
                                                        unsigned int& n_send,
                                                        rank_element<group_storage<3>>* d_out_ranks,
                                                        CachedAllocator& alloc);

template void gpu_update_ranks_table<3>(unsigned int n_groups,
                                        group_storage<3>* d_group_ranks,
                                        unsigned int* d_group_rtag,
                                        unsigned int n_recv,
                                        const rank_element<group_storage<3>>* d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<3>(unsigned int n_groups,
                                                         const group_storage<3>* d_groups,
                                                         const typeval_union* d_group_typeval,
                                                         const unsigned int* d_group_tag,
                                                         unsigned int* d_group_rtag,
                                                         const group_storage<3>* d_group_ranks,
                                                         unsigned int* d_rank_mask,
                                                         const unsigned int* d_rtag,
                                                         const unsigned int* d_comm_flags,
                                                         unsigned int my_rank,
                                                         const unsigned int* d_scan,
                                                         unsigned int* d_marked_groups,
                                                         packed_storage<3>* d_out_groups,
                                                         unsigned int* d_out_rank_mask,
                                                         bool local_multiple);

template void gpu_remove_groups(unsigned int n_groups,
                                const group_storage<3>* d_groups,
                                group_storage<3>* d_groups_alt,
                                const typeval_union* d_group_typeval,
                                typeval_union* d_group_typeval_alt,
                                const unsigned int* d_group_tag,
                                unsigned int* d_group_tag_alt,
                                const group_storage<3>* d_group_ranks,
                                group_storage<3>* d_group_ranks_alt,
                                unsigned int* d_group_rtag,
                                unsigned int& new_ngroups,
                                const unsigned int* d_marked_groups,
                                unsigned int* d_scan,
                                CachedAllocator& alloc);

template void gpu_add_groups(unsigned int n_groups,
                             unsigned int n_recv,
                             const packed_storage<3>* d_groups_in,
                             group_storage<3>* d_groups,
                             typeval_union* d_group_typeval,
                             unsigned int* d_group_tag,
                             group_storage<3>* d_group_ranks,
                             unsigned int* d_group_rtag,
                             unsigned int& new_ngroups,
                             unsigned int* d_marked_groups,
                             unsigned int* d_tmp,
                             bool local_multiple,
                             unsigned int myrank,
                             CachedAllocator& alloc);

template void gpu_mark_bonded_ghosts<3>(unsigned int n_groups,
                                        group_storage<3>* d_groups,
                                        group_storage<3>* d_ranks,
                                        const Scalar4* d_postype,
                                        const BoxDim& box,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_plan,
                                        Index3D& di,
                                        uint3 my_pos,
                                        const unsigned int* d_cart_ranks,
                                        unsigned int my_rank,
                                        unsigned int mask);

/*
 *! Explicit template instantiations for DihedralData and ImproperData and MeshBondData (n=4)
 */

template void
gpu_mark_groups<4, group_storage<4>, group_storage<4>>(unsigned int N,
                                                       const unsigned int* d_comm_flags,
                                                       unsigned int n_groups,
                                                       const group_storage<4>* d_members,
                                                       group_storage<4>* d_group_ranks,
                                                       unsigned int* d_rank_mask,
                                                       const unsigned int* d_rtag,
                                                       unsigned int* d_marked_groups,
                                                       unsigned int* d_scan,
                                                       unsigned int& n_out,
                                                       const Index3D di,
                                                       uint3 my_pos,
                                                       const unsigned int* d_cart_ranks,
                                                       bool incomplete,
                                                       CachedAllocator& alloc);

template void gpu_scatter_ranks_and_mark_send_groups<4>(unsigned int n_groups,
                                                        const unsigned int* d_group_tag,
                                                        const group_storage<4>* d_group_ranks,
                                                        unsigned int* d_rank_mask,
                                                        const group_storage<4>* d_groups,
                                                        const unsigned int* d_rtag,
                                                        const unsigned int* d_comm_flags,
                                                        unsigned int* d_marked_send_groups,
                                                        unsigned int* d_scan,
                                                        unsigned int& n_send,
                                                        rank_element<group_storage<4>>* d_out_ranks,
                                                        CachedAllocator& alloc);

template void gpu_update_ranks_table<4>(unsigned int n_groups,
                                        group_storage<4>* d_group_ranks,
                                        unsigned int* d_group_rtag,
                                        unsigned int n_recv,
                                        const rank_element<group_storage<4>>* d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<4>(unsigned int n_groups,
                                                         const group_storage<4>* d_groups,
                                                         const typeval_union* d_group_typeval,
                                                         const unsigned int* d_group_tag,
                                                         unsigned int* d_group_rtag,
                                                         const group_storage<4>* d_group_ranks,
                                                         unsigned int* d_rank_mask,
                                                         const unsigned int* d_rtag,
                                                         const unsigned int* d_comm_flags,
                                                         unsigned int my_rank,
                                                         const unsigned int* d_scan,
                                                         unsigned int* d_marked_groups,
                                                         packed_storage<4>* d_out_groups,
                                                         unsigned int* d_out_rank_mask,
                                                         bool local_multiple);

template void gpu_remove_groups(unsigned int n_groups,
                                const group_storage<4>* d_groups,
                                group_storage<4>* d_groups_alt,
                                const typeval_union* d_group_typeval,
                                typeval_union* d_group_typeval_alt,
                                const unsigned int* d_group_tag,
                                unsigned int* d_group_tag_alt,
                                const group_storage<4>* d_group_ranks,
                                group_storage<4>* d_group_ranks_alt,
                                unsigned int* d_group_rtag,
                                unsigned int& new_ngroups,
                                const unsigned int* d_marked_groups,
                                unsigned int* d_scan,
                                CachedAllocator& alloc);

template void gpu_add_groups(unsigned int n_groups,
                             unsigned int n_recv,
                             const packed_storage<4>* d_groups_in,
                             group_storage<4>* d_groups,
                             typeval_union* d_group_typeval,
                             unsigned int* d_group_tag,
                             group_storage<4>* d_group_ranks,
                             unsigned int* d_group_rtag,
                             unsigned int& new_ngroups,
                             unsigned int* d_marked_groups,
                             unsigned int* d_tmp,
                             bool local_multiple,
                             unsigned int myrank,
                             CachedAllocator& alloc);

template void gpu_mark_bonded_ghosts<4>(unsigned int n_groups,
                                        group_storage<4>* d_groups,
                                        group_storage<4>* d_ranks,
                                        const Scalar4* d_postype,
                                        const BoxDim& box,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_plan,
                                        Index3D& di,
                                        uint3 my_pos,
                                        const unsigned int* d_cart_ranks,
                                        unsigned int my_rank,
                                        unsigned int mask);

/*
 *! Explicit template instantiations for ConstraintData (n=2)
 */
template void gpu_make_ghost_group_exchange_plan<2>(unsigned int* d_ghost_group_plan,
                                                    const group_storage<2>* d_groups,
                                                    unsigned int N,
                                                    const unsigned int* d_rtag,
                                                    const unsigned int* d_plans,
                                                    unsigned int n_local);

template void gpu_exchange_ghost_groups_pack(unsigned int n_out,
                                             const uint2* d_ghost_idx_adj,
                                             const unsigned int* d_group_tag,
                                             const group_storage<2>* d_groups,
                                             const typeval_union* d_group_typeval,
                                             const group_storage<2>* d_group_ranks,
                                             packed_storage<2>* d_groups_sendbuf);

template void gpu_exchange_ghost_groups_copy_buf<2>(unsigned int nrecv,
                                                    const packed_storage<2>* d_groups_recvbuf,
                                                    unsigned int* d_group_tag,
                                                    group_storage<2>* d_groups,
                                                    typeval_union* d_group_typeval,
                                                    group_storage<2>* d_group_ranks,
                                                    unsigned int* d_keep,
                                                    unsigned int* d_scan,
                                                    const unsigned int* d_group_rtag,
                                                    const unsigned int* d_rtag,
                                                    unsigned int max_n_local,
                                                    unsigned int& n_keep,
                                                    CachedAllocator& alloc);

    } // end namespace hoomd
#endif // ENABLE_MPI
