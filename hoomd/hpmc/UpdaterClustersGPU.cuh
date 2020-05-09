// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file UpdaterClustersGPU.cuh
    \brief Implements the overlap kernels for the geometric cluster algorithm the GPU
*/

#pragma once

#include <hip/hip_runtime.h>

#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "HPMCMiscFunctions.h"
#include "hoomd/CachedAllocator.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/hpmc/GPUHelpers.cuh"

#include "IntegratorHPMCMonoGPU.cuh"

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit the number of template instantiations
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

namespace hpmc
{

namespace gpu
{

//! Wraps arguments to GPU driver functions
/*! \ingroup hpmc_data_structs */
struct cluster_args_t
    {
    //! Construct a cluster_args_t
    cluster_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const unsigned int *_d_tag,
                const Index3D& _ci,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                const unsigned int *_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _timestep,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _tpp,
                const unsigned int _overlap_threads,
                Scalar4 *_d_trial_postype,
                Scalar4 *_d_trial_orientation,
                unsigned int *_d_trial_tag,
                unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                uint2 *_d_adjacency,
                unsigned int *_d_nneigh,
                const unsigned int _maxn,
                const unsigned int _dim,
                const bool _line,
                const vec3<Scalar> _pivot,
                const quat<Scalar> _q,
                const bool _update_shape_param,
                const hipDeviceProp_t &_devprop,
                const GPUPartition& _gpu_partition,
                const hipStream_t *_streams)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_tag(_d_tag),
                  ci(_ci),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  d_check_overlaps(_check_overlaps),
                  overlap_idx(_overlap_idx),
                  timestep(_timestep),
                  box(_box),
                  block_size(_block_size),
                  tpp(_tpp),
                  overlap_threads(_overlap_threads),
                  d_trial_postype(_d_trial_postype),
                  d_trial_orientation(_d_trial_orientation),
                  d_trial_tag(_d_trial_tag),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  d_adjacency(_d_adjacency),
                  d_nneigh(_d_nneigh),
                  maxn(_maxn),
                  dim(_dim),
                  line(_line),
                  pivot(_pivot),
                  q(_q),
                  update_shape_param(_update_shape_param),
                  devprop(_devprop),
                  gpu_partition(_gpu_partition),
                  streams(_streams)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const unsigned int *d_tag;        //!< tag array
    const Index3D& ci;                //!< Cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer into interaction matrix
    const unsigned int timestep;      //!< Current time step
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute
    unsigned int tpp;                 //!< Threads per particle
    unsigned int overlap_threads;     //!< Threads per overlap check
    Scalar4 *d_trial_postype;         //!< New positions (and type) of particles
    Scalar4 *d_trial_orientation;     //!< New orientations of particles
    unsigned int *d_trial_tag;        //!< List of tags of particles in new configuration
    unsigned int *d_excell_idx;       //!< Expanded cell list
    const unsigned int *d_excell_size;//!< Size of expanded cells
    const Index2D& excli;             //!< Excell indexer
    uint2 *d_adjacency;               //!< Neighbor list of overlapping particle pairs after trial move
    unsigned int *d_nneigh;       //!< Number of overlapping particles after trial move
    unsigned int maxn;                //!< Width of neighbor list
    const unsigned int dim;           //!< Spatial dimension
    const bool line;                  //!< Is this a line reflection?
    const vec3<Scalar> pivot;         //!< pivot point
    const quat<Scalar> q;             //!< Rotation
    const bool update_shape_param;    //!< True if shape parameters have changed
    const hipDeviceProp_t& devprop;   //!< CUDA device properties
    const GPUPartition& gpu_partition; //!< Multi-GPU partition
    const hipStream_t *streams;        //!< kernel streams
    };

void connected_components(
    const uint2 *d_adj,
    unsigned int N,
    unsigned int n_elements,
    int *d_components,
    unsigned int &num_components,
    const hipDeviceProp_t& dev_prop,
    CachedAllocator& alloc);

//! Kernel driver for kernel::hpmc_clusters_overlaps
template< class Shape >
void hpmc_cluster_overlaps(const cluster_args_t& args, const typename Shape::param_type *params);

//! Kernel driver for kernel::hpmc_clusters_depletants()
template< class Shape >
void hpmc_clusters_depletants(const cluster_args_t& args, const hpmc_implicit_args_t& depletants_args, const typename Shape::param_type *params);

#ifdef __HIPCC__
namespace kernel
{

//! Check narrow-phase overlaps
template< class Shape, unsigned int max_threads >
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
__global__ void hpmc_cluster_overlaps(const Scalar4 *d_postype,
                           const Scalar4 *d_orientation,
                           const unsigned int *d_tag,
                           const Scalar4 *d_trial_postype,
                           const Scalar4 *d_trial_orientation,
                           const unsigned int *d_trial_tag,
                           const unsigned int *d_excell_idx,
                           const unsigned int *d_excell_size,
                           const Index2D excli,
                           uint2 *d_adjacency,
                           unsigned int *d_nneigh,
                           const unsigned int maxn,
                           const unsigned int num_types,
                           const BoxDim box,
                           const Scalar3 ghost_width,
                           const uint3 cell_dim,
                           const Index3D ci,
                           const unsigned int *d_check_overlaps,
                           const Index2D overlap_idx,
                           const typename Shape::param_type *d_params,
                           const unsigned int max_extra_bytes,
                           const unsigned int max_queue_size,
                           const unsigned int work_offset,
                           const unsigned int nwork)
    {
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0) && (threadIdx.x == 0);
    unsigned int n_groups = blockDim.z;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int *s_tag_group = (unsigned int*)(s_type_group + n_groups);

        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_tag_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    if (master && group == 0)
        {
        s_queue_size = 0;
        s_still_searching = 1;
        }

    bool active = true;
    unsigned int idx = blockIdx.x*n_groups+group;
    if (idx >= nwork)
        active = false;
    idx += work_offset;

    unsigned int my_cell;

    unsigned int overlap_checks = 0;
    unsigned int overlap_err_count = 0;

    if (active)
        {
        // load particle i
        Scalar4 postype_i(d_trial_postype[idx]);
        vec3<Scalar> pos_i(postype_i);
        unsigned int type_i = __scalar_as_int(postype_i.w);
        unsigned int tag_i = d_trial_tag[idx];

        // find the cell this particle should be in
        my_cell = computeParticleCell(vec_to_scalar3(pos_i), box, ghost_width,
            cell_dim, ci, false);

        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = type_i;
            s_orientation_group[group] = d_trial_orientation[idx];
            s_tag_group[group] = tag_i;
            }
        }

     // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
     __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;

    // true if we are checking against the old configuration
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    // loop while still searching

    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active && threadIdx.x == 0)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found

            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && k < excell_size)
                {
                // build some shapes, but we only need them to get diameters, so don't load orientations
                // build shape i from shared memory
                vec3<Scalar> pos_i(s_pos_group[group]);
                Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                // prefetch next j
                j = next_j;
                k += group_size;
                if (k < excell_size)
                    {
                    next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                    }

                // check particle circumspheres

                // load particle j (always load ghosts from particle data)
                const Scalar4 postype_j = d_postype[j];
                unsigned int type_j = __scalar_as_int(postype_j.w);
                vec3<Scalar> pos_j(postype_j);
                Shape shape_j(quat<Scalar>(), s_params[type_j]);

                // place ourselves into the minimum image
                vec3<Scalar> r_ij = pos_j - pos_i;
                r_ij = box.minImage(r_ij);

                unsigned int tag_j = d_tag[j];

                if (s_tag_group[group] != tag_j && check_circumsphere_overlap(r_ij, shape_i, shape_j))
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                    if (insert_point < max_queue_size)
                        {
                        s_queue_gid[insert_point] = group;
                        s_queue_j[insert_point] = j;
                        }
                    else
                        {
                        // or back up if the queue is already full
                        // we will recheck and insert this on the next time through
                        k -= group_size;
                        }
                    }
                } // end while (s_queue_size < max_queue_size && (k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j = s_queue_j[tidx_1d];

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            Scalar4 postype_j = d_postype[check_j];
            Scalar4 orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

            // put particle j into the coordinate system of particle i
            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)]
                && test_overlap(r_ij, shape_i, shape_j, overlap_err_count))
                {
                // write out to global memory
                #if (__CUDA_ARCH__ >= 600)
                unsigned int n = atomicAdd_system(d_nneigh, 1);
                #else
                unsigned int n = atomicAdd(d_nneigh, 1);
                #endif
                if (n < maxn)
                    {
                    d_adjacency[n] = make_uint2(s_tag_group[check_group],d_tag[check_j]);
                    }
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && threadIdx.x == 0 && k < excell_size)
            atomicAdd(&s_still_searching, 1);

        __syncthreads();
        } // end while (s_still_searching)
    }

//! Launcher for narrow phase kernel with templated launch bounds
template< class Shape, unsigned int cur_launch_bounds >
void cluster_overlaps_launcher(const cluster_args_t& args, const typename Shape::param_type *params,
    unsigned int max_threads, detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds*MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_cluster_overlaps<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int overlap_threads = args.overlap_threads;
        unsigned int tpp = min(args.tpp,run_block_size);

        while (overlap_threads*tpp > run_block_size || run_block_size % (overlap_threads*tpp) != 0)
            {
            tpp--;
            }

        unsigned int n_groups = run_block_size/(tpp*overlap_threads);
        n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);

        unsigned int max_queue_size = n_groups*tpp;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type)
            + args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups * (2*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            run_block_size -= args.devprop.warpSize;
            if (run_block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");

            tpp = min(tpp, run_block_size);
            while (overlap_threads*tpp > run_block_size || run_block_size % (overlap_threads*tpp) != 0)
                {
                tpp--;
                }

            n_groups = run_block_size/(tpp*overlap_threads);
            n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);
            max_queue_size = n_groups*tpp;

            shared_bytes = n_groups * (2*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
                + max_queue_size * 2 * sizeof(unsigned int)
                + min_shared_bytes;
            }

        // determine dynamically allocated shared memory size
        static unsigned int base_shared_bytes = UINT_MAX;
        bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
        base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

        unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
        static unsigned int extra_bytes = UINT_MAX;
        if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
            {
            // determine dynamically requested shared memory
            char *ptr = (char *)nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                params[i].allocate_shared(ptr, available_bytes);
                }
            extra_bytes = max_extra_bytes - available_bytes;
            }

        shared_bytes += extra_bytes;
        dim3 thread(overlap_threads, tpp, n_groups);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            unsigned int nwork = range.second - range.first;
            const unsigned int num_blocks = nwork/n_groups + 1;

            dim3 grid(num_blocks, 1, 1);

            hipLaunchKernelGGL((hpmc_cluster_overlaps<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>), grid, thread, shared_bytes, args.streams[idev],
                args.d_postype, args.d_orientation, args.d_tag, args.d_trial_postype, args.d_trial_orientation, args.d_trial_tag,
                args.d_excell_idx, args.d_excell_size, args.excli,
                args.d_adjacency, args.d_nneigh, args.maxn, args.num_types,
                args.box, args.ghost_width, args.cell_dim, args.ci, args.d_check_overlaps,
                args.overlap_idx, params,
                max_extra_bytes, max_queue_size, range.first, nwork);
            }
        }
    else
        {
        cluster_overlaps_launcher<Shape>(args, params, max_threads, detail::int2type<cur_launch_bounds/2>());
        }
    }

//! Kernel to insert depletants on-the-fly
template< class Shape, unsigned int max_threads >
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
__global__ void clusters_insert_depletants(const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     const unsigned int *d_tag,
                                     bool line,
                                     vec3<Scalar> pivot,
                                     quat<Scalar> q,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const Index3D ci,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes,
                                     unsigned int depletant_type,
                                     const Index2D depletant_idx,
                                     unsigned int *d_nneigh,
                                     uint2 *d_adjacency,
                                     const unsigned int maxn,
                                     unsigned int work_offset,
                                     unsigned int max_depletant_queue_size,
                                     const unsigned int *d_n_depletants)
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.z;

    unsigned int err_count = 0;

    // shared particle configuation
    __shared__ Scalar4 s_orientation_i;
    __shared__ Scalar3 s_pos_i;
    __shared__ unsigned int s_type_i;
    __shared__ unsigned int s_tag_i;
    __shared__ int3 s_img_i;

    // shared queue variables
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_adding_depletants;
    __shared__ unsigned int s_depletant_queue_size;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_reject_group = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_reject_group + n_groups);
    unsigned int *s_queue_j = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_didx = (unsigned int *)(s_queue_gid + max_queue_size);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_queue_didx + max_depletant_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    // identify the active cell that this thread handles
    unsigned int i = blockIdx.x + work_offset;

    // load updated particle position
    if (master && group == 0)
        {
        Scalar4 postype_i = d_postype[i];
        s_pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        s_type_i = __scalar_as_int(postype_i.w);
        s_tag_i = d_tag[i];
        s_orientation_i = d_orientation[i];

        // get image of particle i after transformation
        vec3<Scalar> pos_i_transf(s_pos_i);
        if (line)
            {
            pos_i_transf = lineReflection(pos_i_transf, pivot, q);
            }
        else
            {
            pos_i_transf = pivot-(pos_i_transf-pivot);
            }
        s_img_i = box.getImage(pos_i_transf);
        }

    // sync so that s_pos_i etc. are available
    __syncthreads();

    // generate random number of depletants from Poisson distribution
    unsigned int n_depletants = d_n_depletants[i];

    unsigned int overlap_checks = 0;
    unsigned int n_inserted = 0;

    // find the cell this particle should be in
    unsigned int my_cell = computeParticleCell(s_pos_i, box, ghost_width,
        cell_dim, ci, false);

    detail::OBB obb_i;
        {
        // get shape OBB
        Shape shape_i(quat<Scalar>(d_orientation[i]), s_params[s_type_i]);
        obb_i = shape_i.getOBB(vec3<Scalar>(s_pos_i));

        // extend by depletant radius
        Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

        Scalar r = 0.5*detail::max(shape_test.getCircumsphereDiameter(),
            shape_test.getCircumsphereDiameter());
        obb_i.lengths.x += r;
        obb_i.lengths.y += r;

        if (dim == 3)
            obb_i.lengths.z += r;
        else
            obb_i.lengths.z = OverlapReal(0.5);
        }

    if (master && group == 0)
        {
        s_depletant_queue_size = 0;
        s_adding_depletants = 1;
        }

    __syncthreads();

    unsigned int gidx = gridDim.y*blockIdx.z+blockIdx.y;
    unsigned int blocks_per_particle = gridDim.y*gridDim.z;
    unsigned int i_dep = group_size*group+offset + gidx*group_size*n_groups;

    while (s_adding_depletants)
        {
        while (s_depletant_queue_size < max_depletant_queue_size && i_dep < n_depletants)
            {
            // one RNG per depletant
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletantsClusters, seed+i, i_dep,
                depletant_idx(depletant_type,depletant_type), timestep);

            n_inserted++;
            overlap_checks += 2;

            // test depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));

            Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng,dim);
                }

            Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
            if (shape_i.hasOrientation())
                shape_i.orientation = quat<Scalar>(s_orientation_i);
            vec3<Scalar> r_ij = vec3<Scalar>(s_pos_i) - pos_test;
            bool overlap = (s_check_overlaps[overlap_idx(s_type_i, depletant_type)]
                && check_circumsphere_overlap(r_ij, shape_test, shape_i)
                && test_overlap(r_ij, shape_test, shape_i, err_count));

            if (overlap)
                {
                // add this particle to the queue
                unsigned int insert_point = atomicAdd(&s_depletant_queue_size, 1);

                if (insert_point < max_depletant_queue_size)
                    {
                    s_queue_didx[insert_point] = i_dep;
                    }
                else
                    {
                    // we will recheck and insert this on the next time through
                    break;
                    }
                } // end if add_to_queue

            // advance depletant idx
            i_dep += group_size*n_groups*blocks_per_particle;
            } // end while (s_depletant_queue_size < max_depletant_queue_size && i_dep < n_depletants)

        __syncthreads();

        // process the queue, group by group
        if (master && group == 0)
            s_adding_depletants = 0;

        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // is this group processing work from the first queue?
        bool active = group < min(s_depletant_queue_size, max_depletant_queue_size);

        if (active)
            {
            // regenerate depletant using seed from queue, this costs a few flops but is probably
            // better than storing one Scalar4 and a Scalar3 per thread in shared mem
            unsigned int i_dep_queue = s_queue_didx[group];
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletantsClusters, seed+i, i_dep_queue,
                depletant_idx(depletant_type,depletant_type), timestep);

            // depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));
            Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng,dim);
                }

            // store them per group
            if (master)
                {
                s_pos_group[group] = vec_to_scalar3(pos_test);
                s_orientation_group[group] = quat_to_scalar4(shape_test.orientation);
                }
            }

        if (master)
            {
            s_reject_group[group] = 0;
            }

        __syncthreads();

        // counters to track progress through the loop over potential neighbors
        unsigned int excell_size;
        unsigned int k = offset;

        // first pass, see if this depletant is active
        if (active)
            {
            excell_size = d_excell_size[my_cell];
            }

        // loop while still searching
        while (s_still_searching)
            {
            // fill the neighbor queue
            // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                // and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1,0,0,0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for circumsphere check
                    // for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                        check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (s_tag_i != d_tag[j] && circumsphere_overlap)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            // when we get here, all threads have either finished their list, or encountered a full queue
            // either way, it is time to process overlaps
            // need to clear the still searching flag and sync first
            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size*group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                    test_overlap(r_jk, shape_test, shape_j, err_count) &&
                    d_tag[check_j] < s_tag_i)
                    {
                    s_reject_group[check_group] = 1;
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        __syncthreads();
        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // second pass, see if transformed depletant overlaps
        vec3<Scalar> pos_test_transf;
        quat<Scalar> orientation_test_transf;
        unsigned int other_cell;

        active &= !s_reject_group[group];

        if (active)
            {
            pos_test_transf = vec3<Scalar>(s_pos_group[group]);
            if (line)
                {
                pos_test_transf = lineReflection(pos_test_transf, pivot, q);
                }
            else
                {
                pos_test_transf = pivot - (pos_test_transf - pivot);
                }

            // wrap back into into i's image (after transformation)
            pos_test_transf = box.shift(pos_test_transf,-s_img_i);
            int3 img = make_int3(0,0,0);
            box.wrap(pos_test_transf,img);

            other_cell = computeParticleCell(vec_to_scalar3(pos_test_transf), box,
                ghost_width, cell_dim, ci, false);
            excell_size = d_excell_size[other_cell];
            }

        k = offset;

        // loop while still searching
        while (s_still_searching)
            {
            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, other_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                // and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1,0,0,0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, other_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for circumsphere check
                    // for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test_transf);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                        check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (circumsphere_overlap)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size*group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                vec3<Scalar> pos_test_transf(s_pos_group[check_group]);

                quat<Scalar> orientation_test_transf(q*quat<Scalar>(s_orientation_group[check_group]));
                if (line)
                    {
                    pos_test_transf = lineReflection(pos_test_transf, pivot, q);
                    }
                else
                    {
                    pos_test_transf = pivot - (pos_test_transf - pivot);
                    }

                // wrap back into into i's image (after transformation)
                pos_test_transf = box.shift(pos_test_transf,-s_img_i);
                int3 img = make_int3(0,0,0);
                box.wrap(pos_test_transf,img);

                Shape shape_test_transf(quat<Scalar>(orientation_test_transf), s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test_transf);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                    test_overlap(r_jk, shape_test_transf, shape_j, err_count))
                    {
                    s_reject_group[check_group] = 1;
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();
            }

        __syncthreads();
        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // third pass, record overlaps
        active &= !s_reject_group[group];

        if (active)
            {
            excell_size = d_excell_size[my_cell];
            }

        k = offset;

        // loop while still searching
        while (s_still_searching)
            {
            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                // and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1,0,0,0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for circumsphere check
                    // for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                        check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (circumsphere_overlap)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size*group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                vec3<Scalar> pos_test(s_pos_group[check_group]);
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)] &&
                    test_overlap(r_jk, shape_test, shape_j, err_count))
                    {
                    // write out to global memory
                    #if (__CUDA_ARCH__ >= 600)
                    unsigned int n = atomicAdd_system(d_nneigh, 1);
                    #else
                    unsigned int n = atomicAdd(d_nneigh, 1);
                    #endif
                    if (n < maxn)
                        {
                        d_adjacency[n] = make_uint2(s_tag_i,d_tag[check_j]);
                        }
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();
            }

        // do we still need to process depletants?
        __syncthreads();
        if (master && group == 0)
            s_depletant_queue_size = 0;
        if (i_dep < n_depletants)
            atomicAdd(&s_adding_depletants, 1);
        __syncthreads();
        } // end loop over depletants
    }

//! Launcher for clusters_insert_depletants kernel with templated launch bounds
template< class Shape, unsigned int cur_launch_bounds>
void clusters_depletants_launcher(const cluster_args_t& args, const hpmc_implicit_args_t& implicit_args,
    const typename Shape::param_type *params, unsigned int max_threads, detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds*MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&kernel::clusters_insert_depletants<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp,block_size);
        unsigned int n_groups = block_size / tpp;
        n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);
        unsigned int max_queue_size = n_groups*tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                   args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                                    max_queue_size*2*sizeof(unsigned int) +
                                    max_depletant_queue_size*sizeof(unsigned int) +
                                    min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            block_size -= args.devprop.warpSize;
            if (block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");
            tpp = min(tpp, block_size);
            n_groups = block_size / tpp;
            n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);
            max_queue_size = n_groups*tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int)) +
                           max_queue_size*2*sizeof(unsigned int) +
                           max_depletant_queue_size*sizeof(unsigned int) +
                           min_shared_bytes;
            }

        static unsigned int base_shared_bytes = UINT_MAX;
        bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
        base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

        unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
        static unsigned int extra_bytes = UINT_MAX;
        if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
            {
            // determine dynamically requested shared memory
            char *ptr = (char *) nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                params[i].allocate_shared(ptr, available_bytes);
                }
            extra_bytes = max_extra_bytes - available_bytes;
            }

        shared_bytes += extra_bytes;

        // setup the grid to run the kernel
        dim3 threads(1, tpp, n_groups);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            if (range.first == range.second)
                continue;

            unsigned int blocks_per_particle = implicit_args.max_n_depletants[idev] /
                (implicit_args.depletants_per_group*n_groups) + 1;

            dim3 grid( range.second-range.first, blocks_per_particle, 1);

            if (blocks_per_particle > args.devprop.maxGridSize[1])
                {
                grid.y = args.devprop.maxGridSize[1];
                grid.z = blocks_per_particle/args.devprop.maxGridSize[1]+1;
                }

            hipLaunchKernelGGL((kernel::clusters_insert_depletants<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE>),
                dim3(grid), dim3(threads), shared_bytes, implicit_args.streams[idev],
                                 args.d_postype,
                                 args.d_orientation,
                                 args.d_tag,
                                 args.line,
                                 args.pivot,
                                 args.q,
                                 args.d_excell_idx,
                                 args.d_excell_size,
                                 args.excli,
                                 args.cell_dim,
                                 args.ghost_width,
                                 args.ci,
                                 args.num_types,
                                 args.seed,
                                 args.d_check_overlaps,
                                 args.overlap_idx,
                                 args.timestep,
                                 args.dim,
                                 args.box,
                                 params,
                                 max_queue_size,
                                 max_extra_bytes,
                                 implicit_args.depletant_type_a,
                                 implicit_args.depletant_idx,
                                 args.d_nneigh,
                                 args.d_adjacency,
                                 args.maxn,
                                 range.first,
                                 max_depletant_queue_size,
                                 implicit_args.d_n_depletants);
            }
        }
    else
        {
        clusters_depletants_launcher<Shape>(args, implicit_args, params, max_threads, detail::int2type<cur_launch_bounds/2>());
        }
    }

} // end namespace kernel

//! Kernel driver for kernel::hpmc_clusters_overlaps
template< class Shape >
void hpmc_cluster_overlaps(const cluster_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);

    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::cluster_overlaps_launcher<Shape>(args, params, launch_bounds, detail::int2type<MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
    }

//! Kernel driver for kernel::hpmc_clusters_depletants()
template< class Shape >
void hpmc_clusters_depletants(const cluster_args_t& args, const hpmc_implicit_args_t& depletants_args, const typename Shape::param_type *params)
    {
    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::clusters_depletants_launcher<Shape>(args, depletants_args, params, launch_bounds, detail::int2type<MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
    }
#endif

} // end namespace gpu
} // end namespace hpmc

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE
