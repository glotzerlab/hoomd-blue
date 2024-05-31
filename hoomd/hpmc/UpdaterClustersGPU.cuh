// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file UpdaterClustersGPU.cuh
    \brief Implements the overlap kernels for the geometric cluster algorithm the GPU
*/

#pragma once

#include <hip/hip_runtime.h>

#include "HPMCMiscFunctions.h"
#include "hoomd/BoxDim.h"
#include "hoomd/CachedAllocator.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"

#include "GPUHelpers.cuh"
#include "Moves.h"

#include "IntegratorHPMCMonoGPUTypes.cuh"

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit the number of template instantiations
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Wraps arguments to GPU driver functions
/*! \ingroup hpmc_data_structs */
struct cluster_args_t
    {
    //! Construct a cluster_args_t
    cluster_args_t(const Scalar4* _d_postype,
                   const Scalar4* _d_orientation,
                   const Index3D& _ci,
                   const uint3& _cell_dim,
                   const Scalar3& _ghost_width,
                   const unsigned int _N,
                   const unsigned int _num_types,
                   const uint16_t _seed,
                   const unsigned int* _check_overlaps,
                   const Index2D& _overlap_idx,
                   const uint64_t _timestep,
                   const BoxDim& _box,
                   const unsigned int _block_size,
                   const unsigned int _tpp,
                   const unsigned int _overlap_threads,
                   const Scalar4* _d_trial_postype,
                   const Scalar4* _d_trial_orientation,
                   unsigned int* _d_excell_idx,
                   const unsigned int* _d_excell_size,
                   const Index2D& _excli,
                   unsigned int* _d_adjacency,
                   unsigned int* _d_nneigh,
                   const unsigned int _maxn,
                   unsigned int* _d_overflow,
                   const unsigned int _dim,
                   const bool _line,
                   const vec3<Scalar> _pivot,
                   const quat<Scalar> _q,
                   const bool _update_shape_param,
                   const hipDeviceProp_t& _devprop,
                   const GPUPartition& _gpu_partition,
                   const hipStream_t* _streams)
        : d_postype(_d_postype), d_orientation(_d_orientation), ci(_ci), cell_dim(_cell_dim),
          ghost_width(_ghost_width), N(_N), num_types(_num_types), seed(_seed),
          d_check_overlaps(_check_overlaps), overlap_idx(_overlap_idx), timestep(_timestep),
          box(_box), block_size(_block_size), tpp(_tpp), overlap_threads(_overlap_threads),
          d_trial_postype(_d_trial_postype), d_trial_orientation(_d_trial_orientation),
          d_excell_idx(_d_excell_idx), d_excell_size(_d_excell_size), excli(_excli),
          d_adjacency(_d_adjacency), d_nneigh(_d_nneigh), maxn(_maxn), d_overflow(_d_overflow),
          dim(_dim), line(_line), pivot(_pivot), q(_q), update_shape_param(_update_shape_param),
          devprop(_devprop), gpu_partition(_gpu_partition), streams(_streams) {};

    const Scalar4* d_postype;             //!< postype array
    const Scalar4* d_orientation;         //!< orientation array
    const Index3D& ci;                    //!< Cell indexer
    const uint3& cell_dim;                //!< Cell dimensions
    const Scalar3& ghost_width;           //!< Width of the ghost layer
    const unsigned int N;                 //!< Number of particles
    const unsigned int num_types;         //!< Number of particle types
    const uint16_t seed;                  //!< RNG seed
    const unsigned int* d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;           //!< Indexer into interaction matrix
    const uint64_t timestep;              //!< Current time step
    const BoxDim box;                     //!< Current simulation box
    unsigned int block_size;              //!< Block size to execute
    unsigned int tpp;                     //!< Threads per particle
    unsigned int overlap_threads;         //!< Threads per overlap check
    const Scalar4* d_trial_postype;       //!< New positions (and type) of particles
    const Scalar4* d_trial_orientation;   //!< New orientations of particles
    unsigned int* d_excell_idx;           //!< Expanded cell list
    const unsigned int* d_excell_size;    //!< Size of expanded cells
    const Index2D& excli;                 //!< Excell indexer
    unsigned int* d_adjacency;            //!< Neighbor list
    unsigned int* d_nneigh;               //!< Number of overlapping particles after trial move
    unsigned int maxn;                    //!< Width of neighbor list
    unsigned int* d_overflow;             //<! Max number of neighbors (output)
    const unsigned int dim;               //!< Spatial dimension
    const bool line;                      //!< Is this a line reflection?
    const vec3<Scalar> pivot;             //!< pivot point
    const quat<Scalar> q;                 //!< Rotation
    const bool update_shape_param;        //!< True if shape parameters have changed
    const hipDeviceProp_t& devprop;       //!< CUDA device properties
    const GPUPartition& gpu_partition;    //!< Multi-GPU partition
    const hipStream_t* streams;           //!< kernel streams
    };

void __attribute__((visibility("default"))) connected_components(uint2* d_adj,
                                                                 unsigned int N,
                                                                 const unsigned int n_elements,
                                                                 int* d_components,
                                                                 unsigned int& num_components,
                                                                 const hipDeviceProp_t& dev_prop,
                                                                 CachedAllocator& alloc);

void get_num_neighbors(const unsigned int* d_nneigh,
                       unsigned int* d_nneigh_scan,
                       unsigned int& nneigh_total,
                       const GPUPartition& gpu_partition,
                       CachedAllocator& alloc);

void concatenate_adjacency_list(const unsigned int* d_adjacency,
                                const unsigned int* d_nneigh,
                                const unsigned int* d_nneigh_scan,
                                const unsigned int maxn,
                                uint2* d_adjacency_out,
                                const GPUPartition& gpu_partition,
                                const unsigned int block_size,
                                const unsigned int group_size);

void flip_clusters(Scalar4* d_postype,
                   Scalar4* d_orientation,
                   int3* d_image,
                   const Scalar4* d_postype_backup,
                   const Scalar4* d_orientation_backup,
                   const int3* d_image_backup,
                   const int* d_components,
                   float flip_probability,
                   uint16_t seed,
                   uint64_t timestep,
                   const GPUPartition& gpu_partition,
                   const unsigned int block_size);

//! Arguments to gpu::transform_particles
struct clusters_transform_args_t
    {
    //! Construct a cluster_args_t
    clusters_transform_args_t(Scalar4* _d_postype,
                              Scalar4* _d_orientation,
                              int3* _d_image,
                              const vec3<Scalar>& _pivot,
                              const quat<Scalar>& _q,
                              const bool _line,
                              const GPUPartition& _gpu_partition,
                              const BoxDim& _box,
                              const unsigned int _num_types,
                              const unsigned int _block_size)
        : d_postype(_d_postype), d_orientation(_d_orientation), d_image(_d_image), pivot(_pivot),
          q(_q), line(_line), gpu_partition(_gpu_partition), box(_box), num_types(_num_types),
          block_size(_block_size)
        {
        }

    Scalar4* d_postype;
    Scalar4* d_orientation;
    int3* d_image;
    const vec3<Scalar> pivot;
    const quat<Scalar> q;
    const bool line;
    const GPUPartition& gpu_partition;
    const BoxDim box;
    const unsigned int num_types;
    const unsigned int block_size;
    };

template<class Shape>
void transform_particles(const clusters_transform_args_t& args,
                         const typename Shape::param_type* d_params);

//! Kernel driver for kernel::hpmc_clusters_overlaps
template<class Shape>
void hpmc_cluster_overlaps(const cluster_args_t& args, const typename Shape::param_type* params);

#ifdef __HIPCC__
namespace kernel
    {
//! Check narrow-phase overlaps
template<class Shape, unsigned int max_threads>
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
    __global__ void hpmc_cluster_overlaps(const Scalar4* d_postype,
                                          const Scalar4* d_orientation,
                                          const Scalar4* d_trial_postype,
                                          const Scalar4* d_trial_orientation,
                                          const unsigned int* d_excell_idx,
                                          const unsigned int* d_excell_size,
                                          const Index2D excli,
                                          unsigned int* d_adjacency,
                                          unsigned int* d_nneigh,
                                          const unsigned int maxn,
                                          unsigned int* d_overflow,
                                          const unsigned int num_types,
                                          const BoxDim box,
                                          const Scalar3 ghost_width,
                                          const uint3 cell_dim,
                                          const Index3D ci,
                                          const unsigned int* d_check_overlaps,
                                          const Index2D overlap_idx,
                                          const typename Shape::param_type* d_params,
                                          const unsigned int max_extra_bytes,
                                          const unsigned int max_queue_size,
                                          const unsigned int work_offset,
                                          const unsigned int nwork)
    {
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0) && (threadIdx.x == 0);
    unsigned int n_groups = blockDim.y;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED(char, s_data)

    typename Shape::param_type* s_params = (typename Shape::param_type*)(&s_data[0]);
    Scalar4* s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3* s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int* s_check_overlaps = (unsigned int*)(s_pos_group + n_groups);
    unsigned int* s_queue_j = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int* s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int* s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int* s_idx_group = (unsigned int*)(s_type_group + n_groups);

        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx
            = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        unsigned int block_size = blockDim.x * blockDim.y * blockDim.z;
        unsigned int param_size = num_types * sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int*)s_params)[cur_offset + tidx] = ((int*)d_params)[cur_offset + tidx];
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
    char* s_extra = (char*)(s_idx_group + n_groups);

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
    unsigned int idx = blockIdx.x * n_groups + group;
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

        // find the cell this particle should be in
        my_cell = computeParticleCell(vec_to_scalar3(pos_i), box, ghost_width, cell_dim, ci, false);

        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = type_i;
            s_orientation_group[group] = d_trial_orientation[idx];
            s_idx_group[group] = idx;
            }
        }

    // sync so that s_postype_group and s_orientation are available before other threads might
    // process overlap checks
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
        // loop through particles in the excell list and add them to the queue if they pass the
        // circumsphere check

        // active threads add to the queue
        if (active && threadIdx.x == 0)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the
            // end of our own list and as long as no overlaps have been found

            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && k < excell_size)
                {
                // build some shapes, but we only need them to get diameters, so don't load
                // orientations build shape i from shared memory
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

                if (s_idx_group[group] != j && check_circumsphere_overlap(r_ij, shape_i, shape_j))
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

        // when we get here, all threads have either finished their list, or encountered a full
        // queue either way, it is time to process overlaps need to clear the still searching flag
        // and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size * group;

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
            Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
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
                unsigned int n = atomicAdd(&d_nneigh[s_idx_group[check_group]], 1);
                if (n < maxn)
                    {
                    d_adjacency[n + s_idx_group[check_group] * maxn] = check_j;
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

    if (active && master)
        {
        // overflowed?
        unsigned int nneigh = d_nneigh[idx];
        if (nneigh > maxn)
            {
#if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
#else
            atomicMax(d_overflow, nneigh);
#endif
            }
        }
    }

//! Launcher for narrow phase kernel with templated launch bounds
template<class Shape, unsigned int cur_launch_bounds>
void cluster_overlaps_launcher(const cluster_args_t& args,
                               const typename Shape::param_type* params,
                               unsigned int max_threads,
                               detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds * MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        int max_block_size;
        hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero
            = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        hipFuncGetAttributes(
            &attr,
            reinterpret_cast<const void*>(
                kernel::hpmc_cluster_overlaps<Shape, launch_bounds_nonzero * MIN_BLOCK_SIZE>));
        max_block_size = attr.maxThreadsPerBlock;
        if (max_block_size % args.devprop.warpSize)
            // handle non-sensical return values from hipFuncGetAttributes
            max_block_size = (max_block_size / args.devprop.warpSize - 1) * args.devprop.warpSize;

        // choose a block size based on the max block size by regs (max_block_size) and include
        // dynamic shared memory usage
        unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int overlap_threads = args.overlap_threads;
        unsigned int tpp = min(args.tpp, run_block_size);

        while (overlap_threads * tpp > run_block_size
               || run_block_size % (overlap_threads * tpp) != 0)
            {
            tpp--;
            }
        tpp = std::min((unsigned int)args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

        unsigned int n_groups = run_block_size / (tpp * overlap_threads);
        unsigned int max_queue_size = n_groups * tpp;

        const unsigned int min_shared_bytes
            = static_cast<unsigned int>(args.num_types * sizeof(typename Shape::param_type)
                                        + args.overlap_idx.getNumElements() * sizeof(unsigned int));

        size_t shared_bytes
            = n_groups * (2 * sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
              + max_queue_size * 2 * sizeof(unsigned int) + min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of "
                                     "particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            run_block_size -= args.devprop.warpSize;
            if (run_block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");

            tpp = min(tpp, run_block_size);
            while (overlap_threads * tpp > run_block_size
                   || run_block_size % (overlap_threads * tpp) != 0)
                {
                tpp--;
                }
            tpp = std::min((unsigned int)args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z

            n_groups = run_block_size / (tpp * overlap_threads);
            max_queue_size = n_groups * tpp;

            shared_bytes = n_groups * (2 * sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
                           + max_queue_size * 2 * sizeof(unsigned int) + min_shared_bytes;
            }

        // determine dynamically allocated shared memory size
        unsigned int base_shared_bytes;
        base_shared_bytes = static_cast<unsigned int>(shared_bytes + attr.sharedSizeBytes);

        unsigned int max_extra_bytes
            = static_cast<unsigned int>(args.devprop.sharedMemPerBlock - base_shared_bytes);
        unsigned int extra_bytes;
        // determine dynamically requested shared memory
        char* ptr = (char*)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;

        shared_bytes += extra_bytes;
        dim3 thread(overlap_threads, n_groups, tpp);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            unsigned int nwork = range.second - range.first;
            const unsigned int num_blocks = nwork / n_groups + 1;

            dim3 grid(num_blocks, 1, 1);

            hipLaunchKernelGGL(
                (hpmc_cluster_overlaps<Shape, launch_bounds_nonzero * MIN_BLOCK_SIZE>),
                grid,
                thread,
                shared_bytes,
                args.streams[idev],
                args.d_postype,
                args.d_orientation,
                args.d_trial_postype,
                args.d_trial_orientation,
                args.d_excell_idx,
                args.d_excell_size,
                args.excli,
                args.d_adjacency,
                args.d_nneigh,
                args.maxn,
                args.d_overflow,
                args.num_types,
                args.box,
                args.ghost_width,
                args.cell_dim,
                args.ci,
                args.d_check_overlaps,
                args.overlap_idx,
                params,
                max_extra_bytes,
                max_queue_size,
                range.first,
                nwork);
            }
        }
    else
        {
        cluster_overlaps_launcher<Shape>(args,
                                         params,
                                         max_threads,
                                         detail::int2type<cur_launch_bounds / 2>());
        }
    }

template<class Shape>
__global__ void transform_particles(Scalar4* d_postype,
                                    Scalar4* d_orientation,
                                    int3* d_image,
                                    const vec3<Scalar> pivot,
                                    const quat<Scalar> q,
                                    const bool line,
                                    const unsigned int num_types,
                                    const BoxDim box,
                                    const unsigned int nwork,
                                    const unsigned int work_offset,
                                    const typename Shape::param_type* d_params)
    {
    unsigned int work_idx = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ char s_data[];
    typename Shape::param_type* s_params = (typename Shape::param_type*)s_data;
        {
        // copy over parameters one int per thread for fast loads
        unsigned int tidx
            = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        unsigned int block_size = blockDim.x * blockDim.y * blockDim.z;
        unsigned int param_size = num_types * sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int*)s_params)[cur_offset + tidx] = ((int*)d_params)[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    if (work_idx >= nwork)
        return;
    unsigned int i = work_idx + work_offset;

    vec3<Scalar> new_pos(d_postype[i]);

    if (!line)
        {
        // point reflection
        new_pos = pivot - (new_pos - pivot);
        }
    else
        {
        // line reflection
        new_pos = lineReflection(new_pos, pivot, q);
        Shape shape_i(quat<Scalar>(), s_params[__scalar_as_int(d_postype[i].w)]);
        if (shape_i.hasOrientation())
            d_orientation[i] = quat_to_scalar4(q * quat<Scalar>(d_orientation[i]));
        }

    // wrap particle back into box, incrementing image flags
    int3 img = box.getImage(new_pos);
    new_pos = box.shift(new_pos, -img);
    d_postype[i] = make_scalar4(new_pos.x, new_pos.y, new_pos.z, d_postype[i].w);
    d_image[i] = d_image[i] + img;
    }

    } // end namespace kernel

//! Kernel driver for kernel::hpmc_clusters_overlaps
template<class Shape>
void hpmc_cluster_overlaps(const cluster_args_t& args, const typename Shape::param_type* params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);

    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::cluster_overlaps_launcher<Shape>(args,
                                             params,
                                             launch_bounds,
                                             detail::int2type<MAX_BLOCK_SIZE / MIN_BLOCK_SIZE>());
    }

template<class Shape>
void transform_particles(const clusters_transform_args_t& args,
                         const typename Shape::param_type* d_params)
    {
    // determine the maximum block size and clamp the input block size down
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&kernel::transform_particles<Shape>));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

    const size_t shared_bytes = sizeof(typename Shape::param_type) * args.num_types;

    dim3 threads(run_block_size, 1, 1);

    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork / run_block_size + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL((kernel::transform_particles<Shape>),
                           grid,
                           threads,
                           shared_bytes,
                           0,
                           args.d_postype,
                           args.d_orientation,
                           args.d_image,
                           args.pivot,
                           args.q,
                           args.line,
                           args.num_types,
                           args.box,
                           nwork,
                           range.first,
                           d_params);
        }
    }
#endif

    } // end namespace gpu
    } // end namespace hpmc
    } // end namespace hoomd
#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE
