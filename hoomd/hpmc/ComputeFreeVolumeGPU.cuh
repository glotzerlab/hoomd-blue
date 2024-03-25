// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _COMPUTE_FREE_VOLUME_CUH_
#define _COMPUTE_FREE_VOLUME_CUH_

#include "HPMCCounters.h"
#include "hip/hip_runtime.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#ifdef __HIPCC__
#include "Moves.h"
#include "hoomd/TextureTools.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/*! \file IntegratorHPMCMonoImplicit.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_free_volume
/*! \ingroup hpmc_data_structs */
struct hpmc_free_volume_args_t
    {
    //! Construct a pair_args_t
    hpmc_free_volume_args_t(unsigned int _n_sample,
                            unsigned int _type,
                            Scalar4* _d_postype,
                            Scalar4* _d_orientation,
                            const unsigned int* _d_cell_idx,
                            const unsigned int* _d_cell_size,
                            const Index3D& _ci,
                            const Index2D& _cli,
                            const unsigned int* _d_excell_idx,
                            const unsigned int* _d_excell_size,
                            const Index2D& _excli,
                            const uint3& _cell_dim,
                            const unsigned int _N,
                            const unsigned int _num_types,
                            const uint16_t _seed,
                            const unsigned int _rank,
                            unsigned int _select,
                            const uint64_t _timestep,
                            const unsigned int _dim,
                            const BoxDim& _box,
                            const unsigned int _block_size,
                            const unsigned int _stride,
                            const unsigned int _group_size,
                            const unsigned int _max_n,
                            unsigned int* _d_n_overlap_all,
                            const Scalar3 _ghost_width,
                            const unsigned int* _d_check_overlaps,
                            Index2D _overlap_idx,
                            const hipDeviceProp_t& _devprop)
        : n_sample(_n_sample), type(_type), d_postype(_d_postype), d_orientation(_d_orientation),
          d_cell_idx(_d_cell_idx), d_cell_size(_d_cell_size), ci(_ci), cli(_cli),
          d_excell_idx(_d_excell_idx), d_excell_size(_d_excell_size), excli(_excli),
          cell_dim(_cell_dim), N(_N), num_types(_num_types), seed(_seed), rank(_rank),
          select(_select), timestep(_timestep), dim(_dim), box(_box), block_size(_block_size),
          stride(_stride), group_size(_group_size), max_n(_max_n),
          d_n_overlap_all(_d_n_overlap_all), ghost_width(_ghost_width),
          d_check_overlaps(_d_check_overlaps), overlap_idx(_overlap_idx), devprop(_devprop) {};

    unsigned int n_sample;                //!< Number of depletants particles to generate
    unsigned int type;                    //!< Type of depletant particle
    Scalar4* d_postype;                   //!< postype array
    Scalar4* d_orientation;               //!< orientation array
    const unsigned int* d_cell_idx;       //!< Index data for each cell
    const unsigned int* d_cell_size;      //!< Number of particles in each cell
    const Index3D& ci;                    //!< Cell indexer
    const Index2D& cli;                   //!< Indexer for d_cell_idx
    const unsigned int* d_excell_idx;     //!< Expanded cell neighbors
    const unsigned int* d_excell_size;    //!< Size of expanded cell list per cell
    const Index2D excli;                  //!< Expanded cell indexer
    const uint3& cell_dim;                //!< Cell dimensions
    const unsigned int N;                 //!< Number of particles
    const unsigned int num_types;         //!< Number of particle types
    const uint16_t seed;                  //!< RNG seed
    const unsigned int rank;              //!< MPI rank
    unsigned int select;                  //!< RNG select value
    const uint64_t timestep;              //!< Current time step
    const unsigned int dim;               //!< Number of dimensions
    const BoxDim box;                     //!< Current simulation box
    unsigned int block_size;              //!< Block size to execute
    unsigned int stride;                  //!< Number of threads per overlap check
    unsigned int group_size;              //!< Size of the group to execute
    const unsigned int max_n;             //!< Maximum size of pdata arrays
    unsigned int* d_n_overlap_all;        //!< Total number of depletants in overlap volume
    const Scalar3 ghost_width;            //!< Width of ghost layer
    const unsigned int* d_check_overlaps; //!< Interaction matrix
    Index2D overlap_idx;                  //!< Interaction matrix indexer
    const hipDeviceProp_t& devprop;       //!< CUDA device properties
    };

template<class Shape>
hipError_t gpu_hpmc_free_volume(const hpmc_free_volume_args_t& args,
                                const typename Shape::param_type* d_params);

#ifdef __HIPCC__

//! Compute the cell that a particle sits in
__device__ inline unsigned int compute_cell_idx(const Scalar3 p,
                                                const BoxDim& box,
                                                const Scalar3& ghost_width,
                                                const uint3& cell_dim,
                                                const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p, ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    return ci(ib, jb, kb);
    }

//! Kernel to estimate the colloid overlap volume and the depletant free volume
/*! \param n_sample Number of probe depletant particles to generate
    \param type Type of depletant particle
    \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_cell_size The size of each cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param d_cell_adj List of adjacent cells
    \param cadji Cell adjacency indexer
    \param cell_dim Dimensions of the cell list
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param a Size of rotation move (per type)
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param d_n_overlap_all Total overlap counter (output value)
    \param ghost_width Width of ghost layer
    \param d_params Per-type shape parameters
    \param d_overlaps Per-type pair interaction matrix
*/
template<class Shape>
__global__ void gpu_hpmc_free_volume_kernel(unsigned int n_sample,
                                            unsigned int type,
                                            Scalar4* d_postype,
                                            Scalar4* d_orientation,
                                            const unsigned int* d_cell_size,
                                            const Index3D ci,
                                            const Index2D cli,
                                            const unsigned int* d_excell_idx,
                                            const unsigned int* d_excell_size,
                                            const Index2D excli,
                                            const uint3 cell_dim,
                                            const unsigned int N,
                                            const unsigned int num_types,
                                            const uint16_t seed,
                                            const unsigned int rank,
                                            const unsigned int select,
                                            const uint64_t timestep,
                                            const unsigned int dim,
                                            const BoxDim box,
                                            unsigned int* d_n_overlap_all,
                                            Scalar3 ghost_width,
                                            const unsigned int* d_check_overlaps,
                                            Index2D overlap_idx,
                                            const typename Shape::param_type* d_params,
                                            unsigned int max_extra_bytes)
    {
    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0 && threadIdx.x == 0);
    unsigned int n_groups = blockDim.z;

    __shared__ unsigned int s_n_overlap;

    // determine sample idx
    unsigned int i;
    i = blockIdx.x * n_groups + group;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED(char, s_data)
    typename Shape::param_type* s_params = (typename Shape::param_type*)(&s_data[0]);
    unsigned int* s_check_overlaps = (unsigned int*)(s_params + num_types);
    unsigned int ntyppairs = overlap_idx.getNumElements();
    unsigned int* s_overlap = (unsigned int*)(&s_check_overlaps[ntyppairs]);

        // copy over parameters one int per thread for fast loads
        {
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
    char* s_extra = (char*)(s_overlap + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    if (master)
        {
        s_overlap[group] = 0;
        }

    if (master && group == 0)
        {
        s_n_overlap = 0;
        }

    __syncthreads();

    bool active = true;

    if (i >= n_sample)
        {
        active = false;
        }

    // one RNG per particle
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::ComputeFreeVolume, timestep, seed),
                               hoomd::Counter(rank, i));

    unsigned int my_cell;

    // test depletant position
    vec3<Scalar> pos_i;
    quat<Scalar> orientation_i;
    Shape shape_i(orientation_i, s_params[type]);

    if (active)
        {
        // select a random particle coordinate in the box
        Scalar xrand = hoomd::detail::generate_canonical<Scalar>(rng);
        Scalar yrand = hoomd::detail::generate_canonical<Scalar>(rng);
        Scalar zrand = hoomd::detail::generate_canonical<Scalar>(rng);

        if (dim == 2)
            {
            zrand = 0;
            }

        Scalar3 f = make_scalar3(xrand, yrand, zrand);
        pos_i = vec3<Scalar>(box.makeCoordinates(f));

        if (shape_i.hasOrientation())
            {
            shape_i.orientation = generateRandomOrientation(rng, dim);
            }

        // find cell the particle is in
        Scalar3 p = vec_to_scalar3(pos_i);
        my_cell = compute_cell_idx(p, box, ghost_width, cell_dim, ci);
        }

    if (active)
        {
        // loop over neighboring cells and check for overlaps
        unsigned int excell_size = d_excell_size[my_cell];

        for (unsigned int k = 0; k < excell_size; k += group_size)
            {
            unsigned int local_k = k + offset;
            if (local_k < excell_size)
                {
                // read in position, and orientation of neighboring particle
                unsigned int j = __ldg(&d_excell_idx[excli(local_k, my_cell)]);

                Scalar4 postype_j = __ldg(d_postype + j);
                Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(__ldg(d_orientation + j));

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i;
                r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                // check for overlaps
                ShortReal rsq = dot(r_ij, r_ij);
                ShortReal DaDb
                    = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                if (rsq * ShortReal(4.0) <= DaDb * DaDb)
                    {
                    // circumsphere overlap
                    unsigned int err_count;
                    if (s_check_overlaps[overlap_idx(typ_j, type)]
                        && test_overlap(r_ij, shape_i, shape_j, err_count))
                        {
                        s_overlap[group] = 1;
                        break;
                        }
                    }
                }
            }
        }

    __syncthreads();

    unsigned int overlap = s_overlap[group];

    if (master)
        {
        // this thread counts towards the total overlap volume
        if (overlap)
            {
            atomicAdd(&s_n_overlap, 1);
            }
        }

    __syncthreads();

    if (master && group == 0 && s_n_overlap)
        {
        // final tally into global mem
        atomicAdd(d_n_overlap_all, s_n_overlap);
        }
    }

//! Kernel driver for gpu_hpmc_free_volume_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or hipSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is
   instantiated for every shape at the bottom of this file.

    \ingroup hpmc_kernels
*/
template<class Shape>
hipError_t gpu_hpmc_free_volume(const hpmc_free_volume_args_t& args,
                                const typename Shape::param_type* d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_cell_size);
    assert(args.group_size >= 1);
    assert(args.group_size <= 32); // note, really should be warp size of the device
    assert(args.block_size % (args.stride * args.group_size) == 0);

    // reset counters
    hipMemsetAsync(args.d_n_overlap_all, 0, sizeof(unsigned int));

    // determine the maximum block size and clamp the input block size down
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(gpu_hpmc_free_volume_kernel<Shape>));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int n_groups
        = min(args.block_size, (unsigned int)max_block_size) / args.group_size / args.stride;

    dim3 threads(args.stride, args.group_size, n_groups);
    dim3 grid(args.n_sample / n_groups + 1, 1, 1);

    size_t shared_bytes = args.num_types * sizeof(typename Shape::param_type)
                          + n_groups * sizeof(unsigned int)
                          + args.overlap_idx.getNumElements() * sizeof(unsigned int);

    if (shared_bytes > args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("HPMC shape parameters exceed the available shared "
                                 "memory per block.");
        }

    unsigned int max_extra_bytes = static_cast<unsigned int>(args.devprop.sharedMemPerBlock
                                                             - attr.sharedSizeBytes - shared_bytes);

    // determine dynamically requested shared memory
    char* ptr = (char*)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < args.num_types; ++i)
        {
        d_params[i].allocate_shared(ptr, available_bytes);
        }
    const unsigned int extra_bytes = max_extra_bytes - available_bytes;

    shared_bytes += extra_bytes;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_hpmc_free_volume_kernel<Shape>),
                       dim3(grid),
                       dim3(threads),
                       shared_bytes,
                       0,
                       args.n_sample,
                       args.type,
                       args.d_postype,
                       args.d_orientation,
                       args.d_cell_size,
                       args.ci,
                       args.cli,
                       args.d_excell_idx,
                       args.d_excell_size,
                       args.excli,
                       args.cell_dim,
                       args.N,
                       args.num_types,
                       args.seed,
                       args.rank,
                       args.select,
                       args.timestep,
                       args.dim,
                       args.box,
                       args.d_n_overlap_all,
                       args.ghost_width,
                       args.d_check_overlaps,
                       args.overlap_idx,
                       d_params,
                       max_extra_bytes);

    return hipSuccess;
    }

#endif // __HIPCC__

    }; // end namespace detail

    } // end namespace hpmc

    } // end namespace hoomd

#endif // _COMPUTE_FREE_VOLUME_CUH_
