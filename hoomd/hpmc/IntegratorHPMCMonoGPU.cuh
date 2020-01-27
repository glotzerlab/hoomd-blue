#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/CachedAllocator.h"

#include "hoomd/hpmc/HPMCCounters.h"

#include <cassert>

namespace hpmc {

namespace gpu {

//! Wraps arguments to hpmc_* template functions
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a hpmc_args_t
    hpmc_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                hpmc_counters_t *_d_counters,
                const unsigned int _counters_pitch,
                const Index3D& _ci,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int _N,
                const unsigned int _N_ghost,
                const unsigned int _num_types,
                const unsigned int _seed,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int *_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _move_ratio,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _tpp,
                unsigned int *_d_reject_out_of_cell,
                Scalar4 *_d_trial_postype,
                Scalar4 *_d_trial_orientation,
                unsigned int *_d_trial_move_type,
                const unsigned int *_d_update_order_by_ptl,
                unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                unsigned int *_d_nlist,
                unsigned int *_d_nneigh,
                const unsigned int _maxn,
                unsigned int *_d_overflow,
                const bool _update_shape_param,
                const hipDeviceProp_t &_devprop,
                const GPUPartition& _gpu_partition)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_counters(_d_counters),
                  counters_pitch(_counters_pitch),
                  ci(_ci),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  N(_N),
                  N_ghost(_N_ghost),
                  num_types(_num_types),
                  seed(_seed),
                  d_d(_d),
                  d_a(_a),
                  d_check_overlaps(_check_overlaps),
                  overlap_idx(_overlap_idx),
                  move_ratio(_move_ratio),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  ghost_fraction(_ghost_fraction),
                  domain_decomposition(_domain_decomposition),
                  block_size(_block_size),
                  tpp(_tpp),
                  d_reject_out_of_cell(_d_reject_out_of_cell),
                  d_trial_postype(_d_trial_postype),
                  d_trial_orientation(_d_trial_orientation),
                  d_trial_move_type(_d_trial_move_type),
                  d_update_order_by_ptl(_d_update_order_by_ptl),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  d_nlist(_d_nlist),
                  d_nneigh(_d_nneigh),
                  maxn(_maxn),
                  d_overflow(_d_overflow),
                  update_shape_param(_update_shape_param),
                  devprop(_devprop),
                  gpu_partition(_gpu_partition)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    hpmc_counters_t *d_counters;      //!< Move accept/reject counters
    const unsigned int counters_pitch;         //!< Pitch of 2D array counters per GPU
    const Index3D& ci;                //!< Cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int N;             //!< Number of particles
    const unsigned int N_ghost;       //!< Number of ghost particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const Scalar* d_d;                //!< Maximum move displacement
    const Scalar* d_a;                //!< Maximum move angular displacement
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer into interaction matrix
    const unsigned int move_ratio;    //!< Ratio of translation to rotation moves
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int select;              //!< Current selection
    const Scalar3 ghost_fraction;     //!< Width of the inactive layer
    const bool domain_decomposition;  //!< Is domain decomposition mode enabled?
    unsigned int block_size;          //!< Block size to execute
    unsigned int tpp;                 //!< Threads per particle
    unsigned int *d_reject_out_of_cell;//!< Set to one to reject particle move
    Scalar4 *d_trial_postype;         //!< New positions (and type) of particles
    Scalar4 *d_trial_orientation;     //!< New orientations of particles
    unsigned int *d_trial_move_type;  //!< per particle flag, whether it is a translation (1) or rotation (2), or inactive (0)
    const unsigned int *d_update_order_by_ptl;  //!< Lookup of update order by particle index
    unsigned int *d_excell_idx;       //!< Expanded cell list
    const unsigned int *d_excell_size;//!< Size of expanded cells
    const Index2D& excli;             //!< Excell indexer
    unsigned int *d_nlist;        //!< Neighbor list of overlapping particles after trial move
    unsigned int *d_nneigh;       //!< Number of overlapping particles after trial move
    unsigned int maxn;                //!< Width of neighbor list
    unsigned int *d_overflow;         //!< Overflow condition for neighbor list
    const bool update_shape_param;    //!< True if shape parameters have changed
    const hipDeviceProp_t& devprop;     //!< CUDA device properties
    const GPUPartition& gpu_partition; //!< Multi-GPU partition
    };

//! Wraps arguments to kernel::hpmc_insert_depletants
/*! \ingroup hpmc_data_structs */
struct hpmc_implicit_args_t
    {
    //! Construct a hpmc_implicit_args_t
    hpmc_implicit_args_t(const unsigned int _depletant_type,
                         hpmc_implicit_counters_t *_d_implicit_count,
                         const unsigned int _implicit_counters_pitch,
                         const Scalar *_d_lambda,
                         const bool _repulsive,
                         const bool _quermass,
                         const Scalar _sweep_radius)
                : depletant_type(_depletant_type),
                  d_implicit_count(_d_implicit_count),
                  implicit_counters_pitch(_implicit_counters_pitch),
                  d_lambda(_d_lambda),
                  repulsive(_repulsive),
                  quermass(_quermass),
                  sweep_radius(_sweep_radius)

        { };

    const unsigned int depletant_type;             //!< Particle type of depletant
    hpmc_implicit_counters_t *d_implicit_count;    //!< Active cell acceptance/rejection counts
    const unsigned int implicit_counters_pitch;    //!< Pitch of 2D array counters per device
    const Scalar *d_lambda;                        //!< Mean number of depletants to insert in excluded volume
    const bool repulsive;                          //!< True if the fugacity is negative
    const bool quermass;                           //!< Enable quermass mode?
    const Scalar sweep_radius;                     //!< Sweep radius in quermass mode
    };

//! Wraps arguments for hpmc_update_pdata
struct hpmc_update_args_t
    {
    //! Construct an hpmc_update_args_t
    hpmc_update_args_t(Scalar4 *_d_postype,
        Scalar4 *_d_orientation,
        hpmc_counters_t *_d_counters,
        const unsigned int _N,
        const Scalar4 *_d_trial_postype,
        const Scalar4 *_d_trial_orientation,
        const unsigned int *_d_trial_move_type,
        const unsigned int *_d_reject,
        const unsigned int _maxn,
        const unsigned int _block_size)
        : d_postype(_d_postype),
          d_orientation(_d_orientation),
          d_counters(_d_counters),
          N(_N),
          d_trial_postype(_d_trial_postype),
          d_trial_orientation(_d_trial_orientation),
          d_trial_move_type(_d_trial_move_type),
          d_reject(_d_reject),
          maxn(_maxn),
          block_size(_block_size)
     {}

    //! See hpmc_args_t for documentation on the meaning of these parameters
    Scalar4 *d_postype;
    Scalar4 *d_orientation;
    hpmc_counters_t *d_counters;
    const unsigned int N;
    const Scalar4 *d_trial_postype;
    const Scalar4 *d_trial_orientation;
    const unsigned int *d_trial_move_type;
    const unsigned int *d_reject;
    const unsigned int maxn;
    const unsigned int block_size;
    };

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size);

//! Driver for kernel::hpmc_gen_moves()
template< class Shape >
void hpmc_gen_moves(const hpmc_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_narrow_phase()
template< class Shape >
void hpmc_narrow_phase(const hpmc_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_update_pdata()
template< class Shape >
void hpmc_update_pdata(const hpmc_update_args_t& args, const typename Shape::param_type *params);

//! Driver for kernel::hpmc_insert_depletants()
template< class Shape >
void hpmc_insert_depletants(const hpmc_args_t& args, const hpmc_implicit_args_t& implicit_args, const typename Shape::param_type *params);

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size);

void hpmc_accept(const unsigned int *d_ptl_by_update_order,
                 const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject,
                 unsigned int *d_reject_out,
                 const unsigned int *d_nneigh,
                 const unsigned int *d_nlist,
                 const unsigned int N_old,
                 const unsigned int N,
                 const unsigned int maxn,
                 unsigned int *d_condition,
                 const unsigned int block_size);

#ifdef __HIPCC__
namespace kernel
{

//! Device function to compute the cell that a particle sits in
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
                                                   const BoxDim& box,
                                                   const Scalar3& ghost_width,
                                                   const uint3& cell_dim,
                                                   const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
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
    if (f.x >= Scalar(0.0) && f.x < Scalar(1.0) && f.y >= Scalar(0.0) && f.y < Scalar(1.0) && f.z >= Scalar(0.0) && f.z < Scalar(1.0))
        return ci(ib,jb,kb);
    else
        return 0xffffffff;
    }


//! Propose trial moves
template< class Shape, unsigned int dim >
__global__ void hpmc_gen_moves(Scalar4 *d_postype,
                           Scalar4 *d_orientation,
                           const unsigned int N,
                           const Index3D ci,
                           const uint3 cell_dim,
                           const Scalar3 ghost_width,
                           const unsigned int num_types,
                           const unsigned int seed,
                           const Scalar* d_d,
                           const Scalar* d_a,
                           const unsigned int move_ratio,
                           const unsigned int timestep,
                           const BoxDim box,
                           const unsigned int select,
                           const Scalar3 ghost_fraction,
                           const bool domain_decomposition,
                           Scalar4 *d_trial_postype,
                           Scalar4 *d_trial_orientation,
                           unsigned int *d_trial_move_type,
                           unsigned int *d_reject_out_of_cell,
                           const typename Shape::param_type *d_params)
    {
    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar *s_d = (Scalar *)(s_params + num_types);
    Scalar *s_a = (Scalar *)(s_d + num_types);

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

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // identify the particle that this thread handles
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // return early if we are not handling a particle
    if (idx >= N)
        return;

    // read in the position and orientation of our particle.
    Scalar4 postype_i = d_postype[idx];
    Scalar4 orientation_i = make_scalar4(1,0,0,0);

    unsigned int typ_i = __scalar_as_int(postype_i.w);
    Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

    if (shape_i.hasOrientation())
        orientation_i = d_orientation[idx];

    shape_i.orientation = quat<Scalar>(orientation_i);

    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
    unsigned int old_cell = computeParticleCell(vec_to_scalar3(pos_i), box, ghost_width, cell_dim, ci);

    // for domain decomposition simulations, we need to leave all particles in the inactive region alone
    // in order to avoid even more divergence, this is done by setting the move_active flag
    // overlap checks are still processed, but the final move acceptance will be skipped
    bool move_active = true;
    if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
        move_active = false;

    // make the move
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoTrialMove, idx, seed, select, timestep);

    unsigned int reject = 0;

    unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
    bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

    if (move_active)
        {
        if (move_type_translate)
            {
            move_translate(pos_i, rng, s_d[typ_i], dim);

            // need to reject any move that puts the particle in the inactive region
            if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                move_active = false;
            }
        else
            {
            move_rotate<dim>(shape_i.orientation, rng, s_a[typ_i]);
            }
        }

    if (move_active && move_type_translate)
        {
        // check if the particle remains in its cell
        Scalar3 xnew_i = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
        unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width, cell_dim, ci);

        if (new_cell != old_cell)
            reject = 1;
        }

    // stash the trial move in global memory
    d_trial_postype[idx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(typ_i));
    d_trial_orientation[idx] = quat_to_scalar4(shape_i.orientation);

    // 0==inactive, 1==translation, 2==rotation
    d_trial_move_type[idx] = move_active ? (move_type_translate ? 1 : 2) : 0;

    // initialize reject flag
    d_reject_out_of_cell[idx] = reject;
    }

//! Check narrow-phase overlaps
template< class Shape >
__global__ void hpmc_narrow_phase(Scalar4 *d_postype,
                           Scalar4 *d_orientation,
                           Scalar4 *d_trial_postype,
                           Scalar4 *d_trial_orientation,
                           const unsigned int *d_excell_idx,
                           const unsigned int *d_excell_size,
                           const Index2D excli,
                           unsigned int *d_nlist,
                           unsigned int *d_nneigh,
                           const unsigned int maxn,
                           hpmc_counters_t *d_counters,
                           const unsigned int num_types,
                           const BoxDim box,
                           const Scalar3 ghost_width,
                           const uint3 cell_dim,
                           const Index3D ci,
                           const unsigned int N_old,
                           const unsigned int N_new,
                           const unsigned int *d_check_overlaps,
                           const Index2D overlap_idx,
                           const typename Shape::param_type *d_params,
                           unsigned int *d_overflow,
                           const unsigned int max_extra_bytes,
                           const unsigned int max_queue_size,
                           const unsigned int work_offset,
                           const unsigned int nwork)
    {
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int *s_idx_group = (unsigned int*)(s_type_group + n_groups);
    unsigned int *s_nneigh_group = (unsigned int *)(s_idx_group + n_groups);

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
    char *s_extra = (char *)(s_nneigh_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }

    if (master)
        {
        // reset number of neighbors
        s_nneigh_group[group] = 0;
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

        // find the cell this particle should be in
        vec3<Scalar> pos_i_old(d_postype[idx]);
        my_cell = computeParticleCell(vec_to_scalar3(pos_i_old), box, ghost_width, cell_dim, ci);

        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = type_i;
            s_orientation_group[group] = d_trial_orientation[idx];
            s_idx_group[group] = idx;
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
        overlap_checks += 2*excell_size;
        }

    // loop while still searching

    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if ((k >> 1) < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found

            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && (k >> 1) < excell_size)
                {
                // build some shapes, but we only need them to get diameters, so don't load orientations
                // build shape i from shared memory
                vec3<Scalar> pos_i(s_pos_group[group]);
                Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                bool old = k & 1;

                // prefetch next j
                j = next_j;
                k += group_size;
                if ((k >> 1) < excell_size)
                    {
                    next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                    }

                // check particle circumspheres

                // load particle j (always load ghosts from particle data)
                const Scalar4 postype_j = (old || j >= N_new) ? d_postype[j] : d_trial_postype[j];
                unsigned int type_j = __scalar_as_int(postype_j.w);
                vec3<Scalar> pos_j(postype_j);
                Shape shape_j(quat<Scalar>(), s_params[type_j]);

                // place ourselves into the minimum image
                vec3<Scalar> r_ij = pos_j - pos_i;
                r_ij = box.minImage(r_ij);

                if (idx != j && (old || j < N_new)
                    && check_circumsphere_overlap(r_ij, shape_i, shape_j))
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                    if (insert_point < max_queue_size)
                        {
                        s_queue_gid[insert_point] = group;
                        s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
                        }
                    else
                        {
                        // or back up if the queue is already full
                        // we will recheck and insert this on the next time through
                        k -= group_size;
                        }
                    }
                } // end while (s_queue_size < max_queue_size && (k>>1) < excell_size)
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
            unsigned int check_j_flag = s_queue_j[tidx_1d];
            bool check_old = check_j_flag & 1;
            unsigned int check_j  = check_j_flag >> 1;

            Scalar4 postype_j;
            Scalar4 orientation_j;
            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = check_old ? quat<Scalar>(d_orientation[check_j]) : quat<Scalar>(d_trial_orientation[check_j]);

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)]
                && test_overlap(r_ij, shape_i, shape_j, overlap_err_count))
                {
                // write out to global memory
                unsigned int n = atomicAdd(&s_nneigh_group[check_group], 1);
                if (n < maxn)
                    {
                    d_nlist[n+maxn*s_idx_group[check_group]] = check_old ? check_j : (check_j + N_old);
                    }
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && (k >> 1) < excell_size)
            atomicAdd(&s_still_searching, 1);

        __syncthreads();
        } // end while (s_still_searching)

    if (active && master)
        {
        // overflowed?
        unsigned int nneigh = s_nneigh_group[group];
        if (nneigh > maxn)
            {
            #if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
            #else
            atomicMax(d_overflow, nneigh);
            #endif
            }

        // write out number of neighbors to global mem
        d_nneigh[idx] = nneigh;
        }

    if (master)
        {
        atomicAdd(&s_overlap_checks, overlap_checks);
        atomicAdd(&s_overlap_err_count, overlap_err_count);
        }

    __syncthreads();

    if (master && group == 0)
        {
        // write out counters to global memory
        #if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd_system(&d_counters->overlap_checks, s_overlap_checks);
        #else
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        #endif
        }
    }

//! Kernel to insert depletants on-the-fly
template< class Shape, bool quermass >
__global__ void hpmc_insert_depletants(const Scalar4 *d_trial_postype,
                                     const Scalar4 *d_trial_orientation,
                                     const unsigned int *d_trial_move_type,
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const Index3D ci,
                                     const unsigned int N_old,
                                     const unsigned int N_new,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     unsigned int *d_reject,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes,
                                     unsigned int depletant_type,
                                     hpmc_implicit_counters_t *d_implicit_counters,
                                     const Scalar *d_lambda,
                                     unsigned int *d_nneigh,
                                     unsigned int *d_nlist,
                                     const unsigned int maxn,
                                     unsigned int *d_overflow,
                                     bool repulsive,
                                     unsigned int work_offset,
                                     unsigned int max_depletant_queue_size,
                                     const Scalar sweep_radius)
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    unsigned int err_count = 0;

    // shared particle configuation
    __shared__ Scalar4 s_orientation_i_new;
    __shared__ Scalar4 s_orientation_i_old;
    __shared__ Scalar3 s_pos_i_new;
    __shared__ Scalar3 s_pos_i_old;
    __shared__ unsigned int s_type_i;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    // shared queue variables
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_adding_depletants;
    __shared__ unsigned int s_depletant_queue_size;

    __shared__ unsigned int s_nneigh;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED( char, s_data)
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_pos_group + n_groups);
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

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        }

    __syncthreads();

    // identify the active cell that this thread handles
    unsigned int i = blockIdx.x + work_offset;

    // load updated particle position
    if (master && group == 0)
        {
        Scalar4 postype_i_new = d_trial_postype[i];
        Scalar4 postype_i_old = d_postype[i];
        s_pos_i_new = make_scalar3(postype_i_new.x, postype_i_new.y, postype_i_new.z);
        s_pos_i_old = make_scalar3(postype_i_old.x, postype_i_old.y, postype_i_old.z);
        s_type_i = __scalar_as_int(postype_i_new.w);

        s_orientation_i_new = d_trial_orientation[i];
        s_orientation_i_old = d_orientation[i];
        }

    __syncthreads();

    // generate random number of depletants from Poisson distribution
    hoomd::RandomGenerator rng_poisson(hoomd::RNGIdentifier::HPMCDepletantNum, i, seed, timestep, select*num_types + depletant_type);
    Index2D typpair_idx(num_types);
    unsigned int n_depletants = hoomd::PoissonDistribution<Scalar>(d_lambda[typpair_idx(depletant_type,s_type_i)])(rng_poisson);

    unsigned int overlap_checks = 0;
    unsigned int n_inserted = 0;

    // find the cell this particle should be in
    unsigned int my_cell = computeParticleCell(s_pos_i_old, box, ghost_width, cell_dim, ci);

    detail::OBB obb_i;
        {
        // get shape OBB
        Shape shape_i(quat<Scalar>(!repulsive ? d_orientation[i] : d_trial_orientation[i]), s_params[s_type_i]);
        obb_i = shape_i.getOBB(repulsive ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old));

        // extend by depletant radius
        Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
        Scalar r_dep(0.5*shape_test.getCircumsphereDiameter());
        Scalar range = sweep_radius + r_dep;
        obb_i.lengths.x += range;
        obb_i.lengths.y += range;
        obb_i.lengths.z += range;
        }

    // load number of neighbors
    if (master && group == 0)
        s_nneigh = d_nneigh[i];

    // sync since we'll be overwriting d_nneigh[i] and so that s_nneigh is available to all threads
    __syncthreads();

    s_depletant_queue_size = 0;
    s_adding_depletants = 1;

    __syncthreads();

    unsigned int i_dep = group_size*group+offset;

    while (s_adding_depletants)
        {
        while (s_depletant_queue_size < max_depletant_queue_size && i_dep < n_depletants)
            {
            // one RNG per depletant
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants, seed+i, i_dep, select*num_types + depletant_type, timestep);

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
                shape_i.orientation = quat<Scalar>(s_orientation_i_old);
            vec3<Scalar> r_ij = vec3<Scalar>(s_pos_i_old) - pos_test;
            bool overlap_old = (s_check_overlaps[overlap_idx(s_type_i, depletant_type)]
                && check_circumsphere_overlap(r_ij, shape_test, shape_i, OverlapReal(0.0), sweep_radius)
                && test_overlap(r_ij, shape_test, shape_i, err_count, Scalar(0.0), sweep_radius));

            if (shape_i.hasOrientation())
                shape_i.orientation = quat<Scalar>(s_orientation_i_new);
            r_ij = vec3<Scalar>(s_pos_i_new) - pos_test;
            bool overlap_new = (s_check_overlaps[overlap_idx(s_type_i, depletant_type)]
                && check_circumsphere_overlap(r_ij, shape_test, shape_i, OverlapReal(0.0), sweep_radius)
                && test_overlap(r_ij, shape_test, shape_i, err_count, Scalar(0.0), sweep_radius));
            bool add_to_queue = (!repulsive && overlap_old && !overlap_new) || (repulsive && !overlap_old && overlap_new);

            if (add_to_queue)
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
            i_dep += group_size*n_groups;
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
            hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants, seed+i, i_dep_queue, select*num_types + depletant_type, timestep);

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

        __syncthreads();

        // counters to track progress through the loop over potential neighbors
        unsigned int excell_size;
        unsigned int k = offset;

        if (active)
            {
            excell_size = d_excell_size[my_cell];

            if (master)
                overlap_checks += 2*excell_size;
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
                if ((k>>1) < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k>>1, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                // and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && (k>>1) < excell_size)
                    {
                    // which configuration are we handling?
                    bool old = k & 1;

                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1,0,0,0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if ((k >> 1) < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k>>1, my_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for circumsphere check
                    // for ghosts always load particle data
                    postype_j = (old || j >= N_new) ? d_postype[j] : d_trial_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool insert_in_queue;

                    if (!quermass)
                        {
                        insert_in_queue = s_check_overlaps[overlap_idx(depletant_type, type_j)]
                            && i != j && (old || j < N_new)
                            && check_circumsphere_overlap(r_jk, shape_test, shape_j);
                        }
                    else
                        {
                        // check triple overlap of circumspheres
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - (repulsive ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old));
                        r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                        Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
                        insert_in_queue = (i != j) && (old || j < N_new)
                            && check_circumsphere_overlap_three(shape_i, shape_j, shape_test, r_ij, -r_jk+r_ij, sweep_radius, sweep_radius, 0.0);
                        }

                    if (insert_in_queue)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
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
                unsigned int check_j_flag = s_queue_j[tidx_1d];
                bool check_old = check_j_flag & 1;
                unsigned int check_j  = check_j_flag >> 1;

                Scalar4 postype_j;
                Scalar4 orientation_j;
                vec3<Scalar> r_jk;

                // build depletant shape from shared memory
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

                // build shape j from global memory
                postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
                orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(check_old ? d_orientation[check_j] : d_trial_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                bool insert_in_nlist;
                if (!quermass)
                    {
                    insert_in_nlist = test_overlap(r_jk, shape_test, shape_j, err_count);
                    }
                else
                    {
                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - (repulsive ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old));
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));
                    Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
                    if (shape_i.hasOrientation())
                        shape_i.orientation = repulsive ? quat<Scalar>(s_orientation_i_new) : quat<Scalar>(s_orientation_i_old);
                    insert_in_nlist = test_overlap_intersection(shape_i, shape_j, shape_test, r_ij, -r_jk+r_ij, err_count,
                                        sweep_radius, sweep_radius, 0.0);
                    }
                if (insert_in_nlist)
                    {
                    // write out to global memory
                    unsigned int n = atomicAdd(&s_nneigh, 1);
                    if (n < maxn)
                        {
                        d_nlist[n+maxn*i] = check_old ? check_j : (check_j + N_old);
                        }
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && (k>>1) < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        // do we still need to process depletants?
        __syncthreads();
        if (master && group == 0)
            s_depletant_queue_size = 0;
        if (i_dep < n_depletants)
            atomicAdd(&s_adding_depletants, 1);
        __syncthreads();
        } // end loop over depletants

    if (master && group == 0)
        {
        // overflowed?
        unsigned int nneigh = s_nneigh;
        if (nneigh > maxn)
            {
            #if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
            #else
            atomicMax(d_overflow, nneigh);
            #endif
            }

        // write out number of neighbors to global mem
        d_nneigh[i] = nneigh;
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    // count the overlap checks
    atomicAdd(&s_overlap_checks, overlap_checks);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        #if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd_system(&d_counters->overlap_checks, s_overlap_checks);
        #else
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        #endif

        // increment number of inserted depletants
        #if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_implicit_counters[depletant_type].insert_count, n_depletants);
        #else
        atomicAdd(&d_implicit_counters[depletant_type].insert_count, n_depletants);
        #endif
        }
    }

//! Kernel to update particle data and statistics after acceptance
template<class Shape>
__global__ void hpmc_update_pdata(Scalar4 *d_postype,
                                  Scalar4 *d_orientation,
                                  hpmc_counters_t *d_counters,
                                  const unsigned int N,
                                  const Scalar4 *d_trial_postype,
                                  const Scalar4 *d_trial_orientation,
                                  const unsigned int *d_trial_move_type,
                                  const unsigned int *d_reject,
                                  const unsigned int maxn,
                                  const typename Shape::param_type *d_params)
    {
    // determine which update step we are handling
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;

    // initialize the shared memory array for communicating overlaps
    if (threadIdx.x == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        }

    __syncthreads();

    if (idx < N)
        {
        unsigned int move_type = d_trial_move_type[idx];
        bool move_active = move_type > 0;
        bool move_type_translate = move_type == 1;
        bool accept = !d_reject[idx];

        unsigned int type_i = __scalar_as_int(d_postype[idx].w);
        Shape shape_i(quat<Scalar>(), d_params[type_i]);

        bool ignore_stats = shape_i.ignoreStatistics();

        // update the data if accepted
        if (move_active)
            {
            if (accept)
                {
                // write out the updated position and orientation
                d_postype[idx] = d_trial_postype[idx];
                d_orientation[idx] = d_trial_orientation[idx];
                }

            if (!ignore_stats && accept && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accept && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accept && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accept && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        }

    __syncthreads();

    // final tally into global mem
    if (threadIdx.x == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        }
    }

} // end namespace kernel

//! Kernel driver for kernel::hpmc_gen_moves
template< class Shape >
void hpmc_gen_moves(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_d);
    assert(args.d_a);

    if (args.dim == 2)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_gen_moves<Shape,2>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
        unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar));

        if (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("hpmc::kernel::gen_moves() exceeds shared memory limits");

        // setup the grid to run the kernel
        dim3 threads( block_size, 1, 1);
        dim3 grid((args.N+block_size-1)/block_size,1,1);

        hipLaunchKernelGGL((kernel::hpmc_gen_moves<Shape,2>), grid, threads, shared_bytes, 0,
                                                                     args.d_postype,
                                                                     args.d_orientation,
                                                                     args.N,
                                                                     args.ci,
                                                                     args.cell_dim,
                                                                     args.ghost_width,
                                                                     args.num_types,
                                                                     args.seed,
                                                                     args.d_d,
                                                                     args.d_a,
                                                                     args.move_ratio,
                                                                     args.timestep,
                                                                     args.box,
                                                                     args.select,
                                                                     args.ghost_fraction,
                                                                     args.domain_decomposition,
                                                                     args.d_trial_postype,
                                                                     args.d_trial_orientation,
                                                                     args.d_trial_move_type,
                                                                     args.d_reject_out_of_cell,
                                                                     params
                                                                );
        }
    else
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_gen_moves<Shape,3>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
        unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar));

        if (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("hpmc::kernel::gen_moves() exceeds shared memory limits");

        // setup the grid to run the kernel
        dim3 threads( block_size, 1, 1);
        dim3 grid((args.N+block_size-1)/block_size,1,1);

        hipLaunchKernelGGL((kernel::hpmc_gen_moves<Shape,3>), grid, threads, shared_bytes, 0,
                                                                     args.d_postype,
                                                                     args.d_orientation,
                                                                     args.N,
                                                                     args.ci,
                                                                     args.cell_dim,
                                                                     args.ghost_width,
                                                                     args.num_types,
                                                                     args.seed,
                                                                     args.d_d,
                                                                     args.d_a,
                                                                     args.move_ratio,
                                                                     args.timestep,
                                                                     args.box,
                                                                     args.select,
                                                                     args.ghost_fraction,
                                                                     args.domain_decomposition,
                                                                     args.d_trial_postype,
                                                                     args.d_trial_orientation,
                                                                     args.d_trial_move_type,
                                                                     args.d_reject_out_of_cell,
                                                                     params
                                                                );
 
        }
    }

//! Kernel driver for kernel::hpmc_narrow_phase
template< class Shape >
void hpmc_narrow_phase(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_narrow_phase<Shape>));
        max_block_size = attr.maxThreadsPerBlock;
        if (max_block_size % args.devprop.warpSize)
            // handle non-sensical return values from hipFuncGetAttributes
            max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
        }

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

    unsigned int tpp = min(args.tpp,run_block_size);
    unsigned int n_groups = run_block_size/tpp;
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type)
        + args.overlap_idx.getNumElements() * sizeof(unsigned int);

    unsigned int shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
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
        n_groups = run_block_size / tpp;
        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3))
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
        // required for memory coherency
        hipDeviceSynchronize();

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
    dim3 thread(tpp, n_groups, 1);
    
    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::hpmc_narrow_phase<Shape>, grid, thread, shared_bytes, 0, 
            args.d_postype, args.d_orientation, args.d_trial_postype, args.d_trial_orientation,
            args.d_excell_idx, args.d_excell_size, args.excli,
            args.d_nlist, args.d_nneigh, args.maxn, args.d_counters+idev*args.counters_pitch, args.num_types,
            args.box, args.ghost_width, args.cell_dim, args.ci, args.N + args.N_ghost, args.N, args.d_check_overlaps,
            args.overlap_idx, params,
            args.d_overflow, max_extra_bytes, max_queue_size, range.first, nwork);
        }
    }

//! Kernel driver for kernel::insert_depletants()
/*! \param args Bundled arguments
    \param implicit_args Bundled arguments related to depletants
    \param d_params Per-type shape parameters

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
void hpmc_insert_depletants(const hpmc_args_t& args, const hpmc_implicit_args_t& implicit_args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_check_overlaps);

    if (!implicit_args.quermass)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&kernel::hpmc_insert_depletants<Shape, false>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp,block_size);
        unsigned int n_groups = block_size / tpp;
        unsigned int max_queue_size = n_groups*tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                   args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3)) +
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
            max_queue_size = n_groups*tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3)) +
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
            // required for memory coherency
            hipDeviceSynchronize();

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
        dim3 threads(tpp, n_groups,1);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            // 1 block per particle
            dim3 grid( range.second-range.first, 1, 1);

            hipLaunchKernelGGL((kernel::hpmc_insert_depletants<Shape, false>), dim3(grid), dim3(threads), shared_bytes, 0, args.d_trial_postype,
                                                                         args.d_trial_orientation,
                                                                         args.d_trial_move_type,
                                                                         args.d_postype,
                                                                         args.d_orientation,
                                                                         args.d_counters,
                                                                         args.d_excell_idx,
                                                                         args.d_excell_size,
                                                                         args.excli,
                                                                         args.cell_dim,
                                                                         args.ghost_width,
                                                                         args.ci,
                                                                         args.N + args.N_ghost,
                                                                         args.N,
                                                                         args.num_types,
                                                                         args.seed,
                                                                         args.d_check_overlaps,
                                                                         args.overlap_idx,
                                                                         args.timestep,
                                                                         args.dim,
                                                                         args.box,
                                                                         args.select,
                                                                         args.d_reject_out_of_cell,
                                                                         params,
                                                                         max_queue_size,
                                                                         max_extra_bytes,
                                                                         implicit_args.depletant_type,
                                                                         implicit_args.d_implicit_count + idev*implicit_args.implicit_counters_pitch,
                                                                         implicit_args.d_lambda,
                                                                         args.d_nneigh,
                                                                         args.d_nlist,
                                                                         args.maxn,
                                                                         args.d_overflow,
                                                                         implicit_args.repulsive,
                                                                         range.first,
                                                                         max_depletant_queue_size,
                                                                         implicit_args.sweep_radius);
            }
        }
    else // !quermass
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_insert_depletants<Shape, true>));
            max_block_size = attr.maxThreadsPerBlock;
            if (max_block_size % args.devprop.warpSize)
                // handle non-sensical return values from hipFuncGetAttributes
                max_block_size = (max_block_size/args.devprop.warpSize-1)*args.devprop.warpSize;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp,block_size);
        unsigned int n_groups = block_size / tpp;
        unsigned int max_queue_size = n_groups*tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                   args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3)) +
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
            max_queue_size = n_groups*tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3)) +
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
            // required for memory coherency
            hipDeviceSynchronize();

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
        dim3 threads(tpp, n_groups,1);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            // 1 block per particle
            dim3 grid( range.second-range.first, 1, 1);

            hipLaunchKernelGGL((kernel::hpmc_insert_depletants<Shape, true>), dim3(grid), dim3(threads), shared_bytes, 0, args.d_trial_postype,
                                                                         args.d_trial_orientation,
                                                                         args.d_trial_move_type,
                                                                         args.d_postype,
                                                                         args.d_orientation,
                                                                         args.d_counters,
                                                                         args.d_excell_idx,
                                                                         args.d_excell_size,
                                                                         args.excli,
                                                                         args.cell_dim,
                                                                         args.ghost_width,
                                                                         args.ci,
                                                                         args.N + args.N_ghost,
                                                                         args.N,
                                                                         args.num_types,
                                                                         args.seed,
                                                                         args.d_check_overlaps,
                                                                         args.overlap_idx,
                                                                         args.timestep,
                                                                         args.dim,
                                                                         args.box,
                                                                         args.select,
                                                                         args.d_reject_out_of_cell,
                                                                         params,
                                                                         max_queue_size,
                                                                         max_extra_bytes,
                                                                         implicit_args.depletant_type,
                                                                         implicit_args.d_implicit_count + idev*implicit_args.implicit_counters_pitch,
                                                                         implicit_args.d_lambda,
                                                                         args.d_nneigh,
                                                                         args.d_nlist,
                                                                         args.maxn,
                                                                         args.d_overflow,
                                                                         implicit_args.repulsive,
                                                                         range.first,
                                                                         max_depletant_queue_size,
                                                                         implicit_args.sweep_radius);
            }
        }
    }

//! Driver for kernel::hpmc_update_pdata()
template<class Shape>
void hpmc_update_pdata(const hpmc_update_args_t& args, const typename Shape::param_type *params)
    {
    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static hipFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_update_pdata<Shape>));
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
    unsigned int num_blocks = (args.N + block_size - 1)/block_size;
    hipLaunchKernelGGL((kernel::hpmc_update_pdata<Shape>), dim3(num_blocks), dim3(block_size), 0, 0, 
        args.d_postype,
        args.d_orientation,
        args.d_counters,
        args.N,
        args.d_trial_postype,
        args.d_trial_orientation,
        args.d_trial_move_type,
        args.d_reject,
        args.maxn,
        params);
    }
#endif


} // end namespace gpu

} // end namespace hpmc
