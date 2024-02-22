// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BoxDim.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/HPMCCounters.h"
#include <hip/hip_runtime.h>

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Wraps arguments to hpmc_* template functions
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a hpmc_args_t
    hpmc_args_t(const Scalar4* _d_postype,
                const Scalar4* _d_orientation,
                const Scalar4* _d_vel,
                hpmc_counters_t* _d_counters,
                const unsigned int _counters_pitch,
                const Index3D& _ci,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int _N,
                const unsigned int _num_types,
                const uint16_t _seed,
                const unsigned int _rank,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int* _check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _move_ratio,
                const uint64_t _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _tpp,
                const unsigned int _overlap_threads,
                const bool _have_auxilliary_variable,
                unsigned int* _d_reject_out_of_cell,
                Scalar4* _d_trial_postype,
                Scalar4* _d_trial_orientation,
                Scalar4* _d_trial_vel,
                unsigned int* _d_trial_move_type,
                const unsigned int* _d_update_order_by_ptl,
                unsigned int* _d_excell_idx,
                const unsigned int* _d_excell_size,
                const Index2D& _excli,
                const unsigned int* _d_reject_in,
                unsigned int* _d_reject_out,
                const hipDeviceProp_t& _devprop,
                const GPUPartition& _gpu_partition,
                const hipStream_t* _streams)
        : d_postype(_d_postype), d_orientation(_d_orientation), d_vel(_d_vel),
          d_counters(_d_counters), counters_pitch(_counters_pitch), ci(_ci), cell_dim(_cell_dim),
          ghost_width(_ghost_width), N(_N), num_types(_num_types), seed(_seed), rank(_rank),
          d_d(_d), d_a(_a), d_check_overlaps(_check_overlaps), overlap_idx(_overlap_idx),
          move_ratio(_move_ratio), timestep(_timestep), dim(_dim), box(_box), select(_select),
          ghost_fraction(_ghost_fraction), domain_decomposition(_domain_decomposition),
          block_size(_block_size), tpp(_tpp), overlap_threads(_overlap_threads),
          have_auxilliary_variable(_have_auxilliary_variable),
          d_reject_out_of_cell(_d_reject_out_of_cell), d_trial_postype(_d_trial_postype),
          d_trial_orientation(_d_trial_orientation), d_trial_vel(_d_trial_vel),
          d_trial_move_type(_d_trial_move_type), d_update_order_by_ptl(_d_update_order_by_ptl),
          d_excell_idx(_d_excell_idx), d_excell_size(_d_excell_size), excli(_excli),
          d_reject_in(_d_reject_in), d_reject_out(_d_reject_out), devprop(_devprop),
          gpu_partition(_gpu_partition), streams(_streams) {};

    const Scalar4* d_postype;             //!< postype array
    const Scalar4* d_orientation;         //!< orientation array
    const Scalar4* d_vel;                 //!< velocities array (used to store auxilliary variables)
    hpmc_counters_t* d_counters;          //!< Move accept/reject counters
    const unsigned int counters_pitch;    //!< Pitch of 2D array counters per GPU
    const Index3D& ci;                    //!< Cell indexer
    const uint3& cell_dim;                //!< Cell dimensions
    const Scalar3& ghost_width;           //!< Width of the ghost layer
    const unsigned int N;                 //!< Number of particles
    const unsigned int num_types;         //!< Number of particle types
    const uint16_t seed;                  //!< RNG seed
    const unsigned int rank;              //!< MPI Rank
    const Scalar* d_d;                    //!< Maximum move displacement
    const Scalar* d_a;                    //!< Maximum move angular displacement
    const unsigned int* d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;           //!< Indexer into interaction matrix
    const unsigned int move_ratio;        //!< Ratio of translation to rotation moves
    const uint64_t timestep;              //!< Current time step
    const unsigned int dim;               //!< Number of dimensions
    const BoxDim box;                     //!< Current simulation box
    unsigned int select;                  //!< Current selection
    const Scalar3 ghost_fraction;         //!< Width of the inactive layer
    const bool domain_decomposition;      //!< Is domain decomposition mode enabled?
    unsigned int block_size;              //!< Block size to execute
    unsigned int tpp;                     //!< Threads per particle
    unsigned int overlap_threads;         //!< Number of parallel threads per overlap check
    const bool have_auxilliary_variable;  //!< True if we are using the velocity field to store
                                          //!< auxilliary state information
    unsigned int* d_reject_out_of_cell;   //!< Set to one to reject particle move
    Scalar4* d_trial_postype;             //!< New positions (and type) of particles
    Scalar4* d_trial_orientation;         //!< New orientations of particles
    Scalar4* d_trial_vel;                 //!< New velocities (auxilliary variables) of particles
    unsigned int* d_trial_move_type;      //!< per particle flag, whether it is a translation (1) or
                                          //!< rotation (2), or inactive (0)
    const unsigned int* d_update_order_by_ptl; //!< Lookup of update order by particle index
    unsigned int* d_excell_idx;                //!< Expanded cell list
    const unsigned int* d_excell_size;         //!< Size of expanded cells
    const Index2D& excli;                      //!< Excell indexer
    const unsigned int* d_reject_in;           //!< Reject flags per particle (in)
    unsigned int* d_reject_out;                //!< Reject flags per particle (out)
    const hipDeviceProp_t& devprop;            //!< CUDA device properties
    const GPUPartition& gpu_partition;         //!< Multi-GPU partition
    const hipStream_t* streams;                //!< kernel streams
    };

//! Wraps arguments for hpmc_update_pdata
struct hpmc_update_args_t
    {
    //! Construct an hpmc_update_args_t
    hpmc_update_args_t(Scalar4* _d_postype,
                       Scalar4* _d_orientation,
                       Scalar4* _d_vel,
                       hpmc_counters_t* _d_counters,
                       unsigned int _counters_pitch,
                       const GPUPartition& _gpu_partition,
                       const bool _have_auxilliary_variable,
                       const Scalar4* _d_trial_postype,
                       const Scalar4* _d_trial_orientation,
                       const Scalar4* _d_trial_vel,
                       const unsigned int* _d_trial_move_type,
                       const unsigned int* _d_reject,
                       const unsigned int _block_size)
        : d_postype(_d_postype), d_orientation(_d_orientation), d_vel(_d_vel),
          d_counters(_d_counters), counters_pitch(_counters_pitch), gpu_partition(_gpu_partition),
          have_auxilliary_variable(_have_auxilliary_variable), d_trial_postype(_d_trial_postype),
          d_trial_orientation(_d_trial_orientation), d_trial_vel(_d_trial_vel),
          d_trial_move_type(_d_trial_move_type), d_reject(_d_reject), block_size(_block_size)
        {
        }

    //! See hpmc_args_t for documentation on the meaning of these parameters
    Scalar4* d_postype;
    Scalar4* d_orientation;
    Scalar4* d_vel;
    hpmc_counters_t* d_counters;
    unsigned int counters_pitch;
    const GPUPartition& gpu_partition;
    const bool have_auxilliary_variable;
    const Scalar4* d_trial_postype;
    const Scalar4* d_trial_orientation;
    const Scalar4* d_trial_vel;
    const unsigned int* d_trial_move_type;
    const unsigned int* d_reject;
    const unsigned int block_size;
    };

//! Driver for kernel::hpmc_narrow_phase()
template<class Shape>
void hpmc_narrow_phase(const hpmc_args_t& args, const typename Shape::param_type* params);

//! Driver for kernel::hpmc_gen_moves()
template<class Shape>
void hpmc_gen_moves(const hpmc_args_t& args, const typename Shape::param_type* params);

//! Driver for kernel::hpmc_update_pdata()
template<class Shape>
void hpmc_update_pdata(const hpmc_update_args_t& args, const typename Shape::param_type* params);

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int* d_excell_idx,
                 unsigned int* d_excell_size,
                 const Index2D& excli,
                 const unsigned int* d_cell_idx,
                 const unsigned int* d_cell_size,
                 const unsigned int* d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size);

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4* d_postype,
                int3* d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size);

//! Kernel to evaluate convergence
void hpmc_check_convergence(const unsigned int* d_trial_move_type,
                            const unsigned int* d_reject_out_of_cell,
                            unsigned int* d_reject_in,
                            unsigned int* d_reject_out,
                            unsigned int* d_condition,
                            const GPUPartition& gpu_partition,
                            unsigned int block_size);

    } // end namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd
