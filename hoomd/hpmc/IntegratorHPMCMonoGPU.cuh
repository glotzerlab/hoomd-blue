// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _INTEGRATOR_HPMC_CUH_
#define _INTEGRATOR_HPMC_CUH_


#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/extern/saruprngCUDA.h"

#include <cassert>

#include "HPMCCounters.h"

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoGPU.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_up
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a pair_args_t
    hpmc_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                hpmc_counters_t *_d_counters,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int *_d_cell_set,
                const unsigned int _n_active_cells,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int _move_ratio,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const bool _has_orientation,
                const unsigned int _max_n,
                const cudaDeviceProp& _devprop,
                unsigned int *_d_active_cell_ptl_idx = NULL,
                unsigned int *_d_active_cell_accept = NULL,
                unsigned int *_d_active_cell_move_type_translate = NULL)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_counters(_d_counters),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  ci(_ci),
                  cli(_cli),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  d_cell_set(_d_cell_set),
                  n_active_cells(_n_active_cells),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  d_d(_d),
                  d_a(_a),
                  move_ratio(_move_ratio),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  ghost_fraction(_ghost_fraction),
                  domain_decomposition(_domain_decomposition),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  has_orientation(_has_orientation),
                  max_n(_max_n),
                  devprop(_devprop),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    hpmc_counters_t *d_counters;      //!< Move accept/reject counters
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const unsigned int *d_excell_idx; //!< Index data for each expanded cell
    const unsigned int *d_excell_size;//!< Number of particles in each expanded cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const Index2D& excli;             //!< Indexer for d_excell_idx
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int *d_cell_set;   //!< List of active cells
    const unsigned int n_active_cells;//!< Number of active cells
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const Scalar* d_d;                //!< Maximum move displacement
    const Scalar* d_a;                //!< Maximum move angular displacement
    const unsigned int move_ratio;    //!< Ratio of translation to rotation moves
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    const unsigned int select;        //!< Current selection
    const Scalar3 ghost_fraction;     //!< Width of the inactive layer
    const bool domain_decomposition;  //!< Is domain decomposition mode enabled?
    const unsigned int block_size;    //!< Block size to execute
    const unsigned int stride;        //!< Number of threads per overlap check
    const unsigned int group_size;    //!< Size of the group to execute
    const bool has_orientation;       //!< True if the shape has orientation
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell (ignore if NULL)
    unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise (ignore if NULL)
    unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    };

cudaError_t gpu_hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size);

template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *d_params);

cudaError_t gpu_hpmc_shift(Scalar4 *d_postype,
                           int3 *d_image,
                           const unsigned int N,
                           const BoxDim& box,
                           const Scalar3 shift,
                           const unsigned int block_size);

}; // end namespace detail

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_CUH_

