#ifndef _COMPUTE_FREE_VOLUME_CUH_
#define _COMPUTE_FREE_VOLUME_CUH_

#include "HPMCCounters.h"


#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/extern/saruprngCUDA.h"

#include <curand_kernel.h>

#include <cassert>

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
    hpmc_free_volume_args_t(
                unsigned int _n_sample,
                unsigned int _type,
                Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                unsigned int _select,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const unsigned int _max_n,
                unsigned int *_d_n_overlap_all,
                const Scalar3 _ghost_width
                )
                : n_sample(_n_sample),
                  type(_type),
                  d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  ci(_ci),
                  cli(_cli),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  select(_select),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  max_n(_max_n),
                  d_n_overlap_all(_d_n_overlap_all),
                  ghost_width(_ghost_width)
        {
        };

    unsigned int n_sample;            //!< Number of depletants particles to generate
    unsigned int type;                //!< Type of depletant particle
    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const unsigned int *d_excell_idx; //!< Expanded cell neighbors
    const unsigned int *d_excell_size; //!< Size of expanded cell list per cell
    const Index2D excli;              //!< Expanded cell indexer
    const uint3& cell_dim;            //!< Cell dimensions
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    unsigned int select;              //!< RNG select value
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    unsigned int block_size;          //!< Block size to execute
    unsigned int stride;              //!< Number of threads per overlap check
    unsigned int group_size;          //!< Size of the group to execute
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    unsigned int *d_n_overlap_all;    //!< Total number of depletants in overlap volume
    const Scalar3 ghost_width;       //!< Width of ghost layer
    };

template< class Shape >
cudaError_t gpu_hpmc_free_volume(const hpmc_free_volume_args_t &args, const typename Shape::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc

#endif // _COMPUTE_FREE_VOLUME_CUH_

