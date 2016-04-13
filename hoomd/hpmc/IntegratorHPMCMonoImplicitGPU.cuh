#ifndef _HPMC_IMPLICIT_CUH_
#define _HPMC_IMPLICIT_CUH_

#include "HPMCCounters.h"


#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/extern/saruprngCUDA.h"

#include "hoomd/extern/util/mgpucontext.h"

#include <curand_kernel.h>

#include <cassert>

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicit.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_implicit_update
/*! \ingroup hpmc_data_structs */
struct hpmc_implicit_args_t
    {
    //! Construct a pair_args_t
    hpmc_implicit_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const Scalar4 *_d_postype_old,
                const Scalar4 *_d_orientation_old,
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
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const bool _has_orientation,
                const unsigned int _max_n,
                const cudaDeviceProp& _devprop,
                curandState_t *_d_state_cell,
                curandState_t *_d_state_cell_new,
                const unsigned int _depletant_type,
                hpmc_counters_t *_d_counters,
                hpmc_implicit_counters_t *_d_implicit_count,
                const curandDiscreteDistribution_t *_d_poisson,
                unsigned int *_d_overlap_cell,
                unsigned int *_d_overlap_cell_scan,
                unsigned int _groups_per_cell,
                const unsigned int *_d_active_cell_ptl_idx,
                const unsigned int *_d_active_cell_accept,
                const unsigned int *_d_active_cell_move_type_translate,
                float *_d_lnb,
                unsigned int *_d_n_success_zero,
                unsigned int _ntrial,
                unsigned int *_d_depletant_active_cell,
                unsigned int &_n_overlaps,
                unsigned int *_d_n_success_forward,
                unsigned int *_d_n_overlap_shape_forward,
                unsigned int *_d_n_success_reverse,
                unsigned int *_d_n_overlap_shape_reverse,
                float *_d_depletant_lnb,
                const Scalar *_d_d_min,
                const Scalar *_d_d_max,
                mgpu::ContextPtr _mgpu_context
                )
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_postype_old(_d_postype_old),
                  d_orientation_old(_d_orientation_old),
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
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  has_orientation(_has_orientation),
                  max_n(_max_n),
                  devprop(_devprop),
                  d_state_cell(_d_state_cell),
                  d_state_cell_new(_d_state_cell_new),
                  depletant_type(_depletant_type),
                  d_counters(_d_counters),
                  d_implicit_count(_d_implicit_count),
                  d_poisson(_d_poisson),
                  d_overlap_cell(_d_overlap_cell),
                  d_overlap_cell_scan(_d_overlap_cell_scan),
                  groups_per_cell(_groups_per_cell),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate),
                  d_lnb(_d_lnb),
                  d_n_success_zero(_d_n_success_zero),
                  ntrial(_ntrial),
                  d_depletant_active_cell(_d_depletant_active_cell),
                  n_overlaps(_n_overlaps),
                  d_n_success_forward(_d_n_success_forward),
                  d_n_overlap_shape_forward(_d_n_overlap_shape_forward),
                  d_n_success_reverse(_d_n_success_reverse),
                  d_n_overlap_shape_reverse(_d_n_overlap_shape_reverse),
                  d_depletant_lnb(_d_depletant_lnb),
                  d_d_min(_d_d_min),
                  d_d_max(_d_d_max),
                  mgpu_context(_mgpu_context)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const Scalar4 *d_postype_old;     //!< old postype array
    const Scalar4 *d_orientation_old; //!< old orientatino array
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
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    const unsigned int select;        //!< Current selection
    const unsigned int block_size;    //!< Block size to execute
    const unsigned int stride;        //!< Number of threads per overlap check
    const unsigned int group_size;    //!< Size of the group to execute
    const bool has_orientation;       //!< True if the shape has orientation
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    curandState_t *d_state_cell;        //!< RNG state per cell
    curandState_t *d_state_cell_new;    //!< RNG state per cell
    const unsigned int depletant_type; //!< Particle type of depletant
    hpmc_counters_t *d_counters;      //!< Aceptance/rejection counters
    hpmc_implicit_counters_t *d_implicit_count; //!< Active cell acceptance/rejection counts
    const curandDiscreteDistribution_t *d_poisson; //!< Handle for precomputed poisson distribution (per type)
    unsigned int *d_overlap_cell;     //!< Overlap flag per active cell
    unsigned int *d_overlap_cell_scan; //!< Overlap flag per active cell (scan result)
    unsigned int groups_per_cell;     //!< Number of groups to process in parallel per cell
    const unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell
    const unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise
    const unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    float *d_lnb;                     //!< Logarithm of Boltzmann factor
    unsigned int *d_n_success_zero;   //!< If the number of successful re-insertion attempts is zero (new position)
    unsigned int ntrial;              //!< Number of reinsertion attempts
    unsigned int *d_depletant_active_cell; //!< Lookup of active-cell idx per depletant
    unsigned int &n_overlaps;          //!< Total number of inserted overlapping depletants
    unsigned int *d_n_success_forward; //!< successful reinsertions in forward move, per depletant
    unsigned int *d_n_overlap_shape_forward; //!< reinsertions into old colloid positiion, per depletant
    unsigned int *d_n_success_reverse; //!< successful reinsertions in reverse move, per depletant
    unsigned int *d_n_overlap_shape_reverse; //!< reinsertions into new colloid position, per depletant
    float *d_depletant_lnb;            //!< logarithm of configurational bias weight, per depletant
    const Scalar *d_d_min;             //!< Minimum insertion diameter for depletants (per type)
    const Scalar *d_d_max;             //!< Maximum insertion diameter for depletants (per type)
    mgpu::ContextPtr mgpu_context;    //!< ModernGPU context
    };

template< class Shape >
void gpu_hpmc_implicit_count_overlaps(const hpmc_implicit_args_t &args, const typename Shape::param_type *d_params);

template< class Shape >
cudaError_t gpu_hpmc_implicit_accept_reject(const hpmc_implicit_args_t &args, const typename Shape::param_type *d_params);


}; // end namespace detail

} // end namespace hpmc

#endif // _HPMC_IMPLICIT_CUH_

