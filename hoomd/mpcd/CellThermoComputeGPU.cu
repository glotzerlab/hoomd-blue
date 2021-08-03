// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoComputeGPU.cu
 * \brief Explicitly instantiates reduction operators and declares kernel drivers
 *        for mpcd::CellThermoComputeGPU.
 */

#include "CellThermoComputeGPU.cuh"
#include "CellThermoTypes.h"

#include "CellCommunicator.cuh"
#include "ReductionOperators.h"

#include "hoomd/WarpTools.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Begins the cell thermo compute by summing cell quantities on outer cells
/*!
 * \param d_cell_vel Velocity and mass per cell (output)
 * \param d_cell_energy Energy, temperature, number of particles per cell (output)
 * \param d_cells Cell indexes to compute
 * \param d_cell_np Number of particles per cell
 * \param d_cell_list MPCD cell list
 * \param cli Indexer into the cell list
 * \param d_vel MPCD particle velocities
 * \param N_mpcd Number of MPCD particles
 * \param mpcd_mass Mass of MPCD particle
 * \param d_embed_vel Embedded particle velocity
 * \param d_embed_idx Embedded particle indexes
 * \param num_cells Number of cells to compute for
 *
 * \tparam need_energy If true, compute the cell-level energy properties
 * \tparam tpp Number of threads to use per cell
 *
 * \b Implementation details:
 * Using \a tpp threads per cell, the cell properties are accumulated into \a d_cell_vel
 * and \a d_cell_energy. Shuffle-based intrinsics are used to reduce the accumulated
 * properties per-cell, and the first thread for each cell writes the result into
 * global memory.
 */
template<bool need_energy, unsigned int tpp>
__global__ void begin_cell_thermo(double4 *d_cell_vel,
                                  double3 *d_cell_energy,
                                  const unsigned int *d_cells,
                                  const unsigned int *d_cell_np,
                                  const unsigned int *d_cell_list,
                                  const Index2D cli,
                                  const Scalar4 *d_vel,
                                  const unsigned int N_mpcd,
                                  const Scalar mpcd_mass,
                                  const Scalar4 *d_embed_vel,
                                  const unsigned int *d_embed_idx,
                                  const unsigned int num_cells)
    {
    // tpp threads per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tpp * num_cells)
        return;

    const unsigned int cell_id = d_cells[idx / tpp];
    const unsigned int np = d_cell_np[cell_id];
    double4 momentum = make_double4(0.0, 0.0, 0.0, 0.0);
    double ke(0.0);

    for (unsigned int offset = (idx % tpp); offset < np; offset += tpp)
        {
        // Load particle data
        const unsigned int cur_p = d_cell_list[cli(offset, cell_id)];
        double3 vel_i;
        double mass_i;
        if (cur_p < N_mpcd)
            {
            Scalar4 vel_cell = d_vel[cur_p];
            vel_i = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
            mass_i = mpcd_mass;
            }
        else
            {
            Scalar4 vel_m = d_embed_vel[d_embed_idx[cur_p - N_mpcd]];
            vel_i = make_double3(vel_m.x, vel_m.y, vel_m.z);
            mass_i = vel_m.w;
            }

        // add momentum
        momentum.x += mass_i * vel_i.x;
        momentum.y += mass_i * vel_i.y;
        momentum.z += mass_i * vel_i.z;
        momentum.w += mass_i;

        // also compute ke of the particle
        if (need_energy)
            ke += (double)(0.5) * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
        }

    // reduce quantities down into the 0-th lane per logical warp
    if (tpp > 1)
        {
        hoomd::detail::WarpReduce<Scalar, tpp> reducer;
        momentum.x = reducer.Sum(momentum.x);
        momentum.y = reducer.Sum(momentum.y);
        momentum.z = reducer.Sum(momentum.z);
        momentum.w = reducer.Sum(momentum.w);
        if (need_energy)
            ke = reducer.Sum(ke);
        }

    // 0-th lane in each warp writes the result
    if (idx % tpp == 0)
        {
        d_cell_vel[cell_id] = make_double4(momentum.x, momentum.y, momentum.z, momentum.w);
        if (need_energy)
            d_cell_energy[cell_id] = make_double3(ke, 0.0, __int_as_double(np));
        }
    }

//! Finalizes the cell thermo compute by properly averaging cell quantities
/*!
 * \param d_cell_vel Cell velocity and masses
 * \param d_cell_energy Cell energy and temperature
 * \param d_cells Cells to compute for
 * \param Ncell Number of cells
 * \param n_dimensions Number of dimensions in system
 *
 * \tparam need_energy If true, compute the cell-level energy properties.
 *
 * \b Implementation details:
 * Using one thread per cell, the properties are averaged by mass, number of particles,
 * etc. The temperature is computed from the cell kinetic energy.
 */
template<bool need_energy>
__global__ void end_cell_thermo(double4 *d_cell_vel,
                                double3 *d_cell_energy,
                                const unsigned int *d_cells,
                                const unsigned int Ncell,
                                const unsigned int n_dimensions)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncell)
        return;

    const unsigned int cell_id = d_cells[idx];

    // average cell properties if the cell has mass
    const double4 cell_vel = d_cell_vel[cell_id];
    double3 vel_cm = make_double3(cell_vel.x, cell_vel.y, cell_vel.z);
    const double mass = cell_vel.w;

    if (mass > 0.)
        {
        // average velocity is only defined when there is some mass in the cell
        vel_cm.x /= mass; vel_cm.y /= mass; vel_cm.z /= mass;
        }
    d_cell_vel[cell_id] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

    if (need_energy)
        {
        const double3 cell_energy = d_cell_energy[cell_id];
        const double ke = cell_energy.x;
        double temp(0.0);
        const unsigned int np = __double_as_int(cell_energy.z);
        // temperature is only defined for 2 or more particles
        if (np > 1)
            {
            const double ke_cm = 0.5 * mass * (vel_cm.x*vel_cm.x + vel_cm.y*vel_cm.y + vel_cm.z*vel_cm.z);
            temp = 2. * (ke - ke_cm) / (n_dimensions * (np-1));
            }
        d_cell_energy[cell_id] = make_double3(ke, temp, __int_as_double(np));
        }
    }

//! Computes the cell thermo for inner cells
/*!
 * \param d_cell_vel Velocity and mass per cell (output)
 * \param d_cell_energy Energy, temperature, number of particles per cell (output)
 * \param ci Cell indexer
 * \param inner_ci Cell indexer for the inner cells
 * \param offset Offset of \a inner_ci from \a ci
 * \param d_cell_np Number of particles per cell
 * \param d_cell_list MPCD cell list
 * \param cli Indexer into the cell list
 * \param d_vel MPCD particle velocities
 * \param N_mpcd Number of MPCD particles
 * \param mpcd_mass Mass of MPCD particle
 * \param d_embed_vel Embedded particle velocity
 * \param d_embed_idx Embedded particle indexes
 * \param n_dimensions System dimensionality
 *
 * \tparam need_energy If true, compute the cell-level energy properties.
 * \tparam tpp Number of threads to use per cell
 *
 * \b Implementation details:
 * Using \a tpp threads per cell, the cell properties are accumulated into \a d_cell_vel
 * and \a d_cell_energy. Shuffle-based intrinsics are used to reduce the accumulated
 * properties per-cell, and the first thread for each cell writes the result into
 * global memory. The properties are properly normalized
 *
 * See mpcd::gpu::kernel::begin_cell_thermo for an almost identical implementation
 * without the normalization at the end, which is used for the outer cells.
 */
template<bool need_energy, unsigned int tpp>
__global__ void inner_cell_thermo(double4 *d_cell_vel,
                                  double3 *d_cell_energy,
                                  const Index3D ci,
                                  const Index3D inner_ci,
                                  const uint3 offset,
                                  const unsigned int *d_cell_np,
                                  const unsigned int *d_cell_list,
                                  const Index2D cli,
                                  const Scalar4 *d_vel,
                                  const unsigned int N_mpcd,
                                  const Scalar mpcd_mass,
                                  const Scalar4 *d_embed_vel,
                                  const unsigned int *d_embed_idx,
                                  const unsigned int n_dimensions)
    {
    // tpp threads per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tpp * inner_ci.getNumElements())
        return;

    // reinterpret the thread id as a cell by first mapping the thread into the inner indexer,
    // shifting by the offset of the inner indexer from the full indexer, and then compressing
    // back into a 1D cell id
    const uint3 inner_cell = inner_ci.getTriple(idx/tpp);
    const uint3 cell = make_uint3(inner_cell.x + offset.x, inner_cell.y + offset.y, inner_cell.z + offset.z);
    const unsigned int cell_id = ci(cell.x, cell.y, cell.z);

    const unsigned int np = d_cell_np[cell_id];
    double4 momentum = make_double4(0.0, 0.0, 0.0, 0.0);
    double ke(0.0);

    for (unsigned int offset = (idx % tpp); offset < np; offset += tpp)
        {
        // Load particle data
        const unsigned int cur_p = d_cell_list[cli(offset, cell_id)];
        double3 vel_i;
        double mass_i;
        if (cur_p < N_mpcd)
            {
            Scalar4 vel_cell = d_vel[cur_p];
            vel_i = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
            mass_i = mpcd_mass;
            }
        else
            {
            Scalar4 vel_m = d_embed_vel[d_embed_idx[cur_p - N_mpcd]];
            vel_i = make_double3(vel_m.x, vel_m.y, vel_m.z);
            mass_i = vel_m.w;
            }

        // add momentum
        momentum.x += mass_i * vel_i.x;
        momentum.y += mass_i * vel_i.y;
        momentum.z += mass_i * vel_i.z;
        momentum.w += mass_i;

        // also compute ke of the particle
        if (need_energy)
            ke += 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
        }

    // reduce quantities down into the 0-th lane per logical warp
    if (tpp > 1)
        {
        hoomd::detail::WarpReduce<Scalar, tpp> reducer;
        momentum.x = reducer.Sum(momentum.x);
        momentum.y = reducer.Sum(momentum.y);
        momentum.z = reducer.Sum(momentum.z);
        momentum.w = reducer.Sum(momentum.w);
        if (need_energy)
            ke = reducer.Sum(ke);
        }

    // 0-th lane in each warp writes the result
    if (idx % tpp == 0)
        {
        const double mass = momentum.w;
        double3 vel_cm = make_double3(0.0,0.0,0.0);
        if (mass > 0.)
            {
            vel_cm.x = momentum.x / mass;
            vel_cm.y = momentum.y / mass;
            vel_cm.z = momentum.z / mass;
            }
        d_cell_vel[cell_id] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

        if (need_energy)
            {
            double temp(0.0);
            if (np > 1)
                {
                const double ke_cm = 0.5 * mass * (vel_cm.x*vel_cm.x + vel_cm.y*vel_cm.y + vel_cm.z*vel_cm.z);
                temp = 2. * (ke - ke_cm) / (n_dimensions * (np-1));
                }
            d_cell_energy[cell_id] = make_double3(ke, temp, __int_as_double(np));
            }
        }
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_vel Cell velocity to reduce
 * \param d_cell_energy Cell energy to reduce
 * \param tmp_ci Temporary cell indexer for cells undergoing reduction
 * \param ci Cell indexer Regular cell list indexer
 *
 * \tparam need_energy If true, compute the cell-level energy properties.
 *
 * \b Implementation details:
 * Using one thread per \a temporary cell, the cell properties are normalized
 * in a way suitable for reduction of net properties, e.g. the cell velocities
 * are converted to momentum. The temperature is set to the cell energy, and a
 * flag is set to 1 or 0 to indicate whether this cell has an energy that should
 * be used in averaging the total temperature.
 */
template<bool need_energy>
__global__ void stage_net_cell_thermo(mpcd::detail::cell_thermo_element *d_tmp_thermo,
                                      const double4 *d_cell_vel,
                                      const double3 *d_cell_energy,
                                      const Index3D tmp_ci,
                                      const Index3D ci)
    {
    // one thread per cell
    unsigned int tmp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_idx >= tmp_ci.getNumElements())
        return;

    // use the temporary cell indexer to map to a cell, then use the real cell indexer to
    // get the read index
    uint3 cell = tmp_ci.getTriple(tmp_idx);
    const unsigned int idx = ci(cell.x, cell.y, cell.z);

    const double4 vel_mass = d_cell_vel[idx];
    const double3 vel = make_double3(vel_mass.x, vel_mass.y, vel_mass.z);
    const double mass = vel_mass.w;

    mpcd::detail::cell_thermo_element thermo;
    thermo.momentum = make_double3(mass * vel.x,
                                   mass * vel.y,
                                   mass * vel.z);

    if (need_energy)
        {
        const double3 cell_energy = d_cell_energy[idx];
        thermo.energy = cell_energy.x;
        if (__double_as_int(cell_energy.z) > 1)
            {
            thermo.temperature = cell_energy.y;
            thermo.flag = 1;
            }
        else
            {
            thermo.temperature = 0.0;
            thermo.flag = 0;
            }
        }
    else
        {
        thermo.energy = 0.; thermo.temperature = 0.; thermo.flag = 0;
        }

    d_tmp_thermo[tmp_idx] = thermo;
    }

} // end namespace kernel

//! Templated launcher for multiple threads-per-cell kernel for outer cells
/*
 * \param args Common arguments to thermo kernels
 * \param d_cells Cell indexes to compute
 * \param num_cells Number of cells to compute for
 * \param block_size Number of threads per block
 * \param tpp Number of threads to use per-cell
 *
 * \tparam cur_tpp Number of threads-per-cell for this template instantiation
 *
 * Launchers are recursively instantiated at compile-time in order to match the
 * correct number of threads at runtime. If the templated number of threads matches
 * the runtime number of threads, then the kernel is launched. Otherwise, the
 * next template (with threads reduced by a factor of 2) is launched. This
 * recursion is broken by a specialized template for 0 threads, which does no
 * work.
 */
template<unsigned int cur_tpp>
inline void launch_begin_cell_thermo(const mpcd::detail::thermo_args_t& args,
                                     const unsigned int *d_cells,
                                     const unsigned int num_cells,
                                     const unsigned int block_size,
                                     const unsigned int tpp)
    {
    if (cur_tpp == tpp)
        {
        if (args.need_energy)
            {
            static unsigned int max_block_size_energy = UINT_MAX;
            if (max_block_size_energy == UINT_MAX)
                {
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::begin_cell_thermo<true,cur_tpp>);
                max_block_size_energy = attr.maxThreadsPerBlock;
                }

            unsigned int run_block_size = min(block_size, max_block_size_energy);
            dim3 grid(cur_tpp*num_cells / run_block_size + 1);
            mpcd::gpu::kernel::begin_cell_thermo<true,cur_tpp><<<grid, run_block_size>>>(args.cell_vel,
                                                                                         args.cell_energy,
                                                                                         d_cells,
                                                                                         args.cell_np,
                                                                                         args.cell_list,
                                                                                         args.cli,
                                                                                         args.vel,
                                                                                         args.N_mpcd,
                                                                                         args.mass,
                                                                                         args.embed_vel,
                                                                                         args.embed_idx,
                                                                                         num_cells);
            }
        else
            {
            static unsigned int max_block_size_noenergy = UINT_MAX;
            if (max_block_size_noenergy == UINT_MAX)
                {
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::begin_cell_thermo<false,cur_tpp>);
                max_block_size_noenergy = attr.maxThreadsPerBlock;
                }

            unsigned int run_block_size = min(block_size, max_block_size_noenergy);
            dim3 grid(cur_tpp*num_cells / run_block_size + 1);
            mpcd::gpu::kernel::begin_cell_thermo<false,cur_tpp><<<grid, run_block_size>>>(args.cell_vel,
                                                                                          args.cell_energy,
                                                                                          d_cells,
                                                                                          args.cell_np,
                                                                                          args.cell_list,
                                                                                          args.cli,
                                                                                          args.vel,
                                                                                          args.N_mpcd,
                                                                                          args.mass,
                                                                                          args.embed_vel,
                                                                                          args.embed_idx,
                                                                                          num_cells);
            }
        }
    else
        {
        launch_begin_cell_thermo<cur_tpp/2>(args,
                                            d_cells,
                                            num_cells,
                                            block_size,
                                            tpp);
        }
    }
//! Template specialization to break recursion
template<>
inline void launch_begin_cell_thermo<0>(const mpcd::detail::thermo_args_t& args,
                                        const unsigned int *d_cells,
                                        const unsigned int num_cells,
                                        const unsigned int block_size,
                                        const unsigned int tpp)
    { }

/*
 * \param args Common arguments to thermo kernels
 * \param d_cells Cell indexes to compute
 * \param num_cells Number of cells to compute for
 * \param block_size Number of threads per block
 * \param tpp Number of threads per cell
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::launch_begin_cell_thermo
 * \sa mpcd::gpu::kernel::begin_cell_thermo
 */
cudaError_t begin_cell_thermo(const mpcd::detail::thermo_args_t& args,
                              const unsigned int *d_cells,
                              const unsigned int num_cells,
                              const unsigned int block_size,
                              const unsigned int tpp)
    {
    if (num_cells == 0) return cudaSuccess;

    launch_begin_cell_thermo<32>(args,
                                 d_cells,
                                 num_cells,
                                 block_size,
                                 tpp);
    return cudaSuccess;
    }

/*!
 * \param d_cell_vel Cell velocity and masses
 * \param d_cell_energy Cell energy and temperature
 * \param d_cells Cells to compute for
 * \param Ncell Number of cells
 * \param n_dimensions Number of dimensions in system
 * \param need_energy If true, compute the cell-level energy properties
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::end_cell_thermo
 */
cudaError_t end_cell_thermo(double4 *d_cell_vel,
                            double3 *d_cell_energy,
                            const unsigned int *d_cells,
                            const unsigned int Ncell,
                            const unsigned int n_dimensions,
                            const bool need_energy,
                            const unsigned int block_size)
    {
    if (Ncell == 0) return cudaSuccess;

    if (need_energy)
        {
        static unsigned int max_block_size_energy = UINT_MAX;
        if (max_block_size_energy == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::end_cell_thermo<true>);
            max_block_size_energy = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size_energy);
        dim3 grid(Ncell / run_block_size + 1);
        mpcd::gpu::kernel::end_cell_thermo<true><<<grid, run_block_size>>>(d_cell_vel,
                                                                           d_cell_energy,
                                                                           d_cells,
                                                                           Ncell,
                                                                           n_dimensions);
        }
    else
        {
        static unsigned int max_block_size_noenergy = UINT_MAX;
        if (max_block_size_noenergy == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::end_cell_thermo<true>);
            max_block_size_noenergy = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size_noenergy);
        dim3 grid(Ncell / run_block_size + 1);
        mpcd::gpu::kernel::end_cell_thermo<false><<<grid, run_block_size>>>(d_cell_vel,
                                                                            d_cell_energy,
                                                                            d_cells,
                                                                            Ncell,
                                                                            n_dimensions);
        }

    return cudaSuccess;
    }

//! Templated launcher for multiple threads-per-cell kernel for inner cells
/*
 * \param args Common arguments to thermo kernels
 * \param ci Cell indexer
 * \param inner_ci Cell indexer for the inner cells
 * \param offset Offset of \a inner_ci from \a ci
 * \param n_dimensions System dimensionality
 * \param block_size Number of threads per block
 * \param tpp Number of threads per cell
 *
 * \tparam cur_tpp Number of threads-per-cell for this template instantiation
 *
 * Launchers are recursively instantiated at compile-time in order to match the
 * correct number of threads at runtime. If the templated number of threads matches
 * the runtime number of threads, then the kernel is launched. Otherwise, the
 * next template (with threads reduced by a factor of 2) is launched. This
 * recursion is broken by a specialized template for 0 threads, which does no
 * work.
 */
template<unsigned int cur_tpp>
inline void launch_inner_cell_thermo(const mpcd::detail::thermo_args_t& args,
                                     const Index3D& ci,
                                     const Index3D& inner_ci,
                                     const uint3& offset,
                                     const unsigned int n_dimensions,
                                     const unsigned int block_size,
                                     const unsigned int tpp)
    {
    if (cur_tpp == tpp)
        {
        if (args.need_energy)
            {
            static unsigned int max_block_size_energy = UINT_MAX;
            if (max_block_size_energy == UINT_MAX)
                {
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::inner_cell_thermo<true,cur_tpp>);
                max_block_size_energy = attr.maxThreadsPerBlock;
                }

            unsigned int run_block_size = min(block_size, max_block_size_energy);
            dim3 grid(cur_tpp*ci.getNumElements() / run_block_size + 1);
            mpcd::gpu::kernel::inner_cell_thermo<true,cur_tpp><<<grid, run_block_size>>>(args.cell_vel,
                                                                                         args.cell_energy,
                                                                                         ci,
                                                                                         inner_ci,
                                                                                         offset,
                                                                                         args.cell_np,
                                                                                         args.cell_list,
                                                                                         args.cli,
                                                                                         args.vel,
                                                                                         args.N_mpcd,
                                                                                         args.mass,
                                                                                         args.embed_vel,
                                                                                         args.embed_idx,
                                                                                         n_dimensions);
            }
        else
            {
            static unsigned int max_block_size_noenergy = UINT_MAX;
            if (max_block_size_noenergy == UINT_MAX)
                {
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::inner_cell_thermo<false,cur_tpp>);
                max_block_size_noenergy = attr.maxThreadsPerBlock;
                }

            unsigned int run_block_size = min(block_size, max_block_size_noenergy);
            dim3 grid(cur_tpp*ci.getNumElements() / run_block_size + 1);
            mpcd::gpu::kernel::inner_cell_thermo<false,cur_tpp><<<grid, run_block_size>>>(args.cell_vel,
                                                                                          args.cell_energy,
                                                                                          ci,
                                                                                          inner_ci,
                                                                                          offset,
                                                                                          args.cell_np,
                                                                                          args.cell_list,
                                                                                          args.cli,
                                                                                          args.vel,
                                                                                          args.N_mpcd,
                                                                                          args.mass,
                                                                                          args.embed_vel,
                                                                                          args.embed_idx,
                                                                                          n_dimensions);
            }
        }
    else
        {
        launch_inner_cell_thermo<cur_tpp/2>(args,
                                            ci,
                                            inner_ci,
                                            offset,
                                            n_dimensions,
                                            block_size,
                                            tpp);
        }
    }
//! Template specialization to break recursion
template<>
inline void launch_inner_cell_thermo<0>(const mpcd::detail::thermo_args_t& args,
                                        const Index3D& ci,
                                        const Index3D& inner_ci,
                                        const uint3& offset,
                                        const unsigned int n_dimensions,
                                        const unsigned int block_size,
                                        const unsigned int tpp)
    { }

/*!
 * \param args Common arguments for cell thermo compute
 * \param ci Cell indexer
 * \param inner_ci Cell indexer for the inner cells
 * \param offset Offset of \a inner_ci from \a ci
 * \param n_dimensions System dimensionality
 * \param block_size Number of threads per block
 * \param tpp Number of threads per cell
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::launch_inner_cell_thermo
 * \sa mpcd::gpu::kernel::inner_cell_thermo
 */
cudaError_t inner_cell_thermo(const mpcd::detail::thermo_args_t& args,
                              const Index3D& ci,
                              const Index3D& inner_ci,
                              const uint3& offset,
                              const unsigned int n_dimensions,
                              const unsigned int block_size,
                              const unsigned int tpp)
    {
    if (inner_ci.getNumElements() == 0) return cudaSuccess;

    launch_inner_cell_thermo<32>(args,
                                 ci,
                                 inner_ci,
                                 offset,
                                 n_dimensions,
                                 block_size,
                                 tpp);

    return cudaSuccess;
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_vel Cell velocity to reduce
 * \param d_cell_energy Cell energy to reduce
 * \param tmp_ci Temporary cell indexer for cells undergoing reduction
 * \param ci Cell indexer Regular cell list indexer
 * \param need_energy If true, compute the cell-level energy properties
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::stage_net_cell_thermo
 */
cudaError_t stage_net_cell_thermo(mpcd::detail::cell_thermo_element *d_tmp_thermo,
                                  const double4 *d_cell_vel,
                                  const double3 *d_cell_energy,
                                  const Index3D& tmp_ci,
                                  const Index3D& ci,
                                  bool need_energy,
                                  const unsigned int block_size)
    {
    if (need_energy)
        {
        static unsigned int max_block_size_energy = UINT_MAX;
        if (max_block_size_energy == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<true>);
            max_block_size_energy = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size_energy);
        dim3 grid(tmp_ci.getNumElements() / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<true><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                 d_cell_vel,
                                                                                 d_cell_energy,
                                                                                 tmp_ci,
                                                                                 ci);
        }
    else
        {
        static unsigned int max_block_size_noenergy = UINT_MAX;
        if (max_block_size_noenergy == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<false>);
            max_block_size_noenergy = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size_noenergy);
        dim3 grid(tmp_ci.getNumElements() / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<false><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                  d_cell_vel,
                                                                                  d_cell_energy,
                                                                                  tmp_ci,
                                                                                  ci);
        }
    return cudaSuccess;
    }

/*!
 * \param d_reduced Cell thermo properties reduced across all cells (output on second call)
 * \param d_tmp Temporary storage for reduction (output on first call)
 * \param tmp_bytes Number of bytes allocated for temporary storage (output on first call)
 * \param d_tmp_thermo Cell thermo properties to reduce
 * \param Ncell The number of cells to reduce across
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation details:
 * CUB DeviceReduce is used to perform the reduction. Hence, this function requires
 * two calls to perform the reduction. The first call sizes the temporary storage,
 * which is returned in \a d_tmp and \a tmp_bytes. The caller must then allocate
 * the required bytes, and call the function a second time. This performs the
 * reduction and returns the result in \a d_reduced.
 */
cudaError_t reduce_net_cell_thermo(mpcd::detail::cell_thermo_element *d_reduced,
                                   void *d_tmp,
                                   size_t& tmp_bytes,
                                   const mpcd::detail::cell_thermo_element *d_tmp_thermo,
                                   const unsigned int Ncell)
    {
    HOOMD_CUB::DeviceReduce::Sum(d_tmp, tmp_bytes, d_tmp_thermo, d_reduced, Ncell);
    return cudaSuccess;
    }

//! Explicit template instantiation of pack for cell velocity
template cudaError_t pack_cell_buffer(typename mpcd::detail::CellVelocityPackOp::element *d_send_buf,
                                      const double4 *d_props,
                                      const unsigned int *d_send_idx,
                                      const mpcd::detail::CellVelocityPackOp op,
                                      const unsigned int num_send,
                                      unsigned int block_size);

//! Explicit template instantiation of pack for cell energy
template cudaError_t pack_cell_buffer(typename mpcd::detail::CellEnergyPackOp::element *d_send_buf,
                                      const double3 *d_props,
                                      const unsigned int *d_send_idx,
                                      const mpcd::detail::CellEnergyPackOp op,
                                      const unsigned int num_send,
                                      unsigned int block_size);

//! Explicit template instantiation of unpack for cell velocity
template cudaError_t unpack_cell_buffer(double4 *d_props,
                                        const unsigned int *d_cells,
                                        const unsigned int *d_recv,
                                        const unsigned int *d_recv_begin,
                                        const unsigned int *d_recv_end,
                                        const typename mpcd::detail::CellVelocityPackOp::element *d_recv_buf,
                                        const mpcd::detail::CellVelocityPackOp op,
                                        const unsigned int num_cells,
                                        const unsigned int block_size);

//! Explicit template instantiation of unpack for cell energy
template cudaError_t unpack_cell_buffer(double3 *d_props,
                                        const unsigned int *d_cells,
                                        const unsigned int *d_recv,
                                        const unsigned int *d_recv_begin,
                                        const unsigned int *d_recv_end,
                                        const typename mpcd::detail::CellEnergyPackOp::element *d_recv_buf,
                                        const mpcd::detail::CellEnergyPackOp op,
                                        const unsigned int num_cells,
                                        const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd
