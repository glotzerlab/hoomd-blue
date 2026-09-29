// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::CellListGPU
 */

#include <cub/device/device_reduce.cuh>
#ifdef ENABLE_MPI
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_iterator.h>
#endif // ENABLE_MPI

#include "CellListGPU.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {

#ifdef ENABLE_MPI
//! Functor to test if a direction is valid (i.e., less than 26)
struct ValidDirection
    {
    __host__ __device__ inline unsigned int operator()(unsigned int dir) const
        {
        return dir < 26;
        }
    };
#endif // ENABLE_MPI

namespace kernel
    {
//! Kernel to compute the MPCD cell list on the GPU
/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_vel_embed Particle velocities
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param global_cell_indexer 3D indexer for global cell id
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param rank_size the size of the local rank
 * \param is_decomposition whether there is domain decomposition
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 * \param need_energy True if computing the energy
 *
 * \b Implementation
 * One thread is launched per particle. The particle is floored into a bin subject to a random grid
 * shift. The number of particles in that bin is atomically incremented and the contribution of the
 * particle's properties to the cell's properties is added to a running sum. The MPCD particle's
 * cell id is stashed into the velocity array.
 */
__global__ void compute_cell_list(unsigned int* d_cell_np,
                                  double4* d_cell_vel,
                                  double* d_cell_energy,
                                  uint3* d_conditions,
                                  Scalar4* d_vel,
                                  double mpcd_mass,
                                  unsigned int* d_embed_cell_ids,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_pos_embed,
                                  const Scalar4* d_vel_embed,
                                  const unsigned int* d_embed_member_idx,
                                  const uchar3 periodic,
                                  const uint3 origin_idx,
                                  const Scalar3 grid_shift,
                                  const BoxDim global_box,
                                  const uint3 global_cell_dim,
                                  const Index3D cell_indexer,
                                  const Index3D global_cell_indexer,
                                  unsigned int* d_ghost_dir,
                                  const uint3 rank_size,
                                  const bool is_decomposition,
                                  const unsigned int N_mpcd,
                                  const unsigned int N_tot,
                                  const bool need_energy)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    Scalar4 postype_i;
    Scalar4 vel_mass_i;
    double mass_i;
    if (idx < N_mpcd)
        {
        postype_i = d_pos[idx];
        vel_mass_i = d_vel[idx];
        mass_i = mpcd_mass;
        }
    else
        {
        postype_i = d_pos_embed[d_embed_member_idx[idx - N_mpcd]];
        vel_mass_i = d_vel_embed[d_embed_member_idx[idx - N_mpcd]];
        mass_i = vel_mass_i.w;
        }
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const double3 vel_i = make_double3(vel_mass_i.x, vel_mass_i.y, vel_mass_i.z);

    if (isnan(pos_i.x) || isnan(pos_i.y) || isnan(pos_i.z))
        {
        (*d_conditions).y = idx + 1;
        return;
        }

    // bin particle with grid shift
    const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i) - grid_shift;
    int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * global_cell_dim.x),
                                (int)std::floor(fractional_pos_i.y * global_cell_dim.y),
                                (int)std::floor(fractional_pos_i.z * global_cell_dim.z));

    // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
    // this is done using periodic from the global box
    if (periodic.x)
        {
        if (global_bin.x >= (int)global_cell_dim.x)
            global_bin.x -= global_cell_dim.x;
        else if (global_bin.x < 0)
            global_bin.x += global_cell_dim.x;
        }
    if (periodic.y)
        {
        if (global_bin.y >= (int)global_cell_dim.y)
            global_bin.y -= global_cell_dim.y;
        else if (global_bin.y < 0)
            global_bin.y += global_cell_dim.y;
        }
    if (periodic.z)
        {
        if (global_bin.z >= (int)global_cell_dim.z)
            global_bin.z -= global_cell_dim.z;
        else if (global_bin.z < 0)
            global_bin.z += global_cell_dim.z;
        }

    // validate and make sure no particles blew out of the global box
    if ((global_bin.x < 0 || global_bin.x >= (int)global_cell_dim.x)
        || (global_bin.y < 0 || global_bin.y >= (int)global_cell_dim.y)
        || (global_bin.z < 0 || global_bin.z >= (int)global_cell_dim.z))
        {
        (*d_conditions).z = idx + 1;
        return;
        }

    // compute the local cell
    int3 bin = make_int3(global_bin.x - origin_idx.x,
                         global_bin.y - origin_idx.y,
                         global_bin.z - origin_idx.z);

    // check if the particle is still in the local box
    const bool is_local = ((bin.x >= 0 && bin.x < (int)cell_indexer.getW())
                           && (bin.y >= 0 && bin.y < (int)cell_indexer.getH())
                           && (bin.z >= 0 && bin.z < (int)cell_indexer.getD()));

    // check if particles outside the local box is allowed
    if (!is_local && !is_decomposition)
        {
        (*d_conditions).z = idx + 1;
        return;
        }

    unsigned int bin_idx;
    if (is_local)
        {
        bin_idx = cell_indexer(bin.x, bin.y, bin.z);
        }
#ifdef ENABLE_MPI
    else
        {
        if (is_decomposition && idx < N_mpcd)
            {
            // determine from the bin which rank the particle's cell belongs to
            int ix = 0;
            if (bin.x >= (int)cell_indexer.getW())
                ix = (rank_size.x > 2) ? 1 : -1;
            else if (bin.x < 0)
                ix = -1;

            int iy = 0;
            if (bin.y >= (int)cell_indexer.getH())
                iy = (rank_size.y > 2) ? 1 : -1;
            else if (bin.y < 0)
                iy = -1;

            int iz = 0;
            if (bin.z >= (int)cell_indexer.getD())
                iz = (rank_size.z > 2) ? 1 : -1;
            else if (bin.z < 0)
                iz = -1;

            // get shifted direction index
            int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
            dir = dir + ((ix == 1) ? -2 : 1) + ((iy == 1) ? -6 : 3) + ((iz == 1) ? -12 : 9);
            // mark particle to be sent to neighboring rank
            d_ghost_dir[idx] = dir;

            // set the bin idx to be the global index with highest bit set to 1
            bin_idx = global_cell_indexer(global_bin.x, global_bin.y, global_bin.z);
            bin_idx |= 1u << 31;
            }
        }
#endif // ENABLE_MPI

    // stash the current particle bin into the velocity array
    if (idx < N_mpcd)
        {
        d_vel[idx].w = __int_as_scalar(bin_idx);
        }
    else
        {
        d_embed_cell_ids[idx - N_mpcd] = bin_idx;
        }

    if (!is_local)
        {
        return;
        }

    // compute the contribution of the particle to cell properties
    const unsigned int offset = atomicInc(&d_cell_np[bin_idx], 0xffffffff);
    double4& cell_vel = d_cell_vel[bin_idx];
    atomicAdd(&cell_vel.x, vel_i.x * mass_i);
    atomicAdd(&cell_vel.y, vel_i.y * mass_i);
    atomicAdd(&cell_vel.z, vel_i.z * mass_i);
    atomicAdd(&cell_vel.w, mass_i);

    if (need_energy)
        {
        double ke = 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
        atomicAdd(&d_cell_energy[bin_idx], ke);
        }
    }

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Cell velocity to finish computing
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature to finish computing
 * \param N_cells Number of cells
 * \param N_dim Number of dimensions
 * \param need_energy If true, compute the cell-level energy properties.
 *
 * \b Implementation details:
 * Using one thread per cell, the cell properties are normalized from the
 * additive contributions of the particles to their final values, e.g. the cell
 * velocities are computed from the net momentum of the cell divided by the final
 * mass. The temperature of each cell is calculated.
 */
__global__ void finish_cell_properties(const unsigned int* d_cell_np,
                                       double4* d_cell_vel,
                                       const double* d_cell_energy,
                                       double* d_cell_temp,
                                       const unsigned int N_cells,
                                       const unsigned int N_dim,
                                       const bool need_energy)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cells)
        return;

    const double4 vel_cell = d_cell_vel[idx];
    double3 vel_cm = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
    const double mass = vel_cell.w;

    // get center of mass velocity from momentum
    if (mass > 0.)
        {
        // average velocity is only defined when there is some mass in the cell
        vel_cm.x /= mass;
        vel_cm.y /= mass;
        vel_cm.z /= mass;
        }
    d_cell_vel[idx] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

    if (need_energy)
        {
        const double cell_energy = d_cell_energy[idx];
        double temp(0.0);
        const unsigned int np = d_cell_np[idx];

        if (np > 1)
            {
            const double ke_cm
                = 0.5 * mass * (vel_cm.x * vel_cm.x + vel_cm.y * vel_cm.y + vel_cm.z * vel_cm.z);
            temp = 2. * (cell_energy - ke_cm) / (N_dim * (np - 1));
            }
        d_cell_temp[idx] = temp;
        }
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_np Number of particles per cell
 * \param d_cell_vel Cell velocity
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temperature Cell temperature
 * \param N_cells Number of cells
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
__global__ void stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                      const unsigned int* d_cell_np,
                                      const double4* d_cell_vel,
                                      const double* d_cell_energy,
                                      const double* d_cell_temp,
                                      const unsigned int N_cells)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cells)
        return;

    const double4 vel_cell = d_cell_vel[idx];
    const double mass = vel_cell.w;

    // add accumulated momentum to net momentum
    mpcd::detail::cell_thermo_element thermo;
    thermo.momentum = make_double3(vel_cell.x * mass, vel_cell.y * mass, vel_cell.z * mass);

    if (need_energy)
        {
        if (d_cell_np[idx] > 1)
            {
            thermo.temperature = d_cell_temp[idx];
            thermo.flag = 1;
            }
        else
            {
            thermo.temperature = 0.;
            thermo.flag = 0;
            }
        thermo.energy = d_cell_energy[idx];
        }
    else
        {
        thermo.energy = 0.;
        thermo.temperature = 0.;
        thermo.flag = 0;
        }

    d_tmp_thermo[idx] = thermo;
    }

#ifdef ENABLE_MPI
/*!
 * \param d_ghost_dir_filter Filtered list of directions to send ghost
 * \param d_ghost_idx_filter Filtered list of ghost particle indexes
 * \param d_num_ghost_scan Total number of ghosts
 * \param d_ghost_dir Unfiltered list of directions for ghost particles
 * \param d_ghost_dir_scan Result of scanning list of directions (filter indexes)
 * \param N Number of particles in group
 */
__global__ void filter_ghosts(unsigned int* d_ghost_dir_filter,
                              unsigned int* d_ghost_idx_filter,
                              unsigned int* d_num_ghost_scan,
                              const unsigned int* d_ghost_dir,
                              const unsigned int* d_ghost_dir_scan,
                              unsigned int N)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // load direction and exit early if invalid AND not the last particle
    const unsigned int dir = d_ghost_dir[idx];
    const unsigned int valid_dir = ValidDirection()(dir);
    const bool last_particle = idx == N - 1;
    if (!valid_dir && !last_particle)
        return;

    const unsigned int scan_result = d_ghost_dir_scan[idx];

    // only valid directions stored in filter
    if (valid_dir)
        {
        d_ghost_dir_filter[scan_result] = dir;
        d_ghost_idx_filter[scan_result] = idx;
        }

    // last particle knows the total number of ghosts if it adds its own result
    if (last_particle)
        {
        *d_num_ghost_scan = scan_result + valid_dir;
        }
    }

/*!
 * \param d_mpcd_send_offsets starting index of points sent to each neighbor
 * \param d_ghost_dir_sorted directions to send MPCD particles as ghosts
 * \param N Number of particles in group
 *
 * \b Implementation
 * Determines the starting index of the list of ghost particles in d_ghost_dir_sorted
 * and the total number of ghost particles to be sent. This is done by checking
 * the position 1 to the left in the array and seeing if it is a different
 * direction. If so, then the current position is the start indexing of the
 * current direction. The total number of ghosts is also written into the last index.
 */
__global__ void find_num_ghost_send(unsigned int* d_mpcd_send_offsets,
                                    const unsigned int* d_ghost_dir_sorted,
                                    const unsigned int N)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // the first thread always writes the number of ghosts into the last entry,
    // put that into flight first
    if (idx == 0)
        {
        d_mpcd_send_offsets[26] = N;
        }

    // load direction and check if it's a valid one to send to.
    // exit early for particles having invalid direction so we can assume all
    // have a valid direction later.
    const unsigned int dir = d_ghost_dir_sorted[idx];
    const bool valid_dir = ValidDirection()(dir);
    if (!valid_dir)
        return;

    // test if particles
    if (idx > 0)
        {
        // other particles need to compare with their left neighbor
        unsigned int left_dir = d_ghost_dir_sorted[idx - 1];
        if (dir != left_dir)
            {
            d_mpcd_send_offsets[dir] = idx;
            }
        }
    else
        {
        // the first particle has nobody to compare with and is always a start if it
        // has a valid direction.
        d_mpcd_send_offsets[dir] = 0;
        }
    }

/*!
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param d_vel MPCD particle velocities
 * \param d_ghost_idx_sorted indexes of MPCD particles to send as ghosts
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 *
 * \b Implementation
 * Fills the velocity buffer with ghost particles to send
 */
__global__ void fill_buffer(Scalar4* d_mpcd_vel_sendbuf,
                            const Scalar4* d_vel,
                            const unsigned int* d_ghost_idx_sorted,
                            const unsigned int num_mpcd_ghosts_send)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_mpcd_ghosts_send)
        return;

    const unsigned int particle_index = d_ghost_idx_sorted[idx];

    // add particle data to send buffers
    d_mpcd_vel_sendbuf[idx] = d_vel[particle_index];
    }

//! Kernel to compute add the contribution to cell properties from ghosts on GPU
/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy
 * \param d_conditions Conditions flags for error reporting
 * \param d_mpcd_ghost_vel ghost MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param N_mpcd_ghosts Number of ghost MPCD particles
 * \param need_energy True if computing the energy
 *
 * \b Implementation
 * One thread is launched per particle. The particle global bin index stored is
 * converted back into a 3 element global bin and used to compute the local bin
 * index. The contribution of the particles' properties is added to the cell's
 * properties in a running sum.
 */

__global__ void add_ghost_cell_properties(unsigned int* d_cell_np,
                                          double4* d_cell_vel,
                                          double* d_cell_energy,
                                          uint3* d_conditions,
                                          Scalar4* d_mpcd_ghost_vel,
                                          double mpcd_mass,
                                          const uint3 origin_idx,
                                          const uint3 global_cell_dim,
                                          const Index3D cell_indexer,
                                          const unsigned int N_mpcd_ghosts,
                                          const bool need_energy)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_mpcd_ghosts)
        return;

    double mass_i = mpcd_mass;
    const Scalar4 vel_mass_i = d_mpcd_ghost_vel[idx];
    const double3 vel_i = make_double3(vel_mass_i.x, vel_mass_i.y, vel_mass_i.z);

    // unset the highest bit of the global bin index
    unsigned int global_bin_index = __scalar_as_int(vel_mass_i.w);
    global_bin_index &= ~(1u << 31);

    // turn global bin index back into global bin
    int3 global_bin = make_int3(0, 0, 0);
    global_bin.x = global_bin_index % global_cell_dim.x;
    global_bin.y = (global_bin_index / global_cell_dim.x) % global_cell_dim.y;
    global_bin.z = int(global_bin_index / (global_cell_dim.x * global_cell_dim.y));

    // compute the local cell
    int3 bin = make_int3(global_bin.x - origin_idx.x,
                         global_bin.y - origin_idx.y,
                         global_bin.z - origin_idx.z);
    unsigned int bin_idx;
    if ((0 <= bin.x && bin.x < (int)cell_indexer.getW())
        && (0 <= bin.y && bin.y < (int)cell_indexer.getH())
        && (0 <= bin.z && bin.z < (int)cell_indexer.getD()))
        {
        bin_idx = cell_indexer(bin.x, bin.y, bin.z);
        }
    else
        {
        (*d_conditions).x = idx + 1;
        return;
        }

    // stash the current particle bin into the velocity array
    d_mpcd_ghost_vel[idx].w = __int_as_scalar(bin_idx);

    // compute the contribution of the particle to cell properties
    const unsigned int offset = atomicInc(&d_cell_np[bin_idx], 0xffffffff);
    double4& cell_vel = d_cell_vel[bin_idx];
    atomicAdd(&cell_vel.x, vel_i.x * mass_i);
    atomicAdd(&cell_vel.y, vel_i.y * mass_i);
    atomicAdd(&cell_vel.z, vel_i.z * mass_i);
    atomicAdd(&cell_vel.w, mass_i);

    if (need_energy)
        {
        double ke = 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
        atomicAdd(&d_cell_energy[bin_idx], ke);
        }
    }

/*!
 * \param d_vel MPCD particle velocities
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param d_ghost_idx_sorted indexes of MPCD particles to send as ghosts
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 *
 * \b Implementation
 * Updates the velocities of the particles with velocities from the send buffer

 */
__global__ void update_local_from_ghosts(Scalar4* d_vel,
                                         const Scalar4* d_mpcd_vel_sendbuf,
                                         const unsigned int* d_ghost_idx_sorted,
                                         unsigned int num_mpcd_ghosts_send)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_mpcd_ghosts_send)
        return;

    const Scalar4 vel_i = d_mpcd_vel_sendbuf[idx];
    const Scalar3 new_vel = make_scalar3(vel_i.x, vel_i.y, vel_i.z);

    const unsigned int particle_index = d_ghost_idx_sorted[idx];
    const Scalar4 old_vel = d_vel[particle_index];

    d_vel[particle_index] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, old_vel.w);
    }
#endif // ENABLE_MPI
    } // end namespace kernel
    } // end namespace gpu
    } // end namespace mpcd

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy, temperature, and dof
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_vel_embed Particle velocities
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 * \param need_energy True if computing the energy
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion, or an error on failure
 */
cudaError_t mpcd::gpu::compute_cell_list(unsigned int* d_cell_np,
                                         double4* d_cell_vel,
                                         double* d_cell_energy,
                                         uint3* d_conditions,
                                         Scalar4* d_vel,
                                         double mpcd_mass,
                                         unsigned int* d_embed_cell_ids,
                                         const Scalar4* d_pos,
                                         const Scalar4* d_pos_embed,
                                         const Scalar4* d_vel_embed,
                                         const unsigned int* d_embed_member_idx,
                                         const uchar3& periodic,
                                         const uint3& origin_idx,
                                         const Scalar3& grid_shift,
                                         const BoxDim& global_box,
                                         const uint3& global_cell_dim,
                                         const Index3D& cell_indexer,
                                         const Index3D& global_cell_indexer,
                                         unsigned int* d_ghost_dir,
                                         const uint3& rank_size,
                                         const bool is_decomposition,
                                         const unsigned int N_mpcd,
                                         const unsigned int N_tot,
                                         const bool need_energy,
                                         const unsigned int block_size)
    {
    // set the number of particles in each cell to zero
    cudaError_t error
        = cudaMemset(d_cell_np, 0, sizeof(unsigned int) * cell_indexer.getNumElements());
    if (error != cudaSuccess)
        return error;

    cudaError_t error_vel
        = cudaMemset(d_cell_vel, 0, sizeof(double4) * cell_indexer.getNumElements());
    if (error_vel != cudaSuccess)
        return error_vel;

    if (need_energy)
        {
        cudaError_t error_energy
            = cudaMemset(d_cell_energy, 0, sizeof(double) * cell_indexer.getNumElements());
        if (error_energy != cudaSuccess)
            return error_energy;
        }

    if (d_ghost_dir)
        {
        cudaError_t error_ghost_dir = cudaMemset(d_ghost_dir, 0xff, sizeof(unsigned int) * N_mpcd);
        if (error_ghost_dir != cudaSuccess)
            return error_ghost_dir;
        }

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::compute_cell_list);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::compute_cell_list<<<grid, run_block_size>>>(d_cell_np,
                                                                   d_cell_vel,
                                                                   d_cell_energy,
                                                                   d_conditions,
                                                                   d_vel,
                                                                   mpcd_mass,
                                                                   d_embed_cell_ids,
                                                                   d_pos,
                                                                   d_pos_embed,
                                                                   d_vel_embed,
                                                                   d_embed_member_idx,
                                                                   periodic,
                                                                   origin_idx,
                                                                   grid_shift,
                                                                   global_box,
                                                                   global_cell_dim,
                                                                   cell_indexer,
                                                                   global_cell_indexer,
                                                                   d_ghost_dir,
                                                                   rank_size,
                                                                   is_decomposition,
                                                                   N_mpcd,
                                                                   N_tot,
                                                                   need_energy);

    return cudaSuccess;
    }

/*!
 * \param d_cell_np Number of particles in cells
 * \param d_cell_vel Cell velocity to reduce
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature to finish computing
 * \param N_cells Number of total cells
 * \param N_dim Number of dimensions
 * \param need_energy If true, compute the cell-level energy properties
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::finish_cell_properties
 */
cudaError_t mpcd::gpu::finish_cell_properties(const unsigned int* d_cell_np,
                                              double4* d_cell_vel,
                                              const double* d_cell_energy,
                                              double* d_cell_temp,
                                              const unsigned int N_cells,
                                              const unsigned int N_dim,
                                              const bool need_energy,
                                              const unsigned int block_size)
    {
    // set all temperatures to zero
    if (need_energy)
        {
        cudaError_t error_temp = cudaMemset(d_cell_temp, 0, sizeof(double) * N_cells);
        if (error_temp != cudaSuccess)
            return error_temp;
        }

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::finish_cell_properties);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_cells / run_block_size + 1);
    mpcd::gpu::kernel::finish_cell_properties<<<grid, run_block_size>>>(d_cell_np,
                                                                        d_cell_vel,
                                                                        d_cell_energy,
                                                                        d_cell_temp,
                                                                        N_cells,
                                                                        N_dim,
                                                                        need_energy);

    return cudaSuccess;
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_np Number of particles per cell
 * \param d_cell_vel Cell velocity
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature
 * \param N_cells Number of total cells
 * \param need_energy If true, compute the cell-level energy properties
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::stage_net_cell_thermo
 */
cudaError_t mpcd::gpu::stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                             const unsigned int* d_cell_np,
                                             const double4* d_cell_vel,
                                             const double* d_cell_energy,
                                             const double* d_cell_temp,
                                             const unsigned int N_cells,
                                             const bool need_energy,
                                             const unsigned int block_size)
    {
    if (need_energy)
        {
        unsigned int max_block_size_energy;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<true>);
        max_block_size_energy = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(block_size, max_block_size_energy);
        dim3 grid(N_cells / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<true><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                 d_cell_np,
                                                                                 d_cell_vel,
                                                                                 d_cell_energy,
                                                                                 d_cell_temp,
                                                                                 N_cells);
        }
    else
        {
        unsigned int max_block_size_noenergy;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<false>);
        max_block_size_noenergy = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(block_size, max_block_size_noenergy);
        dim3 grid(N_cells / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<false><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                  d_cell_np,
                                                                                  d_cell_vel,
                                                                                  d_cell_energy,
                                                                                  d_cell_temp,
                                                                                  N_cells);
        }
    return cudaSuccess;
    }

/*!
 * \param d_reduced Cell thermo properties reduced across all cells (output on second call)
 * \param d_tmp Temporary storage for reduction (output on first call)
 * \param tmp_bytes Number of bytes allocated for temporary storage (output on first call)
 * \param d_tmp_thermo Cell thermo properties to reduce
 * \param N_cells The number of cells to reduce across
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
cudaError_t mpcd::gpu::reduce_net_cell_thermo(mpcd::detail::cell_thermo_element* d_reduced,
                                              void* d_tmp,
                                              size_t& tmp_bytes,
                                              const mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                              const size_t N_cells)
    {
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_tmp_thermo, d_reduced, (unsigned int)N_cells);
    return cudaSuccess;
    }

#ifdef ENABLE_MPI
/*!
 * \param d_tmp Temporary memory for sorting.
 * \param tmp_bytes Number of temporary bytes for sorting.
 * \param d_ghost_dir The particle types to sort.
 * \param d_ghost_dir_sorted The sorted particle types.
 * \param N Number of particles.
 */
void mpcd::gpu::scan_for_ghosts(void* d_tmp,
                                size_t& tmp_bytes,
                                unsigned int* d_ghost_dir,
                                unsigned int* d_ghost_dir_scan,
                                unsigned int N)
    {
    cub::DeviceScan::ExclusiveSum(d_tmp,
                                  tmp_bytes,
                                  thrust::make_transform_iterator(d_ghost_dir, ValidDirection()),
                                  d_ghost_dir_scan,
                                  N);
    }

/*!
 * \param d_ghost_dir_filter Filtered list of directions to send ghost
 * \param d_ghost_idx_filter Filtered list of ghost particle indexes
 * \param d_num_ghost_scan Total number of ghosts
 * \param d_ghost_dir Unfiltered list of directions for ghost particles
 * \param d_ghost_dir_scan Result of scanning list of directions (filter indexes)
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::filter_ghosts
 */
cudaError_t mpcd::gpu::filter_ghosts(unsigned int* d_ghost_dir_filter,
                                     unsigned int* d_ghost_idx_filter,
                                     unsigned int* d_num_ghost_scan,
                                     const unsigned int* d_ghost_dir,
                                     const unsigned int* d_ghost_dir_scan,
                                     unsigned int N,
                                     unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::filter_ghosts);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::filter_ghosts<<<grid, run_block_size>>>(d_ghost_dir_filter,
                                                               d_ghost_idx_filter,
                                                               d_num_ghost_scan,
                                                               d_ghost_dir,
                                                               d_ghost_dir_scan,
                                                               N);

    return cudaSuccess;
    }

/*!
 * \param d_tmp Temporary memory for sorting.
 * \param tmp_bytes Number of temporary bytes for sorting.
 * \param d_types The particle types to sort.
 * \param d_sorted_types The sorted particle types.
 * \param d_indexes The particle indexes to sort.
 * \param d_sorted_indexes The sorted particle indexes.
 * \param N Number of particles to sort.
 * \returns A pair of flags saying if the output data needs to be swapped with the input.
 *
 * The sorting is done using CUB with the DoubleBuffer. On the first call, the temporary memory
 * is sized. On the second call, the sorting is actually performed. The sorted data may not actually
 * lie in \a d_*_sorted because of the double buffers, but this sorting seems to be more efficient.
 * The user should accordingly swap the input and output arrays if the returned values are true.
 */
uchar2 mpcd::gpu::sort_ghosts_by_dir(void* d_tmp,
                                     size_t& tmp_bytes,
                                     unsigned int* d_ghost_dir,
                                     unsigned int* d_ghost_dir_sorted,
                                     unsigned int* d_ghost_idx,
                                     unsigned int* d_ghost_idx_sorted,
                                     const unsigned int N)
    {
    // max valid value of dir is 26, which can be encoded in 5 bits
    constexpr unsigned int num_bits = 5;
    cub::DoubleBuffer<unsigned int> d_keys(d_ghost_dir, d_ghost_dir_sorted);
    cub::DoubleBuffer<unsigned int> d_vals(d_ghost_idx, d_ghost_idx_sorted);
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, num_bits);

    uchar2 swap = make_uchar2(0, 0);
    if (d_tmp != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the sorted
        // array (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

/*!
 * \param d_mpcd_send_offsets starting index of points sent to each neighbor
 * \param d_ghost_dir_sorted directions to send MPCD particles as ghosts
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::find_num_ghost_send
 */
cudaError_t mpcd::gpu::find_num_ghost_send(unsigned int* d_mpcd_send_offsets,
                                           const unsigned int* d_ghost_dir_sorted,
                                           const unsigned int N,
                                           const unsigned int block_size)
    {
    // fill the offsets for *valid* directions with invalid sentinel
    // the last one doesn't need to be filled because it is always assigned by
    // the first thread
    cudaError_t error = cudaMemset(d_mpcd_send_offsets, 0xff, sizeof(unsigned int) * 26);
    if (error != cudaSuccess)
        return error;

    // prepare kernel
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::find_num_ghost_send);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::find_num_ghost_send<<<grid, run_block_size>>>(d_mpcd_send_offsets,
                                                                     d_ghost_dir_sorted,
                                                                     N);

    return cudaSuccess;
    }

/*!
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param d_vel MPCD particle velocities
 * \param d_ghost_idx_sorted indexes of MPCD particles to send as ghosts
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::fill_buffer
 */
cudaError_t mpcd::gpu::fill_buffer(Scalar4* d_mpcd_vel_sendbuf,
                                   const Scalar4* d_vel,
                                   const unsigned int* d_ghost_idx_sorted,
                                   unsigned int num_mpcd_ghosts_send,
                                   unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::fill_buffer);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(num_mpcd_ghosts_send / run_block_size + 1);
    mpcd::gpu::kernel::fill_buffer<<<grid, run_block_size>>>(d_mpcd_vel_sendbuf,
                                                             d_vel,
                                                             d_ghost_idx_sorted,
                                                             num_mpcd_ghosts_send);

    return cudaSuccess;
    }

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy
 * \param d_conditions Conditions flags for error reporting
 * \param d_mpcd_ghost_vel ghost MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param N_mpcd_ghosts Number of ghost MPCD particles
 * \param need_energy True if computing the energy
 *
 * \b Implementation
 * One thread is launched per particle. The particle is floored into a bin subject to a random grid
 * shift. The number of particles in that bin is atomically incremented and the contribution of the
 * particle's properties to the cell's properties is added to a running sum. If the addition of the
 * particle will not overflow the allocated memory, the particle is written into that bin.
 * Otherwise, a flag is set to resize the cell list and recompute. The MPCD particle's cell id is
 * stashed into the velocity array.
 */
cudaError_t mpcd::gpu::add_ghost_cell_properties(unsigned int* d_cell_np,
                                                 double4* d_cell_vel,
                                                 double* d_cell_energy,
                                                 uint3* d_conditions,
                                                 Scalar4* d_mpcd_ghost_vel,
                                                 double mpcd_mass,
                                                 const uint3& origin_idx,
                                                 const uint3& global_cell_dim,
                                                 const Index3D& cell_indexer,
                                                 const unsigned int N_mpcd_ghosts,
                                                 const bool need_energy,
                                                 const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::add_ghost_cell_properties);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_mpcd_ghosts / run_block_size + 1);
    mpcd::gpu::kernel::add_ghost_cell_properties<<<grid, run_block_size>>>(d_cell_np,
                                                                           d_cell_vel,
                                                                           d_cell_energy,
                                                                           d_conditions,
                                                                           d_mpcd_ghost_vel,
                                                                           mpcd_mass,
                                                                           origin_idx,
                                                                           global_cell_dim,
                                                                           cell_indexer,
                                                                           N_mpcd_ghosts,
                                                                           need_energy);

    return cudaSuccess;
    }

/*!
 * \param d_vel MPCD particle velocities
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param d_ghost_idx_sorted indexes of MPCD particles to send as ghosts
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::update_local_from_ghosts
 */
cudaError_t mpcd::gpu::update_local_from_ghosts(Scalar4* d_vel,
                                                const Scalar4* d_mpcd_vel_sendbuf,
                                                const unsigned int* d_ghost_idx_sorted,
                                                unsigned int num_mpcd_ghosts_send,
                                                unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::update_local_from_ghosts);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(num_mpcd_ghosts_send / run_block_size + 1);
    mpcd::gpu::kernel::update_local_from_ghosts<<<grid, run_block_size>>>(d_vel,
                                                                          d_mpcd_vel_sendbuf,
                                                                          d_ghost_idx_sorted,
                                                                          num_mpcd_ghosts_send);

    return cudaSuccess;
    }
#endif // ENABLE_MPI
    } // end namespace hoomd
