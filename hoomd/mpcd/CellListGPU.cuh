// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_CELL_LIST_GPU_CUH_
#define MPCD_CELL_LIST_GPU_CUH_

/*!
 * \file mpcd/CellListGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellListGPU
 */

#include <cuda_runtime.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif
//! Custom cell thermo element for reductions on the gpu
struct cell_thermo_element
    {
    double3 momentum;   //!< Momentum of the cell
    double energy;      //!< Energy of the cell
    double temperature; //!< Temperature of the cell (0 if < 2 particles)
    unsigned int flag;  //!< Flag to be used to compute filled cells

    //! Addition operator for summed reduction
    HOSTDEVICE cell_thermo_element operator+(const cell_thermo_element& other) const
        {
        cell_thermo_element sum;
        sum.momentum.x = momentum.x + other.momentum.x;
        sum.momentum.y = momentum.y + other.momentum.y;
        sum.momentum.z = momentum.z + other.momentum.z;
        sum.energy = energy + other.energy;
        sum.temperature = temperature + other.temperature;
        sum.flag = flag + other.flag;

        return sum;
        }
    };
    } // namespace detail
namespace gpu
    {
//! Kernel driver to compute mpcd cell list
cudaError_t compute_cell_list(unsigned int* d_cell_np,
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
                              const unsigned int block_size);

//! Kernel driver to finalize cell property calculation
cudaError_t finish_cell_properties(const unsigned int* d_cell_np,
                                   double4* d_cell_vel,
                                   const double* d_cell_energy,
                                   double* d_cell_temp,
                                   const unsigned int N_cells,
                                   const unsigned int N_dim,
                                   const bool need_energy,
                                   const unsigned int block_size);

//! Kernel driver to normalize center of mass velocity and compute net properties
cudaError_t stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                  const unsigned int* d_cell_np,
                                  const double4* d_cell_vel,
                                  const double* d_cell_energy,
                                  const double* d_cell_temp,
                                  const unsigned int N_cells,
                                  const bool need_energy,
                                  const unsigned int block_size);

//! Wrapper to cub device reduce for cell thermo properties
cudaError_t reduce_net_cell_thermo(mpcd::detail::cell_thermo_element* d_reduced,
                                   void* d_tmp,
                                   size_t& tmp_bytes,
                                   const mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                   const size_t N_cells);

#ifdef ENABLE_MPI
//! Scan for ghost particles in directions
void scan_for_ghosts(void* d_tmp,
                     size_t& tmp_bytes,
                     unsigned int* d_ghost_dir,
                     unsigned int* d_ghost_dir_scan,
                     unsigned int N);

//! Kernel driver to filter ghosts down from scanned results
cudaError_t filter_ghosts(unsigned int* d_ghost_dir_filter,
                          unsigned int* d_ghost_idx_filter,
                          unsigned int* d_num_ghost_scan,
                          const unsigned int* d_ghost_dir,
                          const unsigned int* d_ghost_dir_scan,
                          unsigned int N,
                          unsigned int block_size);

//! Sort ghosts by direction to send
uchar2 sort_ghosts_by_dir(void* d_tmp,
                          size_t& tmp_bytes,
                          unsigned int* d_ghost_dir,
                          unsigned int* d_ghost_dir_sorted,
                          unsigned int* d_ghost_idx,
                          unsigned int* d_ghost_idx_sorted,
                          const unsigned int N);

//! Kernel driver to determine how many particles will be sent as ghosts
cudaError_t find_num_ghost_send(unsigned int* d_mpcd_send_offsets,
                                const unsigned int* d_ghost_dir_sorted,
                                const unsigned int N,
                                const unsigned int block_size);

//! Kernel driver to fill up the send buffer
cudaError_t fill_buffer(Scalar4* d_mpcd_vel_sendbuf,
                        const Scalar4* d_vel,
                        const unsigned int* d_ghost_idx_sorted,
                        unsigned int num_mpcd_ghosts_send,
                        unsigned int block_size);

//! Kernel driver to add ghost particles to cells
cudaError_t add_ghost_cell_properties(unsigned int* d_cell_np,
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
                                      const unsigned int block_size);

//! Kernel driver to update the local velocities after collision
cudaError_t update_local_from_ghosts(Scalar4* d_vel,
                                     const Scalar4* d_mpcd_vel_sendbuf,
                                     const unsigned int* d_ghost_idx_sorted,
                                     unsigned int num_mpcd_ghosts_send,
                                     unsigned int block_size);
#endif // ENABLE_MPI
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CELL_LIST_GPU_CUH_
