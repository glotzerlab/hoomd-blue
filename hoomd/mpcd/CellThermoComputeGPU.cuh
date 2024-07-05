// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellThermoComputeGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellThermoComputeGPU
 */

#ifndef MPCD_CELL_THERMO_COMPUTE_GPU_CUH_
#define MPCD_CELL_THERMO_COMPUTE_GPU_CUH_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <cuda_runtime.h>

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

//! Convenience struct for common parameters passed to GPU kernels
struct thermo_args_t
    {
    thermo_args_t(double4* cell_vel_,
                  double3* cell_energy_,
                  const unsigned int* cell_np_,
                  const unsigned int* cell_list_,
                  const Index2D& cli_,
                  const Scalar4* vel_,
                  const unsigned int N_mpcd_,
                  const Scalar mass_,
                  const Scalar4* embed_vel_,
                  const unsigned int* embed_idx_,
                  bool need_energy_)
        : cell_vel(cell_vel_), cell_energy(cell_energy_), cell_np(cell_np_), cell_list(cell_list_),
          cli(cli_), vel(vel_), N_mpcd(N_mpcd_), mass(mass_), embed_vel(embed_vel_),
          embed_idx(embed_idx_), need_energy(need_energy_)
        {
        }

    double4* cell_vel;    //!< Cell velocities (output)
    double3* cell_energy; //!< Cell energies (output)

    const unsigned int* cell_np;   //!< Number of particles per cell
    const unsigned int* cell_list; //!< MPCD cell list
    const Index2D cli;             //!< MPCD cell list indexer
    const Scalar4* vel;            //!< MPCD particle velocities
    const unsigned int N_mpcd;     //!< Number of MPCD particles
    const Scalar mass;             //!< MPCD particle mass
    const Scalar4* embed_vel;      //!< Embedded particle velocities
    const unsigned int* embed_idx; //!< Embedded particle indexes
    const bool need_energy;        //!< Flag if energy calculations are required
    };
#undef HOSTDEVICE
    } // namespace detail

namespace gpu
    {
//! Kernel driver to begin cell thermo compute of outer cells
cudaError_t begin_cell_thermo(const mpcd::detail::thermo_args_t& args,
                              const unsigned int* d_cells,
                              const unsigned int num_cells,
                              const unsigned int block_size,
                              const unsigned int tpp);

//! Kernel driver to finalize cell thermo compute of outer cells
cudaError_t end_cell_thermo(double4* d_cell_vel,
                            double3* d_cell_energy,
                            const unsigned int* d_cells,
                            const unsigned int Ncell,
                            const unsigned int n_dimensions,
                            const bool need_energy,
                            const unsigned int block_size);

//! Kernel driver to perform cell thermo compute for inner cells
cudaError_t inner_cell_thermo(const mpcd::detail::thermo_args_t& args,
                              const Index3D& ci,
                              const Index3D& inner_ci,
                              const uint3& offset,
                              const unsigned int n_dimensions,
                              const unsigned int block_size,
                              const unsigned int tpp);

//! Kernel driver to stage cell properties for net thermo reduction
cudaError_t stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                  const double4* d_cell_vel,
                                  const double3* d_cell_energy,
                                  const Index3D& tmp_ci,
                                  const Index3D& ci,
                                  const bool need_energy,
                                  const unsigned int block_size);

//! Wrapper to cub device reduce for cell thermo properties
cudaError_t reduce_net_cell_thermo(mpcd::detail::cell_thermo_element* d_reduced,
                                   void* d_tmp,
                                   size_t& tmp_bytes,
                                   const mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                   const size_t Ncell);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CELL_THERMO_COMPUTE_GPU_CUH_
