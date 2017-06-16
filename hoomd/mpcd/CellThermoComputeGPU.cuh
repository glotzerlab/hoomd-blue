// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoComputeGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellThermoComputeGPU
 */

#ifndef MPCD_CELL_THERMO_COMPUTE_GPU_CUH_
#define MPCD_CELL_THERMO_COMPUTE_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/Index1D.h"

#include <cuda_runtime.h>

namespace mpcd
{
namespace detail
{
#ifdef NVCC
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
#undef HOSTDEVICE
}

namespace gpu
{
//! Kernel driver to begin cell thermo compute of outer cells
cudaError_t begin_cell_thermo(Scalar4 *d_cell_vel,
                              Scalar3 *d_cell_energy,
                              const unsigned int *d_cells,
                              const unsigned int *d_cell_np,
                              const unsigned int *d_cell_list,
                              const Index2D& cli,
                              const Scalar4 *d_vel,
                              const unsigned int N_mpcd,
                              const Scalar mpcd_mass,
                              const Scalar4 *d_embed_vel,
                              const unsigned int *d_embed_cell,
                              const unsigned int num_cells,
                              const unsigned int block_size,
                              const unsigned int tpp);

//! Kernel driver to finalize cell thermo compute of outer cells
cudaError_t end_cell_thermo(Scalar4 *d_cell_vel,
                            Scalar3 *d_cell_energy,
                            const unsigned int *d_cells,
                            const unsigned int Ncell,
                            const unsigned int n_dimensions,
                            const unsigned int block_size);

//! Kernel driver to perform cell thermo compute for inner cells
cudaError_t inner_cell_thermo(Scalar4 *d_cell_vel,
                              Scalar3 *d_cell_energy,
                              const Index3D& ci,
                              const Index3D& inner_ci,
                              const uint3& offset,
                              const unsigned int *d_cell_np,
                              const unsigned int *d_cell_list,
                              const Index2D& cli,
                              const Scalar4 *d_vel,
                              const unsigned int N_mpcd,
                              const Scalar mpcd_mass,
                              const Scalar4 *d_embed_vel,
                              const unsigned int *d_embed_cell,
                              const unsigned int n_dimensions,
                              const unsigned int block_size,
                              const unsigned int tpp);

//! Kernel driver to stage cell properties for net thermo reduction
cudaError_t stage_net_cell_thermo(mpcd::detail::cell_thermo_element *d_tmp_thermo,
                                  const Scalar4 *d_cell_vel,
                                  const Scalar3 *d_cell_energy,
                                  const Index3D& tmp_ci,
                                  const Index3D& ci,
                                  const unsigned int block_size);

//! Wrapper to cub device reduce for cell thermo properties
cudaError_t reduce_net_cell_thermo(mpcd::detail::cell_thermo_element *d_reduced,
                                   void *d_tmp,
                                   size_t& tmp_bytes,
                                   const mpcd::detail::cell_thermo_element *d_tmp_thermo,
                                   const unsigned int Ncell);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_CELL_THERMO_COMPUTE_GPU_CUH_
