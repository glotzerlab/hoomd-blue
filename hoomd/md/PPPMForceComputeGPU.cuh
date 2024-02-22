// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include "hoomd/GPUPartition.cuh"

#include "hip/hip_runtime.h"

#if defined(ENABLE_HIP)
#ifdef __HIP_PLATFORM_HCC__
#include <hipfft.h>
#else
#include <cufft.h>
typedef cufftComplex hipfftComplex;
#endif
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
void gpu_assign_particles(const uint3 mesh_dim,
                          const uint3 n_ghost_bins,
                          const uint3 grid_dim,
                          unsigned int group_size,
                          const unsigned int* d_index_array,
                          const Scalar4* d_postype,
                          const Scalar* d_charge,
                          hipfftComplex* d_mesh,
                          hipfftComplex* d_mesh_scratch,
                          const unsigned int mesh_elements,
                          int order,
                          const BoxDim& box,
                          unsigned int block_size,
                          const Scalar* d_rho_coeff,
                          const hipDeviceProp_t& dev_prop,
                          const GPUPartition& gpu_partition);

void gpu_reduce_meshes(const unsigned int mesh_elements,
                       const hipfftComplex* d_mesh_scratch,
                       hipfftComplex* d_mesh,
                       const unsigned int ngpu,
                       const unsigned int block_size);

void gpu_compute_mesh_virial(const unsigned int n_wave_vectors,
                             hipfftComplex* d_fourier_mesh,
                             Scalar* d_inf_f,
                             Scalar* d_virial_mesh,
                             const Scalar3* d_k,
                             const bool exclude_dc,
                             Scalar kappa);

void gpu_update_meshes(const unsigned int n_wave_vectors,
                       hipfftComplex* d_fourier_mesh,
                       hipfftComplex* d_fourier_mesh_G_x,
                       hipfftComplex* d_fourier_mesh_G_y,
                       hipfftComplex* d_fourier_mesh_G_z,
                       const Scalar* d_inf_f,
                       const Scalar3* d_k,
                       unsigned int NNN,
                       unsigned int block_size);

void gpu_compute_forces(const unsigned int N,
                        const Scalar4* d_postype,
                        Scalar4* d_force,
                        const hipfftComplex* d_inv_fourier_mesh_x,
                        const hipfftComplex* d_inv_fourier_mesh_y,
                        const hipfftComplex* d_inv_fourier_mesh_z,
                        const uint3 grid_dim,
                        const uint3 n_ghost_cells,
                        const Scalar* d_charge,
                        const BoxDim& box,
                        int order,
                        const unsigned int* d_index_array,
                        const GPUPartition& gpu_partition,
                        const GPUPartition& all_gpu_partition,
                        const Scalar* d_rho_coeff,
                        unsigned int block_size,
                        bool local_fft,
                        unsigned int inv_mesh_elements);

void gpu_compute_pe(unsigned int n_wave_vectors,
                    Scalar* d_sum_partial,
                    Scalar* d_sum,
                    const hipfftComplex* d_fourier_mesh,
                    const Scalar* d_inf_f,
                    const unsigned int block_size,
                    const uint3 mesh_dim,
                    const bool exclude_dc);

void gpu_compute_virial(unsigned int n_wave_vectors,
                        Scalar* d_sum_virial_partial,
                        Scalar* d_sum_virial,
                        const Scalar* d_mesh_virial,
                        const unsigned int block_size);

void gpu_compute_influence_function(const uint3 mesh_dim,
                                    const uint3 global_dim,
                                    Scalar* d_inf_f,
                                    Scalar3* d_k,
                                    const BoxDim& global_box,
                                    const bool local_fft,
                                    const uint3 pidx,
                                    const uint3 pdim,
                                    const Scalar EPS_HOC,
                                    Scalar kappa,
                                    Scalar alpha,
                                    const Scalar* gf_b,
                                    int order,
                                    unsigned int block_size);

hipError_t gpu_fix_exclusions(Scalar4* d_force,
                              Scalar* d_virial,
                              const size_t virial_pitch,
                              const unsigned int N,
                              const Scalar4* d_pos,
                              const Scalar* d_charge,
                              const BoxDim& box,
                              const unsigned int* d_n_ex,
                              const unsigned int* d_exlist,
                              const Index2D nex,
                              Scalar kappa,
                              Scalar alpha,
                              unsigned int* d_group_members,
                              unsigned int group_size,
                              int block_size);

void gpu_initialize_coeff(Scalar* CPU_rho_coeff, int order, const GPUPartition& gpu_partition);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
