/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander / Everyone is free to add additional potentials

/*! \file AllDriverPotentialPairGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__

#include "PotentialPairGPU.cuh"
#include "PotentialPairDPDThermoGPU.cuh"
#include "EvaluatorPairDPDThermo.h"

//! Compute lj pair forces on the GPU with PairEvaluatorLJ
cudaError_t gpu_compute_ljtemp_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const unsigned int *d_n_neigh,
                                      const unsigned int *d_nlist,
                                      const Index2D& nli,
                                      const float2 *d_params,
                                      const float *d_rcutsq,
                                      const float *d_ronsq,
                                      const unsigned int ntypes,
                                      const unsigned int block_size,
                                      const unsigned int shift_mode);

//! Compute gauss pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_gauss_forces(const gpu_force_data_arrays& force_data,
                                     const gpu_pdata_arrays &pdata,
                                     const gpu_boxsize &box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const Index2D& nli,
                                     const float2 *d_params,
                                     const float *d_rcutsq,
                                     const float *d_ronsq,
                                     const unsigned int ntypes,
                                     const unsigned int block_size,
                                     const unsigned int shift_mode);

//! Compute slj pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_slj_forces(const gpu_force_data_arrays& force_data,
                                   const gpu_pdata_arrays &pdata,
                                   const gpu_boxsize &box,
                                   const unsigned int *d_n_neigh,
                                   const unsigned int *d_nlist,
                                   const Index2D& nli,
                                   const float2 *d_params,
                                   const float *d_rcutsq,
                                   const float *d_ronsq,
                                   const unsigned int ntypes,
                                   const unsigned int block_size,
                                   const unsigned int shift_mode);

//! Compute yukawa pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_yukawa_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const unsigned int *d_n_neigh,
                                      const unsigned int *d_nlist,
                                      const Index2D& nli,
                                      const float2 *d_params,
                                      const float *d_rcutsq,
                                      const float *d_ronsq,
                                      const unsigned int ntypes,
                                      const unsigned int block_size,
                                      const unsigned int shift_mode);

//! Compute morse pair forces on the GPU with PairEvaluatorMorse
cudaError_t gpu_compute_morse_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const unsigned int *d_n_neigh,
                                      const unsigned int *d_nlist,
                                      const Index2D& nli,
                                      const float4 *d_params,
                                      const float *d_rcutsq,
                                      const float *d_ronsq,
                                      const unsigned int ntypes,
                                      const unsigned int block_size,
                                      const unsigned int shift_mode);

//! Compute dpd thermostat on GPU with PairEvaluatorDPDThermo 
cudaError_t gpu_compute_dpdthermodpd_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const unsigned int *d_n_neigh,
                                      const unsigned int *d_nlist,
                                      const Index2D& nli,
                                      const float2 *d_params,
                                      const float *d_rcutsq,
                                      const int ntypes,
                                      const dpd_pair_args& args);

//! Compute dpd conservative force on GPU with PairEvaluatorDPDThermo
cudaError_t gpu_compute_dpdthermo_forces(const gpu_force_data_arrays& force_data,
                                         const gpu_pdata_arrays &pdata,
                                         const gpu_boxsize &box,
                                         const unsigned int *d_n_neigh,
                                         const unsigned int *d_nlist,
                                         const Index2D& nli,
                                         const float2 *d_params,
                                         const float *d_rcutsq,
                                         const float *d_ronsq,
                                         const unsigned int ntypes,
                                         const unsigned int block_size,
                                         const unsigned int shift_mode);

//! Compute ewlad pair forces on the GPU with PairEvaluatorEwald
cudaError_t gpu_compute_ewald_forces(const gpu_force_data_arrays& force_data,
                                      const gpu_pdata_arrays &pdata,
                                      const gpu_boxsize &box,
                                      const gpu_nlist_array &nlist,
                                      float3 *d_params,
                                      float *d_rcutsq,
                                      float *d_ronsq,
                                      int ntypes,
                                      const pair_args& args);
#endif

