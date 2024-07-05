// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file EAMForceGPU.cuh
 \brief Declares GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
 */

#ifndef __EAMTexInterForceGPU_CUH__
#define __EAMTexInterForceGPU_CUH__

namespace hoomd
    {
namespace metal
    {
namespace kernel
    {
//! Collection of parameters for EAM force GPU kernels
struct EAMTexInterData
    {
    int ntypes;     //!< number of potential element types
    int nrho;       //!< number of tabulated values of F(rho)
    int nr;         //!< number of tabulated values of rho(r), r*phi(r)
    Scalar dr;      //!< interval of r
    Scalar rdr;     //!< 1.0 / dr
    Scalar drho;    //!< interval of rho
    Scalar rdrho;   //!< 1.0 / drho
    Scalar r_cut;   //!< cut-off radius
    Scalar r_cutsq; //!< r_cut^2
    };

//! Kernel driver that computes EAM forces on the GPU for EAMForceComputeGPU
hipError_t gpu_compute_eam_tex_inter_forces(Scalar4* d_force,
                                            Scalar* d_virial,
                                            const size_t virial_pitch,
                                            const unsigned int N,
                                            const Scalar4* d_pos,
                                            const BoxDim& box,
                                            const unsigned int* d_n_neigh,
                                            const unsigned int* d_nlist,
                                            const size_t* d_head_list,
                                            const size_t size_nlist,
                                            const EAMTexInterData* d_eam_data,
                                            Scalar* d_dFdP,
                                            const Scalar4* d_F,
                                            const Scalar4* d_rho,
                                            const Scalar4* d_rphi,
                                            const Scalar4* d_dF,
                                            const Scalar4* d_drho,
                                            const Scalar4* d_drphi,
                                            const unsigned int block_size);

    } // end namespace kernel
    } // end namespace metal
    } // end namespace hoomd

#endif
