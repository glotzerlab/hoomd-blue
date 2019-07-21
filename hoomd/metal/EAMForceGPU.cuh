// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: Lin Yang, Alex Travesset
// Previous Maintainer: Morozov

#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

/*! \file EAMForceGPU.cuh
 \brief Declares GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
 */

#ifndef __EAMTexInterForceGPU_CUH__
#define __EAMTexInterForceGPU_CUH__

//! Collection of parameters for EAM force GPU kernels
struct EAMTexInterData
    {
    int ntypes;             //!< number of potential element types
    int nrho;               //!< number of tabulated values of F(rho)
    int nr;                 //!< number of tabulated values of rho(r), r*phi(r)
    int block_size;         //!< block size, for GPU kernel
    Scalar dr;              //!< interval of r
    Scalar rdr;             //!< 1.0 / dr
    Scalar drho;            //!< interval of rho
    Scalar rdrho;           //!< 1.0 / drho
    Scalar r_cut;           //!< cut-off radius
    Scalar r_cutsq;         //!< r_cut^2
    };

//! Kernel driver that computes EAM forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_tex_inter_forces(Scalar4* d_force, Scalar* d_virial, const unsigned int virial_pitch,
        const unsigned int N, const Scalar4 *d_pos, const BoxDim& box, const unsigned int *d_n_neigh,
        const unsigned int *d_nlist, const unsigned int *d_head_list, const unsigned int size_nlist,
        const EAMTexInterData& eam_data, Scalar *d_dFdP, const Scalar4 *d_F, const Scalar4 *d_rho,
        const Scalar4 *d_rphi, const Scalar4 *d_dF, const Scalar4 *d_drho, const Scalar4 *d_drphi);

#endif
