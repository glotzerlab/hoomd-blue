// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



// Maintainer: morozov

/**
powered by:
Moscow group.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

/*! \file EAMForceGPU.cuh
    \brief Declares GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
*/

#ifndef __EAMTexInterForceGPU_CUH__
#define __EAMTexInterForceGPU_CUH__

//! Collection of parameters for EAM force GPU kernels
struct EAMTexInterData{
    int ntypes;             //!< Undocumented parameter
    int nr;                 //!< Undocumented parameter
    int nrho;               //!< Undocumented parameter
    int block_size;         //!< Undocumented parameter
    Scalar dr;               //!< Undocumented parameter
    Scalar rdr;              //!< Undocumented parameter
    Scalar drho;             //!< Undocumented parameter
    Scalar rdrho;            //!< Undocumented parameter
    Scalar r_cutsq;          //!< Undocumented parameter
    Scalar r_cut;            //!< Undocumented parameter
};

//! Collection of pointers for EAM force GPU kernels
struct EAMTexInterArrays{
    Scalar* atomDerivativeEmbeddingFunction;    //!< Undocumented parameter
};

//! Collection of cuda Arrays for EAM force GPU kernels
struct EAMtex{
    cudaArray* electronDensity;             //!< Undocumented parameter
    cudaArray* pairPotential;               //!< Undocumented parameter
    cudaArray* embeddingFunction;           //!< Undocumented parameter
    cudaArray* derivativeElectronDensity;   //!< Undocumented parameter
    cudaArray* derivativePairPotential;     //!< Undocumented parameter
    cudaArray* derivativeEmbeddingFunction; //!< Undocumented parameter

};

//! Kernel driver that computes lj forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_tex_inter_forces(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    const BoxDim& box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    const unsigned int size_nlist,
    const EAMtex& eam_tex,
    const EAMTexInterArrays& eam_arrays,
    const EAMTexInterData& eam_data,
    const unsigned int compute_capability,
    const unsigned int max_tex1d_width);

#endif
