// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: Alex Travesset

/**
 powered by:
 Iowa State University.
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
struct EAMTexInterData {
	int ntypes;             //!< number of potential element types
	int nr;                 //!< number of tabulated values of rho(r), r*phi(r)
	int nrho;               //!< number of tabulated values of F(rho)
	int block_size;         //!< block size, for GPU kernel
	Scalar dr;              //!< interval of r
	Scalar rdr;             //!< 1.0 / dr
	Scalar drho;            //!< interval of rho
	Scalar rdrho;           //!< 1.0 / drho
	Scalar r_cut;           //!< cut-off radius
	Scalar r_cutsq;         //!< r_cut^2
};

//! Collection of pointers for EAM force GPU kernels
struct EAMTexInterArrays {
	Scalar* atomDerivativeEmbeddingFunction; //!< array d(F(rho))/drho for each particle
};

//! Collection of cuda Arrays for EAM force GPU kernels
struct EAMtex {
	cudaArray* electronDensity;              //!< array rho(r), electron density
	cudaArray* pairPotential;                //!< array r*phi(r), pairwise energy
	cudaArray* embeddingFunction;            //!< array F(rho), embedding energy
	cudaArray* derivativeElectronDensity;    //!< array d(rho(r))/dr
	cudaArray* derivativePairPotential;      //!< array d(r*phi(r))/dr
	cudaArray* derivativeEmbeddingFunction;  //!< array d(F(rho))/drho

};

//! Kernel driver that computes EAM forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_tex_inter_forces(Scalar4* d_force, Scalar* d_virial,
		const unsigned int virial_pitch, const unsigned int N,
		const Scalar4 *d_pos, const BoxDim& box, const unsigned int *d_n_neigh,
		const unsigned int *d_nlist, const unsigned int *d_head_list,
		const unsigned int size_nlist, const EAMtex& eam_tex,
		const EAMTexInterArrays& eam_arrays, const EAMTexInterData& eam_data,
		const unsigned int compute_capability,
		const unsigned int max_tex1d_width);

#endif
