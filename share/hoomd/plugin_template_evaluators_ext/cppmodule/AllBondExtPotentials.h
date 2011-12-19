#ifndef __BOND_EXT_POTENTIALS__H__
#define __BOND_EXT_POTENTIALS__H__

// need to include hoomd_config and PotentialBond here
#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialBond.h"

// include all of the evaluators that the plugin contains
#include "EvaluatorBondHarmonicDPD.h"

#ifdef ENABLE_CUDA
// PotentialBondGPU is the class that performs the bond computations on the GPU
#include "hoomd/PotentialBondGPU.h"
// AllDriverPotentialBondExtGPU.cuh is a header file containing the kernel driver functions for computing the bond
// potentials defined in this plugin. See it for more details
#include "AllDriverPotentialBondExtGPU.cuh"
#endif

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Bond potential force compute for Harmonic+DPD forces
typedef PotentialBond<EvaluatorBondHarmonicDPD> PotentialBondHarmonicDPD;

#ifdef ENABLE_CUDA
//! Bond potential force compute for Harmonic+DPD forces on the GPU
typedef PotentialBondGPU< EvaluatorBondHarmonicDPD, gpu_compute_harmonic_dpd_forces > PotentialBondHarmonicDPDGPU;
#endif

#endif // __BOND_EXT_POTENTIALS_H__
