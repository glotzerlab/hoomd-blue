#ifndef __PAIR_EXT_POTENTIALS__H__
#define __PAIR_EXT_POTENTIALS__H__

// need to include hoomd_config and PotentialPair here
#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialPair.h"

// include all of the evaluators that the plugin contains
#include "EvaluatorPairLJ2.h"

#ifdef ENABLE_CUDA
// PotentialPairGPU is the class that performs the pair computations on the GPU
#include "hoomd/PotentialPairGPU.h"
// AllDriverPotentialPairExtGPU.cuh is a header file containing the kernel driver functions for computing the pair
// potentials defined in this plugin. See it for more details
#include "AllDriverPotentialPairExtGPU.cuh"
#endif

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for LJ forces
typedef PotentialPair<EvaluatorPairLJ2> PotentialPairLJ2;

#ifdef ENABLE_CUDA
//! Pair potential force compute for LJ forces on the GPU
typedef PotentialPairGPU< EvaluatorPairLJ2, gpu_compute_lj2_forces > PotentialPairLJ2GPU;
#endif

#endif // __PAIR_EXT_POTENTIALS_H__
