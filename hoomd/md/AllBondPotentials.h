// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __BOND_POTENTIALS__H__
#define __BOND_POTENTIALS__H__

#include "PotentialBond.h"
#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondFENE.h"

#ifdef ENABLE_CUDA
#include "PotentialBondGPU.h"
#include "AllDriverPotentialBondGPU.cuh"
#endif

/*! \file AllBondPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Bond potential force compute for harmonic forces
typedef PotentialBond<EvaluatorBondHarmonic> PotentialBondHarmonic;
//! Bond potential force compute for FENE forces
typedef PotentialBond<EvaluatorBondFENE> PotentialBondFENE;

#ifdef ENABLE_CUDA
//! Bond potential force compute for harmonic forces on the GPU
typedef PotentialBondGPU< EvaluatorBondHarmonic, gpu_compute_harmonic_forces > PotentialBondHarmonicGPU;
//! Bond potential force compute for FENE forces on the GPU
typedef PotentialBondGPU< EvaluatorBondFENE, gpu_compute_fene_forces > PotentialBondFENEGPU;
#endif

#endif // __BOND_POTENTIALS_H__
