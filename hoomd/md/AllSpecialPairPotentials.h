// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __SPECIAL_PAIR_POTENTIALS__H__
#define __SPECIAL_PAIR_POTENTIALS__H__

#include "EvaluatorSpecialPairCoulomb.h"
#include "EvaluatorSpecialPairLJ.h"
#include "PotentialSpecialPair.h"

#ifdef ENABLE_HIP
#include "AllDriverPotentialSpecialPairGPU.cuh"
#include "PotentialSpecialPairGPU.h"
#endif

/*! \file AllSpecialPairPotentials.h
    \brief Handy list of typedefs for all of the templated special pair potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Special pair potential force compute for LJ forces
typedef PotentialSpecialPair<EvaluatorSpecialPairLJ> PotentialSpecialPairLJ;
//! Special pair potential force compute for Coulomb forces
typedef PotentialSpecialPair<EvaluatorSpecialPairCoulomb> PotentialSpecialPairCoulomb;

#ifdef ENABLE_HIP
//! Special potential force compute for LJ forces on the GPU
typedef PotentialSpecialPairGPU<EvaluatorSpecialPairLJ, kernel::gpu_compute_lj_forces>
    PotentialSpecialPairLJGPU;
//! Special potential force compute for Coulomb forces on the GPU
typedef PotentialSpecialPairGPU<EvaluatorSpecialPairCoulomb, kernel::gpu_compute_coulomb_forces>
    PotentialSpecialPairCoulombGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __SPECIAL_PAIR_POTENTIALS_H__
