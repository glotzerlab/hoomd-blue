// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#ifndef __ALL_ANISO_PAIR_POTENTIALS__H__
#define __ALL_ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"

#include "EvaluatorPairDipole.h"
#include "EvaluatorPairGB.h"

#ifdef ENABLE_HIP
#include "AllDriverAnisoPotentialPairGPU.cuh"
#include "AnisoPotentialPairGPU.cuh"
#include "AnisoPotentialPairGPU.h"
#endif

/*! \file AllAnisoPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

namespace hoomd
    {
namespace md
    {
//! Pair potential force compute for Gay-Berne forces and torques
typedef AnisoPotentialPair<EvaluatorPairGB> AnisoPotentialPairGB;
//! Pair potential force compute for dipole forces and torques
typedef AnisoPotentialPair<EvaluatorPairDipole> AnisoPotentialPairDipole;

#ifdef ENABLE_HIP
//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairGB, kernel::gpu_compute_pair_aniso_forces_gb>
    AnisoPotentialPairGBGPU;
//! Pair potential force compute for dipole forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairDipole, kernel::gpu_compute_pair_aniso_forces_dipole>
    AnisoPotentialPairDipoleGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __ALL_ANISO_PAIR_POTENTIALS_H__
