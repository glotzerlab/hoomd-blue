// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __ALL_ANISO_PAIR_POTENTIALS__H__
#define __ALL_ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"

#include "EvaluatorPairALJ.h"
#include "EvaluatorPairGB.h"
#include "EvaluatorPairDipole.h"

#ifdef ENABLE_HIP
#include "AnisoPotentialPairGPU.h"
#include "AnisoPotentialPairGPU.cuh"
#include "AllDriverAnisoPotentialPairGPU.cuh"
#endif

/*! \file AllAnisoPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

//! Pair potential force compute for ALJ forces and torques in 2D
typedef AnisoPotentialPair<EvaluatorPairALJ<2>> AnisoPotentialPairALJ2D;

//! Pair potential force compute for ALJ forces and torques in 3D
typedef AnisoPotentialPair<EvaluatorPairALJ<3>> AnisoPotentialPairALJ3D;

//! Pair potential force compute for Gay-Berne forces and torques
typedef AnisoPotentialPair<EvaluatorPairGB> AnisoPotentialPairGB;
//! Pair potential force compute for dipole forces and torques
typedef AnisoPotentialPair<EvaluatorPairDipole> AnisoPotentialPairDipole;

#ifdef ENABLE_HIP
//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<2>,
                              gpu_compute_pair_aniso_forces_alj_2d> AnisoPotentialPairALJ2DGPU;

//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<3>,
                              gpu_compute_pair_aniso_forces_alj_3d> AnisoPotentialPairALJ3DGPU;

//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairGB,gpu_compute_pair_aniso_forces_gb> AnisoPotentialPairGBGPU;
//! Pair potential force compute for dipole forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairDipole,gpu_compute_pair_aniso_forces_dipole> AnisoPotentialPairDipoleGPU;
#endif

#endif // __ALL_ANISO_PAIR_POTENTIALS_H__
