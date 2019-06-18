// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __ALL_ANISO_PAIR_POTENTIALS__H__
#define __ALL_ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"

#include "EvaluatorPairGB.h"
#include "EvaluatorPairDipole.h"
#include "EvaluatorPairALJTable.h"

#ifdef ENABLE_CUDA
#include "AnisoPotentialPairGPU.h"
#include "AnisoPotentialPairGPU.cuh"
#include "AllDriverAnisoPotentialPairGPU.cuh"
#endif

/*! \file AllAnisoPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

//! Pair potential force compute for Gay-Berne forces and torques
typedef AnisoPotentialPair<EvaluatorPairGB> AnisoPotentialPairGB;
//! Pair potential force compute for dipole forces and torques
typedef AnisoPotentialPair<EvaluatorPairDipole> AnisoPotentialPairDipole;
//! Pair potential force compute for 2D anisotropic LJ forces and torques
typedef AnisoPotentialPair<EvaluatorPairALJTable<2> > AnisoPotentialPair2DALJ;
//! Pair potential force compute for 3D anisotropic LJ forces and torques
typedef AnisoPotentialPair<EvaluatorPairALJTable<3> > AnisoPotentialPairALJTable;

#ifdef ENABLE_CUDA
//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairGB,gpu_compute_pair_aniso_forces_gb> AnisoPotentialPairGBGPU;
//! Pair potential force compute for dipole forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairDipole,gpu_compute_pair_aniso_forces_dipole> AnisoPotentialPairDipoleGPU;
//! Pair potential force compute for 2D anisotropic LJ forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJTable<2>, gpu_compute_pair_aniso_forces_2DALJ> AnisoPotentialPair2DALJGPU;
//! Pair potential force compute for 3D anisotropicl LJ forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJTable<3>, gpu_compute_pair_aniso_forces_ALJTable> AnisoPotentialPairALJTableGPU;
#endif

//

#endif // __ALL_ANISO_PAIR_POTENTIALS_H__
