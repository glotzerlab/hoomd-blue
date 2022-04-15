// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __ALL_ANISO_PAIR_POTENTIALS__H__
#define __ALL_ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"

#include "EvaluatorPairALJ.h"
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
//! Pair potential force compute for 2D anisotropic LJ forces and torques
typedef AnisoPotentialPair<EvaluatorPairALJ<2>> AnisoPotentialPairALJ2D;
//! Pair potential force compute for 3D anisotropic LJ forces and torques
typedef AnisoPotentialPair<EvaluatorPairALJ<3>> AnisoPotentialPairALJ3D;

#ifdef ENABLE_HIP
//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairGB>
    AnisoPotentialPairGBGPU;
//! Pair potential force compute for dipole forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairDipole>
    AnisoPotentialPairDipoleGPU;
//! Pair potential force compute for 2D anisotropic LJ forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<2>>
    AnisoPotentialPairALJ2DGPU;
//! Pair potential force compute for 3D anisotropicl LJ forces and torques on the GPU
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<3>>
    AnisoPotentialPairALJ3DGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __ALL_ANISO_PAIR_POTENTIALS_H__
