// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __ALL_ANISO_PAIR_POTENTIALS__H__
#define __ALL_ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"

#include "EvaluatorPairALJ.h"
#include "EvaluatorPairDipole.h"
#include "EvaluatorPairGB.h"

#ifdef ENABLE_HIP
#include "AnisoPotentialPairGPU.cuh"
#include "AnisoPotentialPairGPU.h"
#endif

namespace hoomd
    {
namespace md
    {
typedef AnisoPotentialPair<EvaluatorPairGB> AnisoPotentialPairGB;
typedef AnisoPotentialPair<EvaluatorPairDipole> AnisoPotentialPairDipole;
typedef AnisoPotentialPair<EvaluatorPairALJ<2>> AnisoPotentialPairALJ2D;
typedef AnisoPotentialPair<EvaluatorPairALJ<3>> AnisoPotentialPairALJ3D;

#ifdef ENABLE_HIP
typedef AnisoPotentialPairGPU<EvaluatorPairGB>
    AnisoPotentialPairGBGPU;
typedef AnisoPotentialPairGPU<EvaluatorPairDipole>
    AnisoPotentialPairDipoleGPU;
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<2>>
    AnisoPotentialPairALJ2DGPU;
typedef AnisoPotentialPairGPU<EvaluatorPairALJ<3>>
    AnisoPotentialPairALJ3DGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __ALL_ANISO_PAIR_POTENTIALS_H__
