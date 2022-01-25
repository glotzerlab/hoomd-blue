// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __ALL_EXTERNAL_POTENTIALS__H__
#define __ALL_EXTERNAL_POTENTIALS__H__

#include "AllPairPotentials.h"
#include "EvaluatorExternalElectricField.h"
#include "EvaluatorExternalPeriodic.h"
#include "EvaluatorWalls.h"
#include "PotentialExternal.h"

#ifdef ENABLE_HIP
#include "PotentialExternalGPU.h"
#endif

/*! \file AllExternalPotentials.h
    \brief Handy list of typedefs for all of the templated external potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! External potential to impose periodic structure
typedef PotentialExternal<EvaluatorExternalPeriodic> PotentialExternalPeriodic;

//! Electric field
typedef PotentialExternal<EvaluatorExternalElectricField> PotentialExternalElectricField;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairLJ>> WallsPotentialLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairSLJ>> WallsPotentialSLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairExpandedMie>> WallsPotentialExpandedMie;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairForceShiftedLJ>> WallsPotentialForceShiftedLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairMie>> WallsPotentialMie;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairGauss>> WallsPotentialGauss;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairYukawa>> WallsPotentialYukawa;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairMorse>> WallsPotentialMorse;

#ifdef ENABLE_HIP
//! External potential to impose periodic structure on the GPU
typedef PotentialExternalGPU<EvaluatorExternalPeriodic> PotentialExternalPeriodicGPU;
typedef PotentialExternalGPU<EvaluatorExternalElectricField> PotentialExternalElectricFieldGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairLJ>> WallsPotentialLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairSLJ>> WallsPotentialSLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairExpandedMie>> WallsPotentialExpandedMieGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairForceShiftedLJ>>
    WallsPotentialForceShiftedLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairMie>> WallsPotentialMieGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairGauss>> WallsPotentialGaussGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairYukawa>> WallsPotentialYukawaGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairMorse>> WallsPotentialMorseGPU;

#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __EXTERNAL_POTENTIALS_H__
