// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __ALL_EXTERNAL_POTENTIALS__H__
#define __ALL_EXTERNAL_POTENTIALS__H__

#include "PotentialExternal.h"
#include "EvaluatorExternalPeriodic.h"
#include "EvaluatorExternalElectricField.h"
#include "EvaluatorWalls.h"
#include "AllPairPotentials.h"

#ifdef ENABLE_CUDA
#include "PotentialExternalGPU.h"
#endif

/*! \file AllExternalPotentials.h
    \brief Handy list of typedefs for all of the templated external potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! External potential to impose periodic structure
typedef PotentialExternal<EvaluatorExternalPeriodic> PotentialExternalPeriodic;

//! Electric field
typedef PotentialExternal<EvaluatorExternalElectricField> PotentialExternalElectricField;

typedef PotentialExternal<EvaluatorWalls<EvaluatorPairLJ> > WallsPotentialLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairSLJ> > WallsPotentialSLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairForceShiftedLJ> > WallsPotentialForceShiftedLJ;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairMie> > WallsPotentialMie;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairGauss> > WallsPotentialGauss;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairYukawa> > WallsPotentialYukawa;
typedef PotentialExternal<EvaluatorWalls<EvaluatorPairMorse> > WallsPotentialMorse;


#ifdef ENABLE_CUDA
//! External potential to impose periodic structure on the GPU
typedef PotentialExternalGPU<EvaluatorExternalPeriodic> PotentialExternalPeriodicGPU;
typedef PotentialExternalGPU<EvaluatorExternalElectricField> PotentialExternalElectricFieldGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairLJ> > WallsPotentialLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairSLJ> > WallsPotentialSLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairForceShiftedLJ> > WallsPotentialForceShiftedLJGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairMie> > WallsPotentialMieGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairGauss> > WallsPotentialGaussGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairYukawa> > WallsPotentialYukawaGPU;
typedef PotentialExternalGPU<EvaluatorWalls<EvaluatorPairMorse> > WallsPotentialMorseGPU;

#endif

#endif // __EXTERNAL_POTENTIALS_H__
