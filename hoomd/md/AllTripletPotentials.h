// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __TRIPLET_POTENTIALS__H__
#define __TRIPLET_POTENTIALS__H__

#include "PotentialPair.h"
#include "PotentialTersoff.h"
#include "EvaluatorTersoff.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorRevCross.h"

#ifdef ENABLE_HIP
#include "PotentialTersoffGPU.h"
#include "DriverTersoffGPU.cuh"
#endif

/*! \file AllTripletPotentials.h
    \brief Handy list of typedefs for all of the templated three-body potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

//! Three-body potential force compute for Tersoff forces
typedef PotentialTersoff< EvaluatorTersoff > PotentialTripletTersoff;

//! Three-body potential force compute forces for soft vdW fluid
typedef PotentialTersoff< EvaluatorSquareDensity > PotentialTripletSquareDensity;

//! Three-body potential force compute forces for reversible crosslinkers
typedef PotentialTersoff< EvaluatorRevCross > PotentialTripletRevCross;

#ifdef ENABLE_HIP
//! Three-body potential force compute for Tersoff forces on the GPU
typedef PotentialTersoffGPU< EvaluatorTersoff, gpu_compute_tersoff_forces > PotentialTripletTersoffGPU;
//! Three-body potential force compute for Tersoff forces on the GPU
typedef PotentialTersoffGPU< EvaluatorSquareDensity, gpu_compute_sq_density_forces > PotentialTripletSquareDensityGPU;
//! Three-body potential force compute for RevCross forces on the GPU
typedef PotentialTersoffGPU< EvaluatorRevCross, gpu_compute_revcross_forces > PotentialTripletRevCrossGPU;
#endif // ENABLE_HIP

#endif // __TRIPLET_POTENTIALS_H__
