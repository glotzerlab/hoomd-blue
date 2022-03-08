// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_POTENTIALS__H__
#define __BOND_POTENTIALS__H__

#include "EvaluatorBondFENE.h"
#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondTether.h"
#include "PotentialBond.h"

#ifdef ENABLE_HIP
#include "AllDriverPotentialBondGPU.cuh"
#include "PotentialBondGPU.h"
#endif

/*! \file AllBondPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Bond potential force compute for harmonic forces
typedef PotentialBond<EvaluatorBondHarmonic> PotentialBondHarmonic;
//! Bond potential force compute for FENE forces
typedef PotentialBond<EvaluatorBondFENE> PotentialBondFENE;
//! Bond potential force compute for Tethering forces
typedef PotentialBond<EvaluatorBondTether> PotentialBondTether;

#ifdef ENABLE_HIP
//! Bond potential force compute for harmonic forces on the GPU
typedef PotentialBondGPU<EvaluatorBondHarmonic, kernel::gpu_compute_harmonic_forces>
    PotentialBondHarmonicGPU;
//! Bond potential force compute for FENE forces on the GPU
typedef PotentialBondGPU<EvaluatorBondFENE, kernel::gpu_compute_fene_forces> PotentialBondFENEGPU;
//! Bond potential force compute for Tethering forces on the GPU
typedef PotentialBondGPU<EvaluatorBondTether, kernel::gpu_compute_tether_forces>
    PotentialBondTetherGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_POTENTIALS_H__
