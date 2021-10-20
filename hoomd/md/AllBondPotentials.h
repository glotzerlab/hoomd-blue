// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __BOND_POTENTIALS__H__
#define __BOND_POTENTIALS__H__

#include "EvaluatorBondFENE.h"
#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondTether.h"
#include "PotentialBond.h"
#include "PotentialMeshBond.h"

#ifdef ENABLE_HIP
#include "AllDriverPotentialBondGPU.cuh"
#include "PotentialBondGPU.h"
#include "PotentialMeshBondGPU.h"
#endif

/*! \file AllBondPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

//! Bond potential force compute for harmonic forces
typedef PotentialBond<EvaluatorBondHarmonic> PotentialBondHarmonic;
//! Bond potential force compute for FENE forces
typedef PotentialBond<EvaluatorBondFENE> PotentialBondFENE;
//! Bond potential force compute for Tethering forces
typedef PotentialBond<EvaluatorBondTether> PotentialBondTether;

//! Mesh Bond potential force compute for harmonic forces
typedef PotentialMeshBond<EvaluatorBondHarmonic> PotentialMeshBondHarmonic;
//! Mesh Bond potential force compute for FENE forces
typedef PotentialMeshBond<EvaluatorBondFENE> PotentialMeshBondFENE;
//! Mesh Bond potential force compute for Tethering forces
typedef PotentialMeshBond<EvaluatorBondTether> PotentialMeshBondTether;

#ifdef ENABLE_HIP
//! Bond potential force compute for harmonic forces on the GPU
typedef PotentialBondGPU<EvaluatorBondHarmonic, gpu_compute_harmonic_forces>
    PotentialBondHarmonicGPU;
//! Bond potential force compute for FENE forces on the GPU
typedef PotentialBondGPU<EvaluatorBondFENE, gpu_compute_fene_forces> PotentialBondFENEGPU;
//! Bond potential force compute for Tethering forces on the GPU
typedef PotentialBondGPU<EvaluatorBondTether, gpu_compute_tether_forces> PotentialBondTetherGPU;

//! Mesh Bond potential force compute for harmonic forces on the GPU
typedef PotentialMeshBondGPU<EvaluatorBondHarmonic, gpu_compute_harmonic_forces_mesh>
    PotentialMeshBondHarmonicGPU;
//! Mesh Bond potential force compute for FENE forces on the GPU
typedef PotentialMeshBondGPU<EvaluatorBondFENE, gpu_compute_fene_forces_mesh> PotentialMeshBondFENEGPU;
//! Mesh Bond potential force compute for Tethering forces on the GPU
typedef PotentialMeshBondGPU<EvaluatorBondTether, gpu_compute_tether_forces_mesh> PotentialMeshBondTetherGPU;
#endif

#endif // __BOND_POTENTIALS_H__
