// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_POTENTIALS__H__
#define __BOND_POTENTIALS__H__

#include "EvaluatorBondFENE.h"
#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondTether.h"
#include "PotentialBond.h"
//#include "PotentialMeshBond.h"

#ifdef ENABLE_HIP
#include "AllDriverPotentialBondGPU.cuh"
#include "PotentialBondGPU.h"
//#include "PotentialMeshBondGPU.h"
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
typedef PotentialBond<EvaluatorBondHarmonic, BondData> PotentialBondHarmonic;
//! Bond potential force compute for FENE forces
typedef PotentialBond<EvaluatorBondFENE, BondData> PotentialBondFENE;
//! Bond potential force compute for Tethering forces
typedef PotentialBond<EvaluatorBondTether, BondData> PotentialBondTether;

//! Mesh Bond potential force compute for harmonic forces
typedef PotentialBond<EvaluatorBondHarmonic, MeshBondData> PotentialMeshBondHarmonic;
//! Mesh Bond potential force compute for FENE forces
typedef PotentialBond<EvaluatorBondFENE, MeshBondData> PotentialMeshBondFENE;
//! Mesh Bond potential force compute for Tethering forces
typedef PotentialBond<EvaluatorBondTether, MeshBondData> PotentialMeshBondTether;

#ifdef ENABLE_HIP
//! Bond potential force compute for harmonic forces on the GPU
typedef PotentialBondGPU<EvaluatorBondHarmonic, BondData, 2, kernel::gpu_compute_harmonic_forces>
    PotentialBondHarmonicGPU;
//! Bond potential force compute for FENE forces on the GPU
typedef PotentialBondGPU<EvaluatorBondFENE, BondData, 2, kernel::gpu_compute_fene_forces>
    PotentialBondFENEGPU;
//! Bond potential force compute for Tethering forces on the GPU
typedef PotentialBondGPU<EvaluatorBondTether, BondData, 2, kernel::gpu_compute_tether_forces>
    PotentialBondTetherGPU;

//! Mesh Bond potential force compute for harmonic forces on the GPU
typedef PotentialBondGPU<EvaluatorBondHarmonic,
                         MeshBondData,
                         4,
                         kernel::gpu_compute_harmonic_forces>
    PotentialMeshBondHarmonicGPU;
//! Mesh Bond potential force compute for FENE forces on the GPU
typedef PotentialBondGPU<EvaluatorBondFENE, MeshBondData, 4, kernel::gpu_compute_fene_forces>
    PotentialMeshBondFENEGPU;
//! Mesh Bond potential force compute for Tethering forces on the GPU
typedef PotentialBondGPU<EvaluatorBondTether, MeshBondData, 4, kernel::gpu_compute_tether_forces>
    PotentialMeshBondTetherGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_POTENTIALS_H__
