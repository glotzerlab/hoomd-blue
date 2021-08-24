// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __PAIR_POTENTIALS__H__
#define __PAIR_POTENTIALS__H__

#include "EvaluatorPairBuckingham.h"
#include "EvaluatorPairCosineSquared.h"
#include "EvaluatorPairDLVO.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "EvaluatorPairDPDThermo.h"
#include "EvaluatorPairEwald.h"
#include "EvaluatorPairExpandedMie.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairFourier.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairLJ0804.h"
#include "EvaluatorPairLJ1208.h"
#include "EvaluatorPairMie.h"
#include "EvaluatorPairMoliere.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairOPP.h"
#include "EvaluatorPairReactionField.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairTWF.h"
#include "EvaluatorPairTable.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairZBL.h"
#include "PotentialPair.h"
#include "PotentialPairDPDThermo.h"

#ifdef ENABLE_HIP
#include "AllDriverPotentialPairGPU.cuh"
#include "PotentialPairDPDThermoGPU.cuh"
#include "PotentialPairDPDThermoGPU.h"
#include "PotentialPairGPU.h"
#endif

/*! \file AllPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for lj forces
typedef PotentialPair<EvaluatorPairLJ> PotentialPairLJ;
//! Pair potential force compute for gaussian forces
typedef PotentialPair<EvaluatorPairGauss> PotentialPairGauss;
//! Pair potential force compute for slj forces
typedef PotentialPair<EvaluatorPairSLJ> PotentialPairSLJ;
//! Pair potential force compute for yukawa forces
typedef PotentialPair<EvaluatorPairYukawa> PotentialPairYukawa;
//! Pair potential force compute for ewald forces
typedef PotentialPair<EvaluatorPairEwald> PotentialPairEwald;
//! Pair potential force compute for morse forces
typedef PotentialPair<EvaluatorPairMorse> PotentialPairMorse;
//! Pair potential force compute for dpd conservative forces
typedef PotentialPair<EvaluatorPairDPDThermo> PotentialPairDPD;
//! Pair potential force compute for Moliere forces
typedef PotentialPair<EvaluatorPairMoliere> PotentialPairMoliere;
//! Pair potential force compute for ZBL forces
typedef PotentialPair<EvaluatorPairZBL> PotentialPairZBL;
//! Pair potential force compute for dpd thermostat and conservative forces
typedef PotentialPairDPDThermo<EvaluatorPairDPDThermo> PotentialPairDPDThermoDPD;
//! Pair potential force compute for dpdlj conservative forces (not intended to be used)
typedef PotentialPair<EvaluatorPairDPDLJThermo> PotentialPairDPDLJ;
//! Pair potential force compute for dpd thermostat and LJ conservative forces
typedef PotentialPairDPDThermo<EvaluatorPairDPDLJThermo> PotentialPairDPDLJThermoDPD;
//! Pair potential force compute for force shifted LJ
typedef PotentialPair<EvaluatorPairForceShiftedLJ> PotentialPairForceShiftedLJ;
//! Pair potential force compute for Mie potential
typedef PotentialPair<EvaluatorPairMie> PotentialPairMie;
//! Pair potential force compute for Expanded Mie potential
typedef PotentialPair<EvaluatorPairExpandedMie> PotentialPairExpandedMie;
//! Pair potential force compute for ReactionField potential
typedef PotentialPair<EvaluatorPairReactionField> PotentialPairReactionField;
//! Pair potential force compute for Buckingham forces
typedef PotentialPair<EvaluatorPairBuckingham> PotentialPairBuckingham;
//! Pair potential force compute for lj1208 forces
typedef PotentialPair<EvaluatorPairLJ1208> PotentialPairLJ1208;
//! Pair potential force compute for lj0804 forces
typedef PotentialPair<EvaluatorPairLJ0804> PotentialPairLJ0804;
//! Pair potential force compute for DLVO potential
typedef PotentialPair<EvaluatorPairDLVO> PotentialPairDLVO;
//! Pair potential force compute for Fourier potential
typedef PotentialPair<EvaluatorPairFourier> PotentialPairFourier;
//! Pair potential force compute for oscillating pair potential
typedef PotentialPair<EvaluatorPairOPP> PotentialPairOPP;
/// Pair potential force compute for Ten wolde and Frenkels globular protein
/// model
typedef PotentialPair<EvaluatorPairTWF> PotentialPairTWF;
/// Pair potential force compute for Cosine Squared with Weeks-Chandler-Anderson repulsion
typedef PotentialPair<EvaluatorPairCosineSquared> PotentialPairCosineSquared;
/// Tabulateed pair potential
typedef PotentialPair<EvaluatorPairTable> PotentialPairTable;

#ifdef ENABLE_HIP
//! Pair potential force compute for lj forces on the GPU
typedef PotentialPairGPU<EvaluatorPairLJ, gpu_compute_ljtemp_forces> PotentialPairLJGPU;
//! Pair potential force compute for gaussian forces on the GPU
typedef PotentialPairGPU<EvaluatorPairGauss, gpu_compute_gauss_forces> PotentialPairGaussGPU;
//! Pair potential force compute for slj forces on the GPU
typedef PotentialPairGPU<EvaluatorPairSLJ, gpu_compute_slj_forces> PotentialPairSLJGPU;
//! Pair potential force compute for yukawa forces on the GPU
typedef PotentialPairGPU<EvaluatorPairYukawa, gpu_compute_yukawa_forces> PotentialPairYukawaGPU;
//! Pair potential force compute for ewald forces on the GPU
typedef PotentialPairGPU<EvaluatorPairEwald, gpu_compute_ewald_forces> PotentialPairEwaldGPU;
//! Pair potential force compute for morse forces on the GPU
typedef PotentialPairGPU<EvaluatorPairMorse, gpu_compute_morse_forces> PotentialPairMorseGPU;
//! Pair potential force compute for dpd conservative forces on the GPU
typedef PotentialPairGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermo_forces> PotentialPairDPDGPU;
//! Pair potential force compute for Moliere forces on the GPU
typedef PotentialPairGPU<EvaluatorPairMoliere, gpu_compute_moliere_forces> PotentialPairMoliereGPU;
//! Pair potential force compute for ZBL forces on the GPU
typedef PotentialPairGPU<EvaluatorPairZBL, gpu_compute_zbl_forces> PotentialPairZBLGPU;
//! Pair potential force compute for dpd thermostat and conservative forces on the GPU
typedef PotentialPairDPDThermoGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermodpd_forces>
    PotentialPairDPDThermoDPDGPU;
//! Pair potential force compute for dpdlj conservative forces on the GPU (not intended to be used)
typedef PotentialPairGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermo_forces>
    PotentialPairDPDLJGPU;
//! Pair potential force compute for dpd thermostat and LJ conservative forces on the GPU
typedef PotentialPairDPDThermoGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermodpd_forces>
    PotentialPairDPDLJThermoDPDGPU;
//! Pair potential force compute for force shifted LJ on the GPU
typedef PotentialPairGPU<EvaluatorPairForceShiftedLJ, gpu_compute_force_shifted_lj_forces>
    PotentialPairForceShiftedLJGPU;
//! Pair potential force compute for Mie potential
typedef PotentialPairGPU<EvaluatorPairMie, gpu_compute_mie_forces> PotentialPairMieGPU;
//! Pair potential force compute for shifted Mie potential on the GPU
typedef PotentialPairGPU<EvaluatorPairExpandedMie, gpu_compute_expanded_mie_forces>
    PotentialPairExpandedMieGPU;
//! Pair potential force compute for reaction field forces on the GPU
typedef PotentialPairGPU<EvaluatorPairReactionField, gpu_compute_reaction_field_forces>
    PotentialPairReactionFieldGPU;
//! Pair potential force compute for Buckingham forces on the GPU
typedef PotentialPairGPU<EvaluatorPairBuckingham, gpu_compute_buckingham_forces>
    PotentialPairBuckinghamGPU;
//! Pair potential force compute for lj1208 forces on the GPU
typedef PotentialPairGPU<EvaluatorPairLJ1208, gpu_compute_lj1208_forces> PotentialPairLJ1208GPU;
//! Pair potential force compute for lj0804 forces on the GPU
typedef PotentialPairGPU<EvaluatorPairLJ0804, gpu_compute_lj0804_forces> PotentialPairLJ0804GPU;
//! Pair potential force compute for DLVO forces on the GPU
typedef PotentialPairGPU<EvaluatorPairDLVO, gpu_compute_dlvo_forces> PotentialPairDLVOGPU;
//! Pair potential force compute for Fourier forces on the gpu
typedef PotentialPairGPU<EvaluatorPairFourier, gpu_compute_fourier_forces> PotentialPairFourierGPU;
//! Pair potential force compute for oscillating pair potential
typedef PotentialPairGPU<EvaluatorPairOPP, gpu_compute_opp_forces> PotentialPairOPPGPU;
//! Pair potential force compute for Table pair potential on the GPU
typedef PotentialPairGPU<EvaluatorPairTable, gpu_compute_table_forces> PotentialPairTableGPU;
//! Pair potential force compute for Cosine Squared pair potential on the GPU
typedef PotentialPairGPU<EvaluatorPairCosineSquared, gpu_compute_cosinesquared_forces>
    PotentialPairCosineSquaredGPU;
/// Pair potential force compute for Ten wolde and Frenkels globular protein
/// model
typedef PotentialPairGPU<EvaluatorPairTWF, gpu_compute_twf_forces> PotentialPairTWFGPU;

#endif

#endif // __PAIR_POTENTIALS_H__
