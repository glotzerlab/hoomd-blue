/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __PAIR_POTENTIALS__H__
#define __PAIR_POTENTIALS__H__

#include "PotentialPair.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairEwald.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairDPDThermo.h"
#include "PotentialPairDPDThermo.h"
#include "EvaluatorPairMoliere.h"
#include "EvaluatorPairZBL.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "EvaluatorPairForceShiftedLJ.h"

#ifdef ENABLE_CUDA
#include "PotentialPairGPU.h"
#include "PotentialPairDPDThermoGPU.h"
#include "PotentialPairDPDThermoGPU.cuh"
#include "AllDriverPotentialPairGPU.cuh"
#endif

/*! \file AllPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef NVCC
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
//! Pair potential force compute for force shifted LJ on the GPU
typedef PotentialPair<EvaluatorPairForceShiftedLJ> PotentialPairForceShiftedLJ;



#ifdef ENABLE_CUDA
//! Pair potential force compute for lj forces on the GPU
typedef PotentialPairGPU< EvaluatorPairLJ, gpu_compute_ljtemp_forces > PotentialPairLJGPU;
//! Pair potential force compute for gaussian forces on the GPU
typedef PotentialPairGPU< EvaluatorPairGauss, gpu_compute_gauss_forces > PotentialPairGaussGPU;
//! Pair potential force compute for slj forces on the GPU
typedef PotentialPairGPU< EvaluatorPairSLJ, gpu_compute_slj_forces > PotentialPairSLJGPU;
//! Pair potential force compute for yukawa forces on the GPU
typedef PotentialPairGPU< EvaluatorPairYukawa, gpu_compute_yukawa_forces > PotentialPairYukawaGPU;
//! Pair potential force compute for ewald forces on the GPU
typedef PotentialPairGPU< EvaluatorPairEwald, gpu_compute_ewald_forces > PotentialPairEwaldGPU;
//! Pair potential force compute for morse forces on the GPU
typedef PotentialPairGPU< EvaluatorPairMorse, gpu_compute_morse_forces > PotentialPairMorseGPU;
//! Pair potential force compute for dpd conservative forces on the GPU
typedef PotentialPairGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermo_forces > PotentialPairDPDGPU;
//! Pair potential force compute for Moliere forces on the GPU
typedef PotentialPairGPU<EvaluatorPairMoliere, gpu_compute_moliere_forces > PotentialPairMoliereGPU;
//! Pair potential force compute for ZBL forces on the GPU
typedef PotentialPairGPU<EvaluatorPairZBL, gpu_compute_zbl_forces > PotentialPairZBLGPU;
//! Pair potential force compute for dpd thermostat and conservative forces on the GPU
typedef PotentialPairDPDThermoGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermodpd_forces > PotentialPairDPDThermoDPDGPU;
//! Pair potential force compute for dpdlj conservative forces on the GPU (not intended to be used)
typedef PotentialPairGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermo_forces > PotentialPairDPDLJGPU;
//! Pair potential force compute for dpd thermostat and LJ conservative forces on the GPU
typedef PotentialPairDPDThermoGPU<EvaluatorPairDPDLJThermo, gpu_compute_dpdljthermodpd_forces > PotentialPairDPDLJThermoDPDGPU;
//! Pair potential force compute for force shifted LJ on the GPU
typedef PotentialPairGPU<EvaluatorPairForceShiftedLJ, gpu_compute_force_shifted_lj_forces> PotentialPairForceShiftedLJGPU;

#endif

#endif // __PAIR_POTENTIALS_H__
