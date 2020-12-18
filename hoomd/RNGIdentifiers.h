// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file RNGIdentifiers.h
    \brief Define constants to use in seeding separate RNG streams across different classes in the
    code

    There should be no correlations between the random numbers used, for example, in the Langevin
    thermostat and the velocity randomization routine. To ensure this a maintainable way, this file
    lists all of the constants in one location and the individual uses of RandomGenerator use the
    constant by name.

    By convention, the RNG identifier should be the first argument to RandomGenerator and the user
    provided seed the second.

    The actual values of these identifiers does not matter, so long as they are unique.
*/

#ifndef HOOMD_RNGIDENTIFIERS_H__
#define HOOMD_RNGIDENTIFIERS_H__

namespace hoomd {

enum class RNGIdentifier : uint8_t
    {
    ComputeFreeVolume,
    HPMCMonoShuffle,
    HPMCMonoTrialMove,
    HPMCMonoShift,
    HPMCDepletants,
    HPMCDepletantNum,
    HPMCMonoAccept,
    UpdaterBoxMC,
    UpdaterClusters,
    UpdaterClustersPairwise,
    UpdaterExternalFieldWall,
    UpdaterMuVT,
    UpdaterMuVTDepletants1,
    UpdaterMuVTDepletants2,
    UpdaterMuVTDepletants3,
    UpdaterMuVTDepletants4,
    UpdaterMuVTDepletants5,
    UpdaterMuVTDepletants6,
    UpdaterMuVTPoisson,
    UpdaterMuVTBox1,
    UpdaterMuVTBox2,
    ActiveForceCompute,
    EvaluatorPairDPDThermo,
    IntegrationMethodTwoStep,
    TwoStepBD,
    TwoStepLangevin,
    TwoStepLangevinAngular,
    TwoStepNPTMTK,
    TwoStepNVTMTK,
    ATCollisionMethod,
    CollisionMethod,
    SRDCollisionMethod,
    SlitGeometryFiller,
    SlitPoreGeometryFiller,
    UpdaterQuickCompress,
    ParticleGroupThermalize
    };

}

#endif
