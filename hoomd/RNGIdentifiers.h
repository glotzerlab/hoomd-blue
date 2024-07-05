// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file RNGIdentifiers.h
    \brief Define constants to use in seeding separate RNG streams across different classes in the
    code

    There should be no correlations between the random numbers used, for example, in the Langevin
    thermostat and the velocity randomization routine. To ensure this a maintainable way, this file
    lists all of the constants in one location and the individual uses of RandomGenerator use the
    constant by name.

    ID values >= 200 are reserved for use by external plugins.

    The actual values of these identifiers does not matter, so long as they are unique.
*/

#pragma once

#include <cstdint>

namespace hoomd
    {
struct RNGIdentifier
    {
    static const uint8_t ComputeFreeVolume = 0;
    static const uint8_t HPMCMonoShuffle = 1;
    static const uint8_t HPMCMonoTrialMove = 2;
    static const uint8_t HPMCMonoShift = 3;
    static const uint8_t HPMCDepletants = 4;
    static const uint8_t HPMCDepletantNum = 5;
    static const uint8_t HPMCMonoAccept = 6;
    static const uint8_t UpdaterBoxMC = 7;
    static const uint8_t UpdaterClusters = 8;
    static const uint8_t UpdaterClustersPairwise = 9;
    static const uint8_t UpdaterExternalFieldWall = 10;
    static const uint8_t UpdaterMuVT = 11;
    static const uint8_t UpdaterMuVTDepletants1 = 12;
    static const uint8_t UpdaterMuVTDepletants2 = 13;
    static const uint8_t UpdaterMuVTDepletants3 = 14;
    static const uint8_t UpdaterMuVTDepletants4 = 15;
    static const uint8_t UpdaterMuVTDepletants5 = 16;
    static const uint8_t UpdaterMuVTDepletants6 = 17;
    static const uint8_t UpdaterMuVTPoisson = 18;
    static const uint8_t UpdaterMuVTBox1 = 19;
    static const uint8_t UpdaterMuVTBox2 = 20;
    static const uint8_t ActiveForceCompute = 21;
    static const uint8_t EvaluatorPairDPDThermo = 22;
    static const uint8_t IntegrationMethodTwoStep = 23;
    static const uint8_t TwoStepBD = 24;
    static const uint8_t TwoStepLangevin = 25;
    static const uint8_t TwoStepLangevinAngular = 26;
    static const uint8_t TwoStepConstantPressureThermalizeBarostat = 27;
    static const uint8_t MTTKThermostat = 28;
    static const uint8_t ATCollisionMethod = 29;
    static const uint8_t CollisionMethod = 30;
    static const uint8_t SRDCollisionMethod = 31;
    static const uint8_t SlitGeometryFiller = 32;
    static const uint8_t SlitPoreGeometryFiller = 33;
    static const uint8_t UpdaterQuickCompress = 34;
    static const uint8_t ParticleGroupThermalize = 35;
    static const uint8_t HPMCDepletantsAccept = 36;
    static const uint8_t HPMCDepletantsClusters = 37;
    static const uint8_t HPMCDepletantNumClusters = 38;
    static const uint8_t HPMCMonoPatch = 39;
    static const uint8_t UpdaterClusters2 = 40;
    static const uint8_t HPMCMonoChainMove = 41;
    static const uint8_t UpdaterShapeUpdate = 42;
    static const uint8_t UpdaterShapeConstruct = 43;
    static const uint8_t HPMCShapeMoveUpdateOrder = 44;
    static const uint8_t BussiThermostat = 45;
    static const uint8_t ConstantPressure = 46;
    };

    } // namespace hoomd
