// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file RNGIdentifiers.h
    \brief Define constants to use in seeding separate RNG streams across different classes in the code

    There should be no correlations between the random numbers used, for example, in the Langevin thermostat
    and the velocity randomization routine. To ensure this a maintainable way, this file lists all of the
    constants in one location and the individual uses of RandomGenerator use the constant by name.

    By convention, the RNG identifier should be the first argument to RandomGenerator and the user provided seed the
    second.

    The actual values of these identifiers does not matter, so long as they are unique.

    TODO: Update HPMCMonoImplicit* after the refactor is merged
*/

#ifndef HOOMD_RNGIDENTIFIERS_H__
#define HOOMD_RNGIDENTIFIERS_H__

namespace hoomd {

struct RNGIdentifier
    {
    static const uint32_t ComputeFreeVolume = 0x23ed56f2;
    static const uint32_t HPMCMonoShuffle = 0xfa870af6;
    static const uint32_t HPMCMonoTrialMove = 0x754dea60;
    static const uint32_t HPMCMonoShift = 0xf4a3210e;
    static const uint32_t UpdaterBoxMC= 0xf6a510ab;
    static const uint32_t UpdaterClusters =  0x09365bf5;
    static const uint32_t UpdaterClustersPairwise = 0x50060112;
    static const uint32_t UpdaterExternalFieldWall = 0xba015a6f;
    static const uint32_t UpdaterMuVT = 0x186df7ba;
    static const uint32_t UpdaterMuVTBox1 = 0x05d4a502;
    static const uint32_t UpdaterMuVTBox2 = 0xa74201bd;
    static const uint32_t ActiveForceCompute = 0x7edf0a42;
    static const uint32_t EvaluatorPairDPDThermo = 0x4a84f5d0;
    static const uint32_t IntegrationMethodTwoStep = 0x11df5642;
    static const uint32_t TwoStepBD = 0x431287ff;
    static const uint32_t TwoStepLangevin = 0x89abcdef;
    static const uint32_t TwoStepLangevinAngular = 0x19fe31ab;
    static const uint32_t TwoStepNPTMTK = 0x9db2f0ab;
    static const uint32_t TwoStepNVTMTK = 0x451234b9;
    static const uint32_t ATCollisionMethod = 0xf4009e6a;
    static const uint32_t CollisionMethod = 0x5af53be6;
    static const uint32_t SRDCollisionMethod = 0x7b61fda0;
    static const uint32_t SlitGeometryFiller = 0xdb68c12c;
    static const uint32_t SlitPoreGeometryFiller = 0xc7af9094;
    };

}

#endif
