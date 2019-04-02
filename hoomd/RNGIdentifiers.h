// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file RNGIdentifiers.h
    \brief Define constants to use in seeding separate RNG streams across different classes in the code

    There should be no correlations between the random numbers used, for example, in the Langevin thermostat
    and the velocity randomization routine. To ensure this a maintainable way, this file lists all of the
    constants in one location and the individual uses of Saru use the constant by name.

    By convention, the RNG identifier should be the first argument to Saru and the user provided seed the second.

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
    };

}

#endif
