// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_PARTICLE_DATA_UTILITIES_H_
#define MPCD_PARTICLE_DATA_UTILITIES_H_

/*!
 * \file mpcd/ParticleDataUtilities.h
 * \brief Utilities for mpcd::ParticleData on the CPU and GPU
 *
 * Some elements must be shared between the CPU and GPU code, and there is not
 * a good place to put this in either header because of HOOMD's include and define
 * organization. To avoid code duplication, this shared code is split out here
 * as utilities.
 *
 * This file should only include common sentinels, structures, etc.
 */

#include "hoomd/HOOMDMath.h"
namespace mpcd
{
namespace detail
{
//! Sentinel value to signify that this particle is not placed in a cell
const unsigned int NO_CELL = 0xffffffff;

#ifdef ENABLE_MPI
//! Structure to store packed MPCD particle data
/*!
 * This structure is used mostly for MPI communication during particle migration.
 *
 * \sa mpcd::ParticleData::addParticles
 * \sa mpcd::ParticleData::removeParticles
 */
struct pdata_element
    {
    Scalar4 pos;            //!< Position
    Scalar4 vel;            //!< Velocity
    unsigned int tag;       //!< Global tag
    unsigned int comm_flag; //!< Communication flag
    };
#endif // ENABLE_MPI

} // end namespace detail
} // end namespace mpcd

#endif // MPCD_PARTICLE_DATA_UTILITIES_H_
