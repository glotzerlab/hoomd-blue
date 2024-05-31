// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CommunicatorUtilities.h
 * \brief Defines utilities for the mpcd::Communicator and mpcd::CommunicatorGPU classes
 */

#ifdef ENABLE_MPI
#ifndef MPCD_COMMUNICATOR_UTILITIES_H_
#define MPCD_COMMUNICATOR_UTILITIES_H_

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
enum struct face : unsigned char
    {
    east = 0,
    west,
    north,
    south,
    up,
    down
    };

enum struct send_mask : unsigned int
    {
    east = 1,
    west = 2,
    north = 4,
    south = 8,
    up = 16,
    down = 32
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_COMMUNICATOR_UTILITIES_H_
#endif // ENABLE_MPI
