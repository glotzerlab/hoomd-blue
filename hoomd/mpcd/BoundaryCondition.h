// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/BoundaryCondition.h
 * \brief Definition of valid MPCD boundary conditions.
 */

#ifndef MPCD_BOUNDARY_CONDITION_H_
#define MPCD_BOUNDARY_CONDITION_H_

namespace mpcd
{
namespace detail
{

//! Boundary conditions at the surface
enum struct boundary : unsigned char
    {
    no_slip=0,
    slip
    };

} // end namespace detail
} // end namespace mpcd

#endif // MPCD_BOUNDARY_CONDITION_H_
