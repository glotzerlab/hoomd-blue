// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BoundaryCondition.h
 * \brief Definition of valid MPCD boundary conditions.
 */

#ifndef MPCD_BOUNDARY_CONDITION_H_
#define MPCD_BOUNDARY_CONDITION_H_

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
//! Boundary conditions at the surface
/*!
 * Boundaries are currently allowed to either be "no slip" or "slip". The tangential
 * component of the fluid velocity is zero at a no-slip surface, while the shear stress
 * is zero at a slip surface. Both boundaries are no-penetration, so the normal component
 * of the fluid velocity is zero.
 */
enum struct boundary : unsigned char
    {
    no_slip = 0,
    slip
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_BOUNDARY_CONDITION_H_
