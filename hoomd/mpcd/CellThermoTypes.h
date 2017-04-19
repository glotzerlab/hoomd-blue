// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoTypes.h
 * \brief Defines enums for mpcd::CellThermoCompute
 */

#ifndef MPCD_CELL_THERMO_TYPES_H_
#define MPCD_CELL_THERMO_TYPES_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace mpcd
{
namespace detail
{

//! Indexes into 1d array containing scalar thermo properties
/*!
 * The last element contains the total number of properties.
 */
struct thermo_index
    {
    enum property
        {
        momentum_x=0,   //!< Net momentum in x
        momentum_y,     //!< Net momentum in y
        momentum_z,     //!< Net momentum in z
        energy,         //!< Net kinetic energy
        temperature,    //!< Average temperature
        num_quantities  //!< Total number of thermo quantities
        };
    };

//! Summed reduction of a Scalar3 which has an int stashed in the last element
struct SumScalar2Int
    {
    //! Functor to sum a Scalar3 having an int as the third element
    /*!
     * \param a First Scalar3
     * \param b Second Scalar3
     * \returns Sum of a and b, with the x and y components added by float and
     *          the z component added by integer
     */
    HOSTDEVICE Scalar3 operator()(const Scalar3& a, const Scalar3& b) const
        {
        return make_scalar3(a.x+b.x,
                            a.y+b.y,
                            __int_as_scalar(__scalar_as_int(a.z)+__scalar_as_int(b.z)));
        }
    };

} // end namespace detail
} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_CELL_THERMO_TYPES_H_
