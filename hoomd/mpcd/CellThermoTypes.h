// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellThermoTypes.h
 * \brief Defines enums for mpcd::CellThermoCompute
 */

#ifndef MPCD_CELL_THERMO_TYPES_H_
#define MPCD_CELL_THERMO_TYPES_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__ __forceinline__
#else
#define DEVICE
#include <bitset>
#endif

namespace hoomd
    {
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
        momentum_x = 0, //!< Net momentum in x
        momentum_y,     //!< Net momentum in y
        momentum_z,     //!< Net momentum in z
        energy,         //!< Net kinetic energy
        temperature,    //!< Average temperature
        num_quantities  //!< Total number of thermo quantities
        };
    };

#ifndef __HIPCC__
//! Flags for optional thermo data
typedef std::bitset<32> ThermoFlags;
#endif
//! Bits corresponding to optional cell thermo data
struct thermo_options
    {
    enum value
        {
        energy = 0 //!< Cell-level energy
        };
    };

struct CellVelocityPackOp
    {
    typedef double4 element;

    DEVICE element pack(const double4& val) const
        {
        return val;
        }

    DEVICE double4 unpack(const element& e, const double4& val) const
        {
        return make_double4(e.x + val.x, e.y + val.y, e.z + val.z, e.w + val.w);
        }
    };

struct CellEnergyPackOp
    {
    typedef double2 element;

    DEVICE element pack(const double3& val) const
        {
        element e;
        e.x = val.x;
        e.y = val.z;
        return e;
        }

    DEVICE double3 unpack(const element& e, const double3& val) const
        {
        return make_double3(e.x + val.x,
                            val.y,
                            __int_as_double(__double_as_int(e.y) + __double_as_int(val.z)));
        }
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#undef DEVICE

#endif // MPCD_CELL_THERMO_TYPES_H_
