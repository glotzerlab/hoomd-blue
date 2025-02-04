// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_REDUCTION_OPERATORS_H_
#define MPCD_REDUCTION_OPERATORS_H_

#include <type_traits>

/*!
 * \file mpcd/ReductionOperators.h
 * \brief Declares and defines binary reduction operators that may be used by
 *        mpcd::CellCommunicator
 */

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace mpcd
    {
//! Operators
/*!
 * The ops namespace contains functors for the CPU and GPU for doing things
 * like reductions, etc.
 */
namespace ops
    {
//! Summed reduction
struct Sum
    {
    template<typename T> HOSTDEVICE T operator()(const T& a, const T& b) const
        {
        return (a + b);
        }
    };

template<> HOSTDEVICE inline Scalar4 Sum::operator()(const Scalar4& a, const Scalar4& b) const
    {
    return make_scalar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

//! Max reduction
struct Max
    {
    template<typename T> HOSTDEVICE T operator()(const T& a, const T& b) const
        {
        return (a > b) ? a : b;
        }
    };

//! Min reduction
struct Min
    {
    template<typename T> HOSTDEVICE T operator()(const T& a, const T& b) const
        {
        return (a < b) ? a : b;
        }
    };

struct BitwiseOr
    {
    template<typename T> HOSTDEVICE T operator()(const T& a, const T& b) const
        {
        static_assert(std::is_integral<T>::value, "Integer required for binary operators");
        return (a | b);
        }
    };

    } // end namespace ops

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_REDUCTION_OPERATORS_H_
