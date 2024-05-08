// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/NoForce.h
 * \brief Definition of mpcd::NoForce.
 */

#ifndef MPCD_NO_FORCE_H_
#define MPCD_NO_FORCE_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define INLINE inline
#else
#define HOSTDEVICE
#define INLINE inline __attribute__((always_inline))
#include <string>
#endif

namespace hoomd
    {
namespace mpcd
    {

//! No force on particles
class NoForce
    {
    public:
    //! Force evaluation method
    /*!
     * \param r Particle position.
     * \returns Force on the particle.
     *
     * This just returns zero, meaning no force. Hopefully the compiler will optimize this out!
     */
    HOSTDEVICE INLINE Scalar3 evaluate(const Scalar3& r) const
        {
        return make_scalar3(0, 0, 0);
        }

#ifndef __HIPCC__
    //! Get the unique name of this force
    static std::string getName()
        {
        return std::string("NoForce");
        }
#endif // __HIPCC__
    };

#ifndef __HIPCC__
namespace detail
    {
void export_NoForce(pybind11::module& m);
    } // end namespace detail
#endif // __HIPCC__

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE
#undef INLINE

#endif // MPCD_NO_FORCE_H_
