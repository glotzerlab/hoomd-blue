// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkGeometry.h
 * \brief Definition of the MPCD bulk geometry
 */

#ifndef MPCD_BULK_GEOMETRY_H_
#define MPCD_BULK_GEOMETRY_H_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // __HIPCC__

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
//! Bulk (periodic) geometry
/*!
 * This geometry is for a bulk fluid, and hence all of its methods simply indicate no collision
 * occurs. It exists so that we can leverage the ConfinedStreamingMethod integrator to still stream
 * in bulk.
 */
class __attribute__((visibility("default"))) BulkGeometry
    {
    public:
    //! Detect collision between the particle and the boundary
    /*!
     * \param pos Proposed particle position
     * \param vel Proposed particle velocity
     * \param dt Integration time remaining
     *
     * \returns True if a collision occurred, and false otherwise
     *
     * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel
     * is updated according to the appropriate bounce back rule, and the integration time \a dt is
     * decreased to the amount of time remaining.
     */
    HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
        {
        dt = Scalar(0);
        return false;
        }

    //! Check if a particle is out of bounds
    /*!
     * \param pos Current particle position
     * \returns True because particles are always in bounds in the bulk geometry.
     */
    HOSTDEVICE bool isOutside(const Scalar3& pos) const
        {
        return false;
        }

    //! Validate the simulation box
    /*!
     * \returns True because the simulation box is always big enough to hold a bulk geometry.
     */
    HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
        {
        return true;
        }

#ifndef __HIPCC__
    //! Get the unique name of this geometry
    static std::string getName()
        {
        return std::string("Bulk");
        }
#endif // __HIPCC__
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // MPCD_BULK_GEOMETRY_H_
