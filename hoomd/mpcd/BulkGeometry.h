// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/BulkGeometry.h
 * \brief Definition of the MPCD bulk geometry
 */

#ifndef MPCD_BULK_GEOMETRY_H_
#define MPCD_BULK_GEOMETRY_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // NVCC

namespace mpcd
{
namespace detail
{

class BulkGeometry
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
         * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel is updated
         *       according to the appropriate bounce back rule, and the integration time \a dt is decreased to the
         *       amount of time remaining.
         */
        HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
            {
            dt = Scalar(0);
            return false;
            }

        //! Validate the simulation box
        /*!
         * \returns True because the simulation box is always big enough to hold a bulk geometry.
         */
        HOSTDEVICE bool validateBox(const BoxDim box, const Scalar cell_size) const
            {
            return true;
            }

        HOSTDEVICE bool isOutside(const Scalar3& pos) const
            {
            return false;
            }

        #ifndef NVCC
        //! Get the unique name of this geometry
        static std::string getName()
            {
            return std::string("Bulk");
            }
        #endif // NVCC
    };

} // end namespace detail
} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_BULK_GEOMETRY_H_
