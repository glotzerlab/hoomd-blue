// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_BOUNDING_VOLUMES_H_
#define NEIGHBOR_BOUNDING_VOLUMES_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace neighbor
{

//! Axis-aligned bounding box
/*!
 * A bounding box is defined by a lower and upper bound that should fully
 * enclose objects inside it. Internally, the bounds are stored using
 * single-precision floating-point values. If the bounds are given in double precision,
 * they are appropriately rounded down or up, respectively, to fully enclose the
 * given bounds.
 *
 * The BoundingBox also serves as an example of a general bounding volume. Every bounding
 * volume must implement constructors for both single and double precision specifiers.
 * They must also implement an overlap method with as many other bounding volumes as is
 * practical or required. At minimum, they must implement an overlap method with a
 * BoundingBox.
 */
struct BoundingBox
    {
    //! Default constructor
    BoundingBox() {}

    //! Single-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     */
    HOSTDEVICE BoundingBox(const float3& lo_, const float3& hi_)
        : lo(lo_), hi(hi_)
        {}

    #ifdef NVCC
    //! Double-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     *
     * \a lo_ is rounded down and \a hi_ is rounded up to the nearest fp32 representable value.
     *
     * \todo This needs a __host__ implementation that does not rely on CUDA intrinsics.
     */
    DEVICE BoundingBox(const double3& lo_, const double3& hi_)
        {
        lo = make_float3(__double2float_rd(lo_.x), __double2float_rd(lo_.y), __double2float_rd(lo_.z));
        hi = make_float3(__double2float_ru(hi_.x), __double2float_ru(hi_.y), __double2float_ru(hi_.z));
        }
    #endif

    DEVICE float3 getCenter() const
        {
        float3 c;
        c.x = 0.5f*(lo.x+hi.x);
        c.y = 0.5f*(lo.y+hi.y);
        c.z = 0.5f*(lo.z+hi.z);

        return c;
        }

    //! Test for overlap between two bounding boxes.
    /*!
     * \param box Bounding box.
     *
     * \returns True if this box overlaps \a box.
     *
     * The overlap test is performed using cheap comparison operators.
     * The two overlap if none of the dimensions of the box overlap.
     */
    HOSTDEVICE bool overlap(const BoundingBox& box) const
        {
        return !(hi.x < box.lo.x || lo.x > box.hi.x ||
                 hi.y < box.lo.y || lo.y > box.hi.y ||
                 hi.z < box.lo.z || lo.z > box.hi.z);
        }

    float3 lo;  //!< Lower bound of box
    float3 hi;  //!< Upper bound of box
    };

//! Bounding sphere
/*!
 * Implements a spherical volume with a given origin and radius that fully encloses
 * its objects. The sphere data is stored internally using single-precision values,
 * and its radius is padded to account for uncertainty due to rounding. Note that as
 * a result, the origin of the sphere may not be identical to its specified value if
 * double-precision was used.
 */
struct BoundingSphere
    {
    BoundingSphere() {}

    #ifdef NVCC
    //! Single-precision constructor.
    /*!
     * \param o Center of sphere.
     * \param rsq Squared radius of sphere.
     *
     * \a r is rounded up to ensure it fully encloses all data.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE BoundingSphere(const float3& o, const float r)
        {
        origin = o;
        Rsq = __fmul_ru(r,r);
        }

    //! Double-precision constructor.
    /*!
     * \param o Center of sphere.
     * \param rsq Squared radius of sphere.
     *
     * \a o is rounded down and \a r is padded to ensure the sphere
     * encloses all data.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE BoundingSphere(const double3& o, const double r)
        {
        const float3 lo = make_float3(__double2float_rd(o.x),
                                      __double2float_rd(o.y),
                                      __double2float_rd(o.z));
        const float3 hi = make_float3(__double2float_ru(o.x),
                                      __double2float_ru(o.y),
                                      __double2float_ru(o.z));
        const float delta = fmaxf(fmaxf(__fsub_ru(hi.x,lo.x),__fsub_ru(hi.y,lo.y)),__fsub_ru(hi.z,lo.z));
        const float R = __fadd_ru(__double2float_ru(r),delta);
        origin = make_float3(lo.x, lo.y, lo.z);
        Rsq = __fmul_ru(R,R);
        }

    //! Test for overlap between a sphere and a BoundingBox.
    /*!
     * \param box Bounding box.
     *
     * \returns True if the sphere overlaps \a box.
     *
     * The intersection test is performed by finding the closest point to \a o
     * that is inside the box using a sequence of min and max ops. The distance
     * to this point from \a o is then computed in round down mode. If the squared
     * distance between the point and \a o is less than \a Rsq, then the two
     * objects intersect.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE bool overlap(const BoundingBox& box) const
        {
        const float3 dr = make_float3(__fsub_rd(fminf(fmaxf(origin.x, box.lo.x), box.hi.x), origin.x),
                                      __fsub_rd(fminf(fmaxf(origin.y, box.lo.y), box.hi.y), origin.y),
                                      __fsub_rd(fminf(fmaxf(origin.z, box.lo.z), box.hi.z), origin.z));
        const float dr2 = __fmaf_rd(dr.x, dr.x, __fmaf_rd(dr.y, dr.y, __fmul_rd(dr.z,dr.z)));

        return (dr2 <= Rsq);
        }
    #endif

    float3 origin;  //!< Center of the sphere
    float Rsq;      //!< Squared radius of the sphere
    };

} // end namespace neighbor

#undef DEVICE
#undef HOSTDEVICE

#endif // NEIGHBOR_BOUNDING_VOLUMES_H_
