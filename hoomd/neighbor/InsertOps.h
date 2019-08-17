// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_INSERT_OPS_H_
#define NEIGHBOR_INSERT_OPS_H_

#include "hoomd/HOOMDMath.h"
#include "BoundingVolumes.h"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace neighbor
{

//! Reference implementation of an (almost trivial) tree insertion operation for points
/*!
 * The get() method returns a BoundingBox object, which will be used to instantiate a leaf node.
 */
struct PointInsertOp
    {
    //! Constructor
    /*!
     * \param points_ Points array (x,y,z,_)
     * \param N_ The number of points
     */
    PointInsertOp(const Scalar4 *points_, unsigned int N_)
        : points(points_), N(N_)
        {}

    #ifdef NVCC
    //! Get the bounding volume for a given primitive
    /*!
     * \param idx the index of the primitive
     *
     * \returns The enclosing BoundingBox
     */
    DEVICE BoundingBox get(const unsigned int idx) const
        {
        const Scalar4 point = points[idx];
        const Scalar3 p = make_scalar3(point.x, point.y, point.z);

        // construct the bounding box for a point
        return BoundingBox(p,p);
        }
    #endif

    //! Get the number of leaf node bounding volumes
    /*!
     * \returns The initial number of leaf nodes
     */
    HOSTDEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4 *points;
    unsigned int N;
    };

//! An insertion operation for spheres of constant radius
struct SphereInsertOp
    {
    //! Constructor
    /*!
     * \param points_ Sphere centers (x,y,z,_)
     * \param r_ Constant sphere radius
     * \param N_ The number of points
     */
    SphereInsertOp(const Scalar4 *points_, const Scalar r_, unsigned int N_)
        : points(points_), r(r_), N(N_)
        {}

    #ifdef NVCC
    //! Get the bounding volume for a given primitive
    /*!
     * \param idx the index of the primitive
     *
     * \returns The enclosing BoundingBox
     */
    DEVICE BoundingBox get(unsigned int idx) const
        {
        const Scalar4 point = points[idx];
        const Scalar3 lo = make_scalar3(point.x-r, point.y-r, point.z-r);
        const Scalar3 hi = make_scalar3(point.x+r, point.y+r, point.z+r);

        return BoundingBox(lo,hi);
        }
    #endif

    //! Get the number of leaf node bounding volumes
    /*!
     * \returns The initial number of leaf nodes
     */
    HOSTDEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4 *points;  //!< Sphere centers
    const Scalar r;         //!< Constant sphere radius
    unsigned int N;         //!< Number of spheres
    };

} // end namespace neighbor

#undef DEVICE
#undef HOSTDEVICE

#endif // NEIGHBOR_INSERT_OPS_H_
