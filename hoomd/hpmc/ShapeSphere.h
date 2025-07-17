// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "Moves.h"
#include "hoomd/AABB.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/HPMCMiscFunctions.h"
#include "hoomd/hpmc/OBB.h"

#include "Moves.h"

#include <sstream>

#include <stdexcept>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
namespace hpmc
    {
/** HPMC shape parameter base class

    HPMC shape parameters must be aligned on 32-byte boundaries for AVX acceleration. The ShapeParam
    base class implements the necessary aligned memory allocation operations. It also provides
    empty load_shared and allocated_shared implementations which enabled caching deep copied managed
    data arrays in shared memory.

    TODO Move base methods out into their own file. ShapeSphere.h will then no longer need to be
          included by everything.
*/
struct ShapeParams
    {
    /// Custom new operator
    static void* operator new(std::size_t sz)
        {
        void* ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

    /// Custom new operator for arrays
    static void* operator new[](std::size_t sz)
        {
        void* ret = 0;
        int retval = posix_memalign(&ret, 32, sz);
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating aligned memory");
            }

        return ret;
        }

// GCC 12 misidentifies this as a mismatched new/delete, it doesn't realize this is the
// *implementation* of delete.
#if defined(__GNUC__) && __GNUC__ >= 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

    /// Custom delete operator
    static void operator delete(void* ptr)
        {
        free(ptr);
        }

    /// Custom delete operator for arrays
    static void operator delete[](void* ptr)
        {
        free(ptr);
        }

#if defined(__GNUC__) && __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

    /** Load dynamic data members into shared memory and increase pointer

        @param ptr Pointer to load data to (will be incremented)
        @param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
        {
        // default implementation does nothing
        }

    /** Determine size of the shared memory allocation

        @param ptr Pointer to increment
        @param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        // default implementation does nothing
        }
    };

/** Parameters that define a sphere shape

    Spheres in HPMC are defined by their radius. Spheres may or may not be orientable. The
    orientation of a sphere does not enter into the overlap check, but the particle's orientation
    may be used by other code paths (e.g. the patch potential).
*/
struct SphereParams : ShapeParams
    {
    /// The radius of the sphere
    ShortReal radius;

    /// True when move statistics should not be counted
    bool ignore;

    /// True when the shape may be oriented
    bool isOriented;

#ifdef ENABLE_HIP
    /// Set CUDA memory hints
    void set_memory_hint() const { }
#endif

#ifndef __HIPCC__

    /// Default constructor
    SphereParams() { }

    /// Construct from a Python dictionary
    SphereParams(pybind11::dict v, bool managed = false)
        {
        ignore = v["ignore_statistics"].cast<bool>();
        radius = v["diameter"].cast<ShortReal>() / ShortReal(2.0);
        isOriented = v["orientable"].cast<bool>();
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["diameter"] = radius * ShortReal(2.0);
        v["orientable"] = isOriented;
        v["ignore_statistics"] = ignore;
        return v;
        }

#endif
    } __attribute__((aligned(32)));

/** Sphere shape

    Shape classes define the interface used by IntegratorHPMCMono, ComputeFreeVolume, and other
    classes to check for overlaps between shapes, find their extend in space, and other operations.
    These classes are specified via template parameters to these classes so that the compiler may
    fully inline all uses of the shape API.

    ShapeSphere defines this API for spheres.

    Some portions of the API (e.g. test_overlap) are implemented as specialized function templates.

    TODO Should we remove orientation as a member variable from the shape API. It should be passed
          when needed.
    TODO Don't use specialized templates for things that should be methods (i.e. a.overlapsWith(b))
    TODO add hpmc::shape namespace
*/
struct ShapeSphere
    {
    /// Define the parameter type
    typedef SphereParams param_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeSphere(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation() const
        {
        return params.isOriented;
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return params.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        return params.radius * ShortReal(2.0);
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        return params.radius;
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return hoomd::detail::AABB(pos, params.radius);
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        return detail::OBB(pos, params.radius);
        }

    /// Returns true if this shape splits the overlap check over several threads of a warp using
    /// threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
        return false;
        }

    /// Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return true;
        }

    quat<Scalar> orientation; //!< Orientation of the sphere (unused)

    /// Sphere parameters
    const SphereParams& params;
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
template<class ShapeA, class ShapeB>
DEVICE inline bool
check_circumsphere_overlap(const vec3<LongReal>& r_ab, const ShapeA& a, const ShapeB& b)
    {
    LongReal r_squared = dot(r_ab, r_ab);
    LongReal diameter_sum = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (r_squared * LongReal(4.0) <= diameter_sum * diameter_sum);
    }

//! Define the general overlap function
/*! This is just a convenient spot to put this to make sure it is defined early
    \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err Incremented if there is an error condition. Left unchanged otherwise.
    \returns true when *a* and *b* overlap, and false when they are disjoint
*/
template<class ShapeA, class ShapeB>
DEVICE inline bool
test_overlap(const vec3<Scalar>& r_ab, const ShapeA& a, const ShapeB& b, unsigned int& err)
    {
    // default implementation returns true, will make it obvious if something calls this
    return true;
    }

//! Sphere-Sphere overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap<ShapeSphere, ShapeSphere>(const vec3<Scalar>& r_ab,
                                                          const ShapeSphere& a,
                                                          const ShapeSphere& b,
                                                          unsigned int& err)
    {
    vec3<ShortReal> dr(r_ab);

    ShortReal rsq = dot(dr, dr);

    ShortReal RaRb = a.params.radius + b.params.radius;
    if (rsq < RaRb * RaRb)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

//! sphere sweep distance
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
DEVICE inline ShortReal sweep_distance(const vec3<Scalar>& r_ab,
                                       const ShapeSphere& a,
                                       const ShapeSphere& b,
                                       const vec3<Scalar>& direction,
                                       unsigned int& err,
                                       vec3<Scalar>& collisionPlaneVector)
    {
    ShortReal sumR = a.params.radius + b.params.radius;
    ShortReal distSQ = ShortReal(dot(r_ab, r_ab));

    ShortReal d_parallel = ShortReal(dot(r_ab, direction));
    if (d_parallel <= 0) // Moving apart
        {
        return -1.0;
        };

    ShortReal discriminant = sumR * sumR - distSQ + d_parallel * d_parallel;
    if (discriminant < 0) // orthogonal distance larger than sum of radii
        {
        return -2.0;
        };

    ShortReal newDist = d_parallel - fast::sqrt(discriminant);

    if (newDist > 0)
        {
        collisionPlaneVector = r_ab - direction * Scalar(newDist);
        return newDist;
        }
    else
        {
        // Two particles overlapping [with negative sweepable distance]
        collisionPlaneVector = r_ab;
        return -10.0;
        }
    }

#ifndef __HIPCC__
template<class Shape> std::string getShapeSpec(const Shape& shape)
    {
    // default implementation
    throw std::runtime_error("Shape definition not supported for this shape class.");
    }

template<> inline std::string getShapeSpec(const ShapeSphere& sphere)
    {
    std::ostringstream shapedef;
    shapedef << "{\"type\": \"Sphere\", \"diameter\": " << sphere.params.radius * ShortReal(2.0)
             << "}";
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
