// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

    /// Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

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

namespace detail
    {
//! APIs for depletant sampling
struct SamplingMethod
    {
    //! This API is used for fast sampling without the need for temporary storage
    enum enumNoStorage
        {
        no_storage = 0
        };

    //! This API is used for accurate sampling, requiring temporary storage
    /* Any hit returned by excludedVolumeOverlap through this API *must* also
       also be a hit for the fast API
     */
    enum enumAccurate
        {
        accurate = 0
        };
    };

    }; // namespace detail

//! Allocate memory for temporary storage in depletant simulations
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension

    \returns the number of Shape::depletion_storage_type elements requested for
    temporary storage
 */
template<typename Method, class Shape>
DEVICE inline unsigned int allocateDepletionTemporaryStorage(const Shape& shape_a,
                                                             const Shape& shape_b,
                                                             const vec3<Scalar>& r_ab,
                                                             ShortReal r,
                                                             unsigned int dim,
                                                             const Method)
    {
    // default implementation doesn't require temporary storage
    return 0;
    }

//! Initialize temporary storage in depletant simulations
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension
    \param storage a pointer to a pre-allocated memory region, the size of which has been
        pre-determined by a call to allocateDepletionTemporaryStorage
    \param V_sample the insertion volume
        V_sample has to to be precomputed for the overlapping shapes using
        getSamplingVolumeIntersection()

    \returns the number of Shape::depletion_storage_type elements initialized for temporary storage
 */
template<typename Method, class Shape>
DEVICE inline unsigned int
initializeDepletionTemporaryStorage(const Shape& shape_a,
                                    const Shape& shape_b,
                                    const vec3<Scalar>& r_ab,
                                    ShortReal r,
                                    unsigned int dim,
                                    typename Shape::depletion_storage_type* storage,
                                    const ShortReal V_sample,
                                    const Method)
    {
    // default implementation doesn't require temporary storage
    return 0;
    }

//! Test for overlap of excluded volumes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param dim the spatial dimension

    returns true if the covering of the intersection is non-empty
 */
template<typename Method, class Shape>
DEVICE inline bool excludedVolumeOverlap(const Shape& shape_a,
                                         const Shape& shape_b,
                                         const vec3<Scalar>& r_ab,
                                         ShortReal r,
                                         unsigned int dim,
                                         const Method)
    {
    if (dim == 3)
        {
        ShortReal Ra = ShortReal(0.5) * shape_a.getCircumsphereDiameter() + r;
        ShortReal Rb = ShortReal(0.5) * shape_b.getCircumsphereDiameter() + r;

        return (dot(r_ab, r_ab) <= (Ra + Rb) * (Ra + Rb));
        }
    else
        {
        hoomd::detail::AABB aabb_a = shape_a.getAABB(vec3<Scalar>(0.0, 0.0, 0.0));
        hoomd::detail::AABB aabb_b = shape_b.getAABB(r_ab);

        // extend AABBs by the excluded volume radius
        vec3<Scalar> lower_a = aabb_a.getLower();
        vec3<Scalar> upper_a = aabb_a.getUpper();
        lower_a.x -= r;
        lower_a.y -= r;
        lower_a.z -= r;
        upper_a.x += r;
        upper_a.y += r;
        upper_a.z += r;

        vec3<Scalar> lower_b = aabb_b.getLower();
        vec3<Scalar> upper_b = aabb_b.getUpper();
        lower_b.x -= r;
        lower_b.y -= r;
        lower_b.z -= r;
        upper_b.x += r;
        upper_b.y += r;
        upper_b.z += r;

        return aabb_b.overlaps(aabb_a);
        }
    }

//! Uniform rejection sampling in a volume covering the intersection of two shapes, defined by their
//! Minkowski sums with a sphere of radius r
/*! \param rng random number generator
    \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the returned point (relative to the origin == shape_a)
    \param dim the spatial dimension
    \param storage_sz the number of temporary storage elements of type
        Shape::depletion_storage_type passed
    \param storage the array of temporary storage elements

    \returns true if the point was not rejected
 */
template<typename Method, class RNG, class Shape>
DEVICE inline bool
sampleInExcludedVolumeIntersection(RNG& rng,
                                   const Shape& shape_a,
                                   const Shape& shape_b,
                                   const vec3<Scalar>& r_ab,
                                   ShortReal r,
                                   vec3<ShortReal>& p,
                                   unsigned int dim,
                                   unsigned int storage_sz,
                                   const typename Shape::depletion_storage_type* storage,
                                   const Method)
    {
    if (dim == 3)
        {
        ShortReal Ra = ShortReal(0.5) * shape_a.getCircumsphereDiameter() + r;
        ShortReal Rb = ShortReal(0.5) * shape_b.getCircumsphereDiameter() + r;

        if (dot(r_ab, r_ab) > (Ra + Rb) * (Ra + Rb))
            return false;

        vec3<ShortReal> dr(r_ab);
        ShortReal d = fast::sqrt(dot(dr, dr));

        // whether the intersection is the entire (smaller) sphere
        bool sphere = (d + Ra - Rb < ShortReal(0.0)) || (d + Rb - Ra < ShortReal(0.0));

        if (!sphere)
            {
            // heights spherical caps that constitute the intersection volume
            ShortReal ha = (Rb * Rb - (d - Ra) * (d - Ra)) / (ShortReal(2.0) * d);
            ShortReal hb = (Ra * Ra - (d - Rb) * (d - Rb)) / (ShortReal(2.0) * d);

            // volumes of spherical caps
            ShortReal Vcap_a = ShortReal(M_PI / 3.0) * ha * ha * (ShortReal(3.0) * Ra - ha);
            ShortReal Vcap_b = ShortReal(M_PI / 3.0) * hb * hb * (ShortReal(3.0) * Rb - hb);

            // choose one of the two caps randomly, with a weight proportional to their volume
            hoomd::UniformDistribution<ShortReal> u;
            ShortReal s = u(rng);
            bool cap_a = s < Vcap_a / (Vcap_a + Vcap_b);

            // generate a depletant position in the spherical cap
            if (cap_a)
                p = generatePositionInSphericalCap(rng, vec3<Scalar>(0.0, 0.0, 0.0), Ra, ha, dr);
            else
                p = generatePositionInSphericalCap(rng, dr, Rb, hb, -dr);
            }
        else
            {
            // generate a random position in the smaller sphere
            if (Ra < Rb)
                {
                p = generatePositionInSphere(rng, vec3<Scalar>(0.0, 0.0, 0.0), Ra);
                }
            else
                {
                p = vec3<ShortReal>(generatePositionInSphere(rng, dr, Rb));
                }
            }

        // sphere (cap) sampling is rejection free
        return true;
        }
    else
        {
        hoomd::detail::AABB aabb_a = shape_a.getAABB(vec3<Scalar>(0.0, 0.0, 0.0));
        hoomd::detail::AABB aabb_b = shape_b.getAABB(r_ab);

        if (!aabb_b.overlaps(aabb_a))
            return false;

        // extend AABBs by the excluded volume radius
        vec3<Scalar> lower_a = aabb_a.getLower();
        vec3<Scalar> upper_a = aabb_a.getUpper();
        lower_a.x -= r;
        lower_a.y -= r;
        lower_a.z -= r;
        upper_a.x += r;
        upper_a.y += r;
        upper_a.z += r;

        vec3<Scalar> lower_b = aabb_b.getLower();
        vec3<Scalar> upper_b = aabb_b.getUpper();
        lower_b.x -= r;
        lower_b.y -= r;
        lower_b.z -= r;
        upper_b.x += r;
        upper_b.y += r;
        upper_b.z += r;

        // we already know the AABBs are overlapping, compute their intersection
        vec3<Scalar> intersect_lower, intersect_upper;
        intersect_lower.x = detail::max(lower_a.x, lower_b.x);
        intersect_lower.y = detail::max(lower_a.y, lower_b.y);
        intersect_lower.z = detail::max(lower_a.z, lower_b.z);
        intersect_upper.x = detail::min(upper_a.x, upper_b.x);
        intersect_upper.y = detail::min(upper_a.y, upper_b.y);
        intersect_upper.z = detail::min(upper_a.z, upper_b.z);

        hoomd::detail::AABB aabb_intersect(intersect_lower, intersect_upper);
        p = vec3<ShortReal>(generatePositionInAABB(rng, aabb_intersect, dim));

        // AABB sampling always succeeds
        return true;
        }
    }

//! Get the sampling volume for an intersection of shapes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the returned point
    \param dim the spatial dimension

    If the shapes are not overlapping, return zero.

    returns the volume of the intersection
 */
template<typename Method, class Shape>
DEVICE inline ShortReal getSamplingVolumeIntersection(const Shape& shape_a,
                                                      const Shape& shape_b,
                                                      const vec3<Scalar>& r_ab,
                                                      ShortReal r,
                                                      unsigned int dim,
                                                      const Method)
    {
    if (dim == 3)
        {
        ShortReal Ra = ShortReal(0.5) * shape_a.getCircumsphereDiameter() + r;
        ShortReal Rb = ShortReal(0.5) * shape_b.getCircumsphereDiameter() + r;

        if (dot(r_ab, r_ab) > (Ra + Rb) * (Ra + Rb))
            return ShortReal(0.0);

        vec3<ShortReal> dr(r_ab);
        ShortReal d = fast::sqrt(dot(dr, dr));

        if ((d + Ra - Rb < ShortReal(0.0)) || (d + Rb - Ra < ShortReal(0.0)))
            {
            // the intersection is the entire (smaller) sphere
            return (Ra < Rb) ? ShortReal(M_PI * 4.0 / 3.0) * Ra * Ra * Ra
                             : ShortReal(M_PI * 4.0 / 3.0) * Rb * Rb * Rb;
            }
        else
            {
            // heights spherical caps that constitute the intersection volume
            ShortReal ha = (Rb * Rb - (d - Ra) * (d - Ra)) / (ShortReal(2.0) * d);
            ShortReal hb = (Ra * Ra - (d - Rb) * (d - Rb)) / (ShortReal(2.0) * d);

            // volumes of spherical caps
            ShortReal Vcap_a = ShortReal(M_PI / 3.0) * ha * ha * (ShortReal(3.0) * Ra - ha);
            ShortReal Vcap_b = ShortReal(M_PI / 3.0) * hb * hb * (ShortReal(3.0) * Rb - hb);

            // volume of intersection
            return Vcap_a + Vcap_b;
            }
        }
    else
        {
        hoomd::detail::AABB aabb_a = shape_a.getAABB(vec3<Scalar>(0.0, 0.0, 0.0));
        hoomd::detail::AABB aabb_b = shape_b.getAABB(r_ab);

        if (!aabb_b.overlaps(aabb_a))
            return ShortReal(0.0);

        // extend AABBs by the excluded volume radius
        vec3<Scalar> lower_a = aabb_a.getLower();
        vec3<Scalar> upper_a = aabb_a.getUpper();
        lower_a.x -= r;
        lower_a.y -= r;
        lower_a.z -= r;
        upper_a.x += r;
        upper_a.y += r;
        upper_a.z += r;

        vec3<Scalar> lower_b = aabb_b.getLower();
        vec3<Scalar> upper_b = aabb_b.getUpper();
        lower_b.x -= r;
        lower_b.y -= r;
        lower_b.z -= r;
        upper_b.x += r;
        upper_b.y += r;
        upper_b.z += r;

        // we already know the AABBs are overlapping, compute their intersection
        vec3<Scalar> intersect_lower, intersect_upper;
        intersect_lower.x = detail::max(lower_a.x, lower_b.x);
        intersect_lower.y = detail::max(lower_a.y, lower_b.y);
        intersect_lower.z = detail::max(lower_a.z, lower_b.z);
        intersect_upper.x = detail::min(upper_a.x, upper_b.x);
        intersect_upper.y = detail::min(upper_a.y, upper_b.y);
        intersect_upper.z = detail::min(upper_a.z, upper_b.z);

        // intersection AABB volume
        Scalar V
            = (intersect_upper.x - intersect_lower.x) * (intersect_upper.y - intersect_lower.y);
        if (dim == 3)
            V *= intersect_upper.z - intersect_lower.z;
        return (ShortReal)V;
        }
    }

//! Test if a point is in the intersection of two excluded volumes
/*! \param shape_a the first shape
    \param shape_b the second shape
    \param r_ab the separation vector between the two shapes (in the same image)
    \param r excluded volume radius
    \param p the point to test (relative to the origin == shape_a)
    \param dim the spatial dimension

    It is assumed that the circumspheres of the shapes are overlapping, otherwise the result is
   invalid

    The point p is in the world frame, with shape a at the origin

    returns true if the point was not rejected
 */
template<typename Method, class Shape>
DEVICE inline bool isPointInExcludedVolumeIntersection(const Shape& shape_a,
                                                       const Shape& shape_b,
                                                       const vec3<Scalar>& r_ab,
                                                       ShortReal r,
                                                       const vec3<ShortReal>& p,
                                                       unsigned int dim,
                                                       const Method)
    {
    if (dim == 3)
        {
        ShortReal Ra = ShortReal(0.5) * shape_a.getCircumsphereDiameter() + r;
        ShortReal Rb = ShortReal(0.5) * shape_b.getCircumsphereDiameter() + r;
        vec3<ShortReal> dr(r_ab);

        bool is_pt_in_sphere_a = dot(p, p) <= Ra * Ra;
        bool is_pt_in_sphere_b = dot(p - dr, p - dr) <= Rb * Rb;

        // point has to be in the intersection of both spheres
        return is_pt_in_sphere_a && is_pt_in_sphere_b;
        }
    else
        {
        hoomd::detail::AABB aabb_a = shape_a.getAABB(vec3<Scalar>(0.0, 0.0, 0.0));
        hoomd::detail::AABB aabb_b = shape_b.getAABB(r_ab);

        // extend AABBs by the excluded volume radius
        vec3<Scalar> lower_a = aabb_a.getLower();
        vec3<Scalar> upper_a = aabb_a.getUpper();
        lower_a.x -= r;
        lower_a.y -= r;
        lower_a.z -= r;
        upper_a.x += r;
        upper_a.y += r;
        upper_a.z += r;

        vec3<Scalar> lower_b = aabb_b.getLower();
        vec3<Scalar> upper_b = aabb_b.getUpper();
        lower_b.x -= r;
        lower_b.y -= r;
        lower_b.z -= r;
        upper_b.x += r;
        upper_b.y += r;
        upper_b.z += r;

        // we already know the AABBs are overlapping, compute their intersection
        vec3<Scalar> intersect_lower, intersect_upper;
        intersect_lower.x = detail::max(lower_a.x, lower_b.x);
        intersect_lower.y = detail::max(lower_a.y, lower_b.y);
        intersect_lower.z = detail::max(lower_a.z, lower_b.z);
        intersect_upper.x = detail::min(upper_a.x, upper_b.x);
        intersect_upper.y = detail::min(upper_a.y, upper_b.y);
        intersect_upper.z = detail::min(upper_a.z, upper_b.z);

        hoomd::detail::AABB aabb_intersect(intersect_lower, intersect_upper);

        return intersect_lower.x <= p.x && p.x <= intersect_upper.x && intersect_lower.y <= p.y
               && p.y <= intersect_upper.y
               && ((dim == 2) || (intersect_lower.z <= p.z && p.z <= intersect_upper.z));
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
