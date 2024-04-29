// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/AABB.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "HPMCMiscFunctions.h"

#ifndef __OBB_H__
#define __OBB_H__

#include <cfloat>
#include <cmath>

#ifndef __HIPCC__
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#if defined(__clang__) || defined(__GNUC__) && __GNUC__ > 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpessimizing-move"
#endif
#include "hoomd/extern/quickhull/QuickHull.hpp"
#if defined(__clang__) || defined(__GNUC__) && __GNUC__ > 8
#pragma GCC diagnostic pop
#endif

#include <random>
#endif

#define DEFAULT_MASK 0xffffffffu

/*! \file OBB.h
    \brief Basic OBB routines
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE __attribute__((always_inline))
#endif

// Check against zero with absolute tolerance
#define CHECK_ZERO(x, abs_tol) ((x < abs_tol && x >= 0) || (-x < abs_tol && x < 0))

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/*! \addtogroup overlap
    @{
*/

/// Update the bounds of an ABB
DEVICE inline void update_bounds(ShortReal& l, ShortReal& u, ShortReal e, ShortReal f)
    {
    if (e < f)
        {
        l += e;
        u += f;
        }
    else
        {
        l += f;
        u += e;
        }
    }

//! Axis aligned bounding box
/*! An OBB represents a bounding volume defined by an axis-aligned bounding box. It is stored as
   plain old data with a lower and upper bound. This is to make the most common operation of OBB
   overlap testing fast.

    Do not access data members directly. OBB uses SSE and AVX optimizations and the internal data
   format changes. It also changes between the CPU and GPU. Instead, use the accessor methods
   getLower(), getUpper() and getPosition().

    Operations are provided as free functions to perform the following operations:

    - merge()
    - overlap()
*/
struct OBB
    {
    vec3<ShortReal> lengths; // half-axes
    vec3<ShortReal> center;
    quat<ShortReal> rotation;
    unsigned int mask;
    unsigned int is_sphere;

    //! Default construct a 0 OBB
    DEVICE OBB() : mask(DEFAULT_MASK), is_sphere(0) { }

    //! Construct an OBB from a sphere
    /*! \param _position Position of the sphere
        \param radius Radius of the sphere

        This constructor internally sets the 'is_sphere' flag to accelerate overlap checks
    */
    DEVICE OBB(const vec3<ShortReal>& _position, ShortReal radius)
        {
        lengths = vec3<ShortReal>(radius, radius, radius);
        center = _position;
        mask = DEFAULT_MASK;
        is_sphere = 1;
        }

    DEVICE explicit OBB(const hoomd::detail::AABB& aabb)
        {
        lengths = ShortReal(0.5)
                  * (vec3<ShortReal>(aabb.getUpper()) - vec3<ShortReal>(aabb.getLower()));
        center = aabb.getPosition();
        mask = DEFAULT_MASK;
        is_sphere = 0;
        }

    //! Construct an OBB from an AABB
    //! Get the OBB's position
    DEVICE vec3<ShortReal> getPosition() const
        {
        return center;
        }

    //! Return true if this OBB is a sphere
    DEVICE bool isSphere() const
        {
        return is_sphere;
        }

#ifndef __HIPCC__
    //! Get list of OBB corners
    std::vector<vec3<ShortReal>> getCorners() const
        {
        std::vector<vec3<ShortReal>> corners(8);

        rotmat3<ShortReal> r(conj(rotation));
        corners[0] = center + r.row0 * lengths.x + r.row1 * lengths.y + r.row2 * lengths.z;
        corners[1] = center - r.row0 * lengths.x + r.row1 * lengths.y + r.row2 * lengths.z;
        corners[2] = center + r.row0 * lengths.x - r.row1 * lengths.y + r.row2 * lengths.z;
        corners[3] = center - r.row0 * lengths.x - r.row1 * lengths.y + r.row2 * lengths.z;
        corners[4] = center + r.row0 * lengths.x + r.row1 * lengths.y - r.row2 * lengths.z;
        corners[5] = center - r.row0 * lengths.x + r.row1 * lengths.y - r.row2 * lengths.z;
        corners[6] = center + r.row0 * lengths.x - r.row1 * lengths.y - r.row2 * lengths.z;
        corners[7] = center - r.row0 * lengths.x - r.row1 * lengths.y - r.row2 * lengths.z;
        return corners;
        }
#endif

    //! Rotate OBB, then translate the given vector
    DEVICE void affineTransform(const quat<ShortReal>& q, const vec3<ShortReal>& v)
        {
        center = rotate(q, center) + v;
        rotation = q * rotation;
        }

    DEVICE ShortReal getVolume(unsigned int dim = 3) const
        {
        if (dim == 3)
            {
            return is_sphere ? ShortReal(4. / 3. * M_PI) * lengths.x * lengths.x * lengths.x
                             : ShortReal(8.0) * lengths.x * lengths.y * lengths.z;
            }
        else
            {
            return is_sphere ? ShortReal(M_PI) * lengths.x * lengths.x
                             : ShortReal(8.0) * lengths.x * lengths.y;
            }
        }

    //! tightly fit an AABB to the OBB
    DEVICE hoomd::detail::AABB getAABB()
        {
        rotmat3<ShortReal> M(rotation);

        vec3<ShortReal> lower_a = -lengths;
        vec3<ShortReal> upper_a = lengths;
        vec3<ShortReal> lower_b = center;
        vec3<ShortReal> upper_b = center;

        update_bounds(lower_b.x, upper_b.x, M.row0.x * lower_a.x, M.row0.x * upper_a.x);
        update_bounds(lower_b.x, upper_b.x, M.row0.y * lower_a.y, M.row0.y * upper_a.y);
        update_bounds(lower_b.x, upper_b.x, M.row0.z * lower_a.z, M.row0.z * upper_a.z);

        update_bounds(lower_b.y, upper_b.y, M.row1.x * lower_a.x, M.row1.x * upper_a.x);
        update_bounds(lower_b.y, upper_b.y, M.row1.y * lower_a.y, M.row1.y * upper_a.y);
        update_bounds(lower_b.y, upper_b.y, M.row1.z * lower_a.z, M.row1.z * upper_a.z);

        update_bounds(lower_b.z, upper_b.z, M.row2.x * lower_a.x, M.row2.x * upper_a.x);
        update_bounds(lower_b.z, upper_b.z, M.row2.y * lower_a.y, M.row2.y * upper_a.y);
        update_bounds(lower_b.z, upper_b.z, M.row2.z * lower_a.z, M.row2.z * upper_a.z);

        return hoomd::detail::AABB(lower_b, upper_b);
        }
    };

// from Christer Ericsen, Real-time collision detection
// https://doi.org/10.1201/b14581
DEVICE inline bool
SqDistPointOBBSmallerThan(const vec3<ShortReal>& p, const OBB& obb, const ShortReal max_sq)
    {
    ShortReal sqDist(0.0);
    const vec3<ShortReal> u = rotate(conj(obb.rotation), p - obb.center);

    // Project vector from box center to p on each axis, getting the distance
    // of p along that axis, and count any excess distance outside box extents

    ShortReal d = dot(u, vec3<ShortReal>(1.0, 0, 0));
    ShortReal excess(0.0);

    if (d < -obb.lengths.x)
        excess = d + obb.lengths.x;
    else if (d > obb.lengths.x)
        excess = d - obb.lengths.x;
    sqDist += excess * excess;

    if (sqDist > max_sq)
        return false;

    d = dot(u, vec3<ShortReal>(0, 1.0, 0));
    excess = ShortReal(0.0);
    if (d < -obb.lengths.y)
        excess = d + obb.lengths.y;
    else if (d > obb.lengths.y)
        excess = d - obb.lengths.y;
    sqDist += excess * excess;

    if (sqDist > max_sq)
        return false;

    d = dot(u, vec3<ShortReal>(0, 0, 1.0));
    excess = ShortReal(0.0);
    if (d < -obb.lengths.z)
        excess = d + obb.lengths.z;
    else if (d > obb.lengths.z)
        excess = d - obb.lengths.z;
    sqDist += excess * excess;

    return sqDist <= max_sq;
    }

//! Check if two OBBs overlap
/*! \param a First OBB
    \param b Second OBB

    \param ignore_mask if true, ignore OBB masks

    \param exact If true, report exact overlaps
    Otherwise, false positives may be reported (which do not hurt
    since this is used in broad phase), which can improve performance

    \returns true when the two OBBs overlap, false otherwise
*/
DEVICE inline bool overlap(const OBB& a, const OBB& b, bool exact = true)
    {
    // exit early if the masks don't match
    if (!(a.mask & b.mask))
        return false;

    // translation vector
    vec3<ShortReal> t = b.center - a.center;

    // if one or both of the OBB are spheres, simplify overlap check
    if (a.isSphere() && b.isSphere())
        {
        ShortReal rsq = dot(t, t);
        ShortReal RaRb = a.lengths.x + b.lengths.x;
        return rsq <= RaRb * RaRb;
        }
    else if (a.isSphere() && !b.isSphere())
        return SqDistPointOBBSmallerThan(a.center, b, a.lengths.x * a.lengths.x);
    else if (!a.isSphere() && b.isSphere())
        return SqDistPointOBBSmallerThan(b.center, a, b.lengths.x * b.lengths.x);

    // check two OBBs

    // rotate B in A's coordinate frame
    rotmat3<ShortReal> r(conj(a.rotation) * b.rotation);

    // rotate translation into A's frame
    t = rotate(conj(a.rotation), t);

    // compute common subexpressions. Add in epsilon term to counteract
    // arithmetic errors when two edges are parallel and their cross prodcut is (near) null
    const ShortReal eps(ShortReal(1e-6)); // can be large, because false positives don't harm

    ShortReal rabs[3][3];
    rabs[0][0] = fabs(r.row0.x) + eps;
    rabs[0][1] = fabs(r.row0.y) + eps;
    rabs[0][2] = fabs(r.row0.z) + eps;

    // test axes L = a0, a1, a2
    ShortReal ra, rb;
    ra = a.lengths.x;
    rb = b.lengths.x * rabs[0][0] + b.lengths.y * rabs[0][1] + b.lengths.z * rabs[0][2];

    if (fabs(t.x) > ra + rb)
        return false;

    rabs[1][0] = fabs(r.row1.x) + eps;
    rabs[1][1] = fabs(r.row1.y) + eps;
    rabs[1][2] = fabs(r.row1.z) + eps;

    ra = a.lengths.y;
    rb = b.lengths.x * rabs[1][0] + b.lengths.y * rabs[1][1] + b.lengths.z * rabs[1][2];
    if (fabs(t.y) > ra + rb)
        return false;

    rabs[2][0] = fabs(r.row2.x) + eps;
    rabs[2][1] = fabs(r.row2.y) + eps;
    rabs[2][2] = fabs(r.row2.z) + eps;

    ra = a.lengths.z;
    rb = b.lengths.x * rabs[2][0] + b.lengths.y * rabs[2][1] + b.lengths.z * rabs[2][2];
    if (fabs(t.z) > ra + rb)
        return false;

    // test axes L = b0, b1, b2
    ra = a.lengths.x * rabs[0][0] + a.lengths.y * rabs[1][0] + a.lengths.z * rabs[2][0];
    rb = b.lengths.x;
    if (fabs(t.x * r.row0.x + t.y * r.row1.x + t.z * r.row2.x) > ra + rb)
        return false;

    ra = a.lengths.x * rabs[0][1] + a.lengths.y * rabs[1][1] + a.lengths.z * rabs[2][1];
    rb = b.lengths.y;
    if (fabs(t.x * r.row0.y + t.y * r.row1.y + t.z * r.row2.y) > ra + rb)
        return false;

    ra = a.lengths.x * rabs[0][2] + a.lengths.y * rabs[1][2] + a.lengths.z * rabs[2][2];
    rb = b.lengths.z;
    if (fabs(t.x * r.row0.z + t.y * r.row1.z + t.z * r.row2.z) > ra + rb)
        return false;

    if (!exact)
        return true; // if exactness is not required, skip some tests

    // test axis L = A0 x B0
    ra = a.lengths.y * rabs[2][0] + a.lengths.z * rabs[1][0];
    rb = b.lengths.y * rabs[0][2] + b.lengths.z * rabs[0][1];
    if (fabs(t.z * r.row1.x - t.y * r.row2.x) > ra + rb)
        return false;

    // test axis L = A0 x B1
    ra = a.lengths.y * rabs[2][1] + a.lengths.z * rabs[1][1];
    rb = b.lengths.x * rabs[0][2] + b.lengths.z * rabs[0][0];
    if (fabs(t.z * r.row1.y - t.y * r.row2.y) > ra + rb)
        return false;

    // test axis L = A0 x B2
    ra = a.lengths.y * rabs[2][2] + a.lengths.z * rabs[1][2];
    rb = b.lengths.x * rabs[0][1] + b.lengths.y * rabs[0][0];
    if (fabs(t.z * r.row1.z - t.y * r.row2.z) > ra + rb)
        return false;

    // test axis L = A1 x B0
    ra = a.lengths.x * rabs[2][0] + a.lengths.z * rabs[0][0];
    rb = b.lengths.y * rabs[1][2] + b.lengths.z * rabs[1][1];
    if (fabs(t.x * r.row2.x - t.z * r.row0.x) > ra + rb)
        return false;

    // test axis L = A1 x B1
    ra = a.lengths.x * rabs[2][1] + a.lengths.z * rabs[0][1];
    rb = b.lengths.x * rabs[1][2] + b.lengths.z * rabs[1][0];
    if (fabs(t.x * r.row2.y - t.z * r.row0.y) > ra + rb)
        return false;

    // test axis L = A1 x B2
    ra = a.lengths.x * rabs[2][2] + a.lengths.z * rabs[0][2];
    rb = b.lengths.x * rabs[1][1] + b.lengths.y * rabs[1][0];
    if (fabs(t.x * r.row2.z - t.z * r.row0.z) > ra + rb)
        return false;

    // test axis L = A2 x B0
    ra = a.lengths.x * rabs[1][0] + a.lengths.y * rabs[0][0];
    rb = b.lengths.y * rabs[2][2] + b.lengths.z * rabs[2][1];
    if (fabs(t.y * r.row0.x - t.x * r.row1.x) > ra + rb)
        return false;

    // test axis L = A2 x B1
    ra = a.lengths.x * rabs[1][1] + a.lengths.y * rabs[0][1];
    rb = b.lengths.x * rabs[2][2] + b.lengths.z * rabs[2][0];
    if (fabs(t.y * r.row0.y - t.x * r.row1.y) > ra + rb)
        return false;

    // test axis L = A2 x B2
    ra = a.lengths.x * rabs[1][2] + a.lengths.y * rabs[0][2];
    rb = b.lengths.x * rabs[2][1] + b.lengths.y * rabs[2][0];
    if (fabs(t.y * r.row0.z - t.x * r.row1.z) > ra + rb)
        return false;

    // no separating axis found, the OBBs must be intersecting
    return true;
    }

// Intersect ray R(t) = p + t*d against OBB a. When intersecting,
// return intersection distance tmin and point q of intersection
// Ericson, Christer, Real-Time Collision Detection (Page 180)
// https://doi.org/10.1201/b14581
DEVICE inline bool IntersectRayOBB(const vec3<ShortReal>& p,
                                   const vec3<ShortReal>& d,
                                   OBB a,
                                   ShortReal& tmin,
                                   vec3<ShortReal>& q,
                                   ShortReal abs_tol)
    {
    tmin = 0.0f;              // set to -FLT_MAX to get first hit on line
    ShortReal tmax = FLT_MAX; // set to max distance ray can travel (for segment)

    // rotate ray in local coordinate system
    quat<ShortReal> a_transp(conj(a.rotation));
    vec3<ShortReal> p_local(rotate(a_transp, p - a.center));
    vec3<ShortReal> d_local(rotate(a_transp, d));

    // For all three slabs
    if (CHECK_ZERO(d_local.x, abs_tol))
        {
        // Ray is parallel to slab. No hit if origin not within slab
        if (p_local.x < -a.lengths.x || p_local.x > a.lengths.x)
            return false;
        }
    else
        {
        // Compute intersection t value of ray with near and far plane of slab
        ShortReal ood = ShortReal(1.0) / d_local.x;
        ShortReal t1 = (-a.lengths.x - p_local.x) * ood;
        ShortReal t2 = (a.lengths.x - p_local.x) * ood;

        // Make t1 be intersection with near plane, t2 with far plane
        if (t1 > t2)
            detail::swap(t1, t2);

        // Compute the intersection of slab intersection intervals
        tmin = detail::max(tmin, t1);
        tmax = detail::min(tmax, t2);

        // Exit with no collision as soon as slab intersection becomes empty
        if (tmin > tmax)
            return false;
        }

    if (CHECK_ZERO(d_local.y, abs_tol))
        {
        // Ray is parallel to slab. No hit if origin not within slab
        if (p_local.y < -a.lengths.y || p_local.y > a.lengths.y)
            return false;
        }
    else
        {
        // Compute intersection t value of ray with near and far plane of slab
        ShortReal ood = ShortReal(1.0) / d_local.y;
        ShortReal t1 = (-a.lengths.y - p_local.y) * ood;
        ShortReal t2 = (a.lengths.y - p_local.y) * ood;

        // Make t1 be intersection with near plane, t2 with far plane
        if (t1 > t2)
            detail::swap(t1, t2);

        // Compute the intersection of slab intersection intervals
        tmin = detail::max(tmin, t1);
        tmax = detail::min(tmax, t2);

        // Exit with no collision as soon as slab intersection becomes empty
        if (tmin > tmax)
            return false;
        }

    if (CHECK_ZERO(d_local.z, abs_tol))
        {
        // Ray is parallel to slab. No hit if origin not within slab
        if (p_local.z < -a.lengths.z || p_local.z > a.lengths.z)
            return false;
        }
    else
        {
        // Compute intersection t value of ray with near and far plane of slab
        ShortReal ood = ShortReal(1.0) / d_local.z;
        ShortReal t1 = (-a.lengths.z - p_local.z) * ood;
        ShortReal t2 = (a.lengths.z - p_local.z) * ood;

        // Make t1 be intersection with near plane, t2 with far plane
        if (t1 > t2)
            detail::swap(t1, t2);

        // Compute the intersection of slab intersection intervals
        tmin = detail::max(tmin, t1);
        tmax = detail::min(tmax, t2);

        // Exit with no collision as soon as slab intersection becomes empty
        if (tmin > tmax)
            return false;
        }

    // Ray intersects all 3 slabs. Return point (q) and intersection t value (tmin) in space frame
    q = rotate(a.rotation, p_local + d_local * tmin);

    return true;
    }

#ifndef __HIPCC__
// Ericson, Christer (2013-05-02). Real-Time Collision Detection (Page 111). Taylor and Francis CRC
// https://doi.org/10.1201/b14581

// Compute the center point, ’c’, and axis orientation, u[0] and u[1], of
// the minimum area rectangle in the xy plane containing the points pt[].
inline double MinAreaRect(vec2<double> pt[], int numPts, vec2<double>& c, vec2<double> u[2])
    {
    double minArea = DBL_MAX;

    // initialize to some default unit vectors
    u[0] = vec2<double>(1, 0);
    u[1] = vec2<double>(0, 1);

    // Loop through all edges; j trails i by 1, modulo numPts
    for (int i = 0, j = numPts - 1; i < numPts; j = i, i++)
        {
        // Get current edge e0 (e0x,e0y), normalized
        vec2<double> e0 = pt[i] - pt[j];

        const double eps_abs(1e-12); // if edge is too short, do not consider
        if (dot(e0, e0) < eps_abs)
            continue;
        e0 = e0 / sqrt(dot(e0, e0));

        // Get an axis e1 orthogonal to edge e0
        vec2<double> e1 = vec2<double>(-e0.y, e0.x); // = Perp2D(e0)

        // Loop through all points to get maximum extents
        double min0 = 0.0, min1 = 0.0, max0 = 0.0, max1 = 0.0;

        for (int k = 0; k < numPts; k++)
            {
            // Project points onto axes e0 and e1 and keep track
            // of minimum and maximum values along both axes
            vec2<double> d = pt[k] - pt[j];
            double dotp = dot(d, e0);
            if (dotp < min0)
                min0 = dotp;
            if (dotp > max0)
                max0 = dotp;
            dotp = dot(d, e1);
            if (dotp < min1)
                min1 = dotp;
            if (dotp > max1)
                max1 = dotp;
            }
        double area = (max0 - min0) * (max1 - min1);

        // If best so far, remember area, center, and axes
        if (area < minArea)
            {
            minArea = area;
            c = pt[j] + 0.5 * ((min0 + max0) * e0 + (min1 + max1) * e1);
            u[0] = e0;
            u[1] = e1;
            }
        }
    return minArea;
    }

// https://stackoverflow.com/questions/33532860/merge-two-spheres-to-get-a-new-one
inline ShortReal
merge_two_spheres(vec3<ShortReal>& c, ShortReal r, vec3<ShortReal> p, ShortReal r_pt)
    {
    vec3<ShortReal> d = p - c;
    ShortReal dist = sqrt(dot(d, d));

    if (dist + r <= r_pt)
        {
        c = p;
        return r_pt;
        }
    else if (dist + r_pt <= r)
        {
        return r;
        }

    ShortReal r_new = ShortReal(0.5) * (r + r_pt + dist);

    if (dist > 0)
        c += d * (r_new - r) / dist;

    return r_new;
    }

inline ShortReal eigen_sphere(const std::vector<vec3<ShortReal>>& verts,
                              vec3<ShortReal>& center,
                              const std::vector<ShortReal>& vertex_radii)
    {
    // compute covariance matrix
    Eigen::MatrixXd m(3, 3);
    m(0, 0) = m(0, 1) = m(0, 2) = m(1, 0) = m(1, 1) = m(1, 2) = m(2, 0) = m(2, 1) = m(2, 2) = 0.0;

    vec3<ShortReal> mean(0, 0, 0);
    for (unsigned int i = 0; i < verts.size(); ++i)
        mean += verts[i] / (ShortReal)verts.size();

    for (unsigned int i = 0; i < verts.size(); i++)
        {
        m(0, 0) += (verts[i].x - mean.x) * (verts[i].x - mean.x);
        m(0, 1) += (verts[i].x - mean.x) * (verts[i].y - mean.y);
        m(0, 2) += (verts[i].x - mean.x) * (verts[i].z - mean.z);
        m(1, 0) += (verts[i].y - mean.y) * (verts[i].x - mean.x);
        m(1, 1) += (verts[i].y - mean.y) * (verts[i].y - mean.y);
        m(1, 2) += (verts[i].y - mean.y) * (verts[i].z - mean.z);
        m(2, 0) += (verts[i].z - mean.z) * (verts[i].x - mean.x);
        m(2, 1) += (verts[i].z - mean.z) * (verts[i].y - mean.y);
        m(2, 2) += (verts[i].z - mean.z) * (verts[i].z - mean.z);
        }

    // compute normalized eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(m);

    rotmat3<ShortReal> r;

    Eigen::VectorXd eigen_val;
    if (es.info() != Eigen::Success)
        {
        // numerical issue, set r to identity matrix
        r.row0 = vec3<ShortReal>(1, 0, 0);
        r.row1 = vec3<ShortReal>(0, 1, 0);
        r.row2 = vec3<ShortReal>(0, 0, 1);
        eigen_val(0) = eigen_val(1) = eigen_val(2) = 0;
        }
    else
        {
        // get the orthonormal basis
        Eigen::MatrixXd eigenvec = es.eigenvectors();

        r.row0 = vec3<ShortReal>(ShortReal(eigenvec(0, 0)),
                                 ShortReal(eigenvec(0, 1)),
                                 ShortReal(eigenvec(0, 2)));
        r.row1 = vec3<ShortReal>(ShortReal(eigenvec(1, 0)),
                                 ShortReal(eigenvec(1, 1)),
                                 ShortReal(eigenvec(1, 2)));
        r.row2 = vec3<ShortReal>(ShortReal(eigenvec(2, 0)),
                                 ShortReal(eigenvec(2, 1)),
                                 ShortReal(eigenvec(2, 2)));
        eigen_val = es.eigenvalues();
        }

    // maximum eigenvalue
    vec3<ShortReal> max_evec(r.row0.x, r.row1.x, r.row2.x);
    ShortReal max_eval = ShortReal(eigen_val(0));
    if (eigen_val(1) > max_eval)
        {
        max_evec = vec3<ShortReal>(r.row0.y, r.row1.y, r.row2.y);
        max_eval = ShortReal(eigen_val(1));
        }
    if (eigen_val(2) > max_eval)
        {
        max_evec = vec3<ShortReal>(r.row0.z, r.row1.z, r.row2.z);
        max_eval = ShortReal(eigen_val(2));
        }

    max_evec /= (ShortReal)sqrt(dot(max_evec, max_evec));

    vec3<ShortReal> min_pt;
    vec3<ShortReal> max_pt;
    ShortReal min_extent = FLT_MAX;
    ShortReal max_extent = -FLT_MAX;
    ShortReal max_vertex_radius = 0.0;
    ShortReal min_vertex_radius = 0.0;
    for (unsigned int i = 0; i < verts.size(); ++i)
        {
        if (dot(max_evec, verts[i]) - vertex_radii[i] < min_extent)
            {
            min_extent = dot(max_evec, verts[i]) - vertex_radii[i];
            min_pt = verts[i];
            min_vertex_radius = vertex_radii[i];
            }
        if (dot(max_evec, verts[i]) + vertex_radii[i] > max_extent)
            {
            max_extent = dot(max_evec, verts[i]) + vertex_radii[i];
            max_pt = verts[i];
            max_vertex_radius = vertex_radii[i];
            }
        }

    center = min_pt;
    return merge_two_spheres(center, min_vertex_radius, max_pt, max_vertex_radius);
    }

inline ShortReal ritter_eigen_sphere(const std::vector<vec3<ShortReal>>& verts,
                                     vec3<ShortReal>& c,
                                     const std::vector<ShortReal>& vertex_radii)
    {
    ShortReal r = eigen_sphere(verts, c, vertex_radii);
    for (unsigned int i = 0; i < verts.size(); ++i)
        {
        assert(i < vertex_radii.size());
        r = merge_two_spheres(c, r, verts[i], vertex_radii[i]);
        }
    return r;
    }

inline ShortReal ritter_iterative(std::vector<vec3<ShortReal>> verts,
                                  vec3<ShortReal>& c,
                                  std::vector<ShortReal> vertex_radii)
    {
    const unsigned int MAX_IT = 16;

    ShortReal r = ritter_eigen_sphere(verts, c, vertex_radii);

    vec3<ShortReal> c2 = c;
    ShortReal r2 = r;

    std::mt19937 g(123);

    for (unsigned int k = 0; k < MAX_IT; ++k)
        {
        r2 *= ShortReal(0.95);

        for (unsigned int i = 0; i < verts.size(); ++i)
            {
            if (i < verts.size() - 1)
                {
                unsigned int j;
                std::uniform_int_distribution<> dis(i + 1, (unsigned int)(verts.size() - 1));
                j = dis(g);
                std::swap(verts[i], verts[j]);
                std::swap(vertex_radii[i], vertex_radii[j]);
                }
            r2 = merge_two_spheres(c2, r2, verts[i], vertex_radii[i]);
            }

        if (r2 < r)
            {
            r = r2;
            c = c2;
            }
        }
    return r;
    }

DEVICE inline OBB compute_obb(const std::vector<vec3<ShortReal>>& pts,
                              const std::vector<ShortReal>& vertex_radii,
                              bool make_sphere)
    {
    OBB res;

    if (!make_sphere)
        {
        // compute mean
        vec3<ShortReal> mean = vec3<ShortReal>(0, 0, 0);

        unsigned int n = (unsigned int)pts.size();
        for (unsigned int i = 0; i < n; ++i)
            {
            mean += pts[i] / (ShortReal)n;
            }

        // compute covariance matrix
        Eigen::MatrixXd m(3, 3);
        m(0, 0) = m(0, 1) = m(0, 2) = m(1, 0) = m(1, 1) = m(1, 2) = m(2, 0) = m(2, 1) = m(2, 2)
            = 0.0;

        std::vector<vec3<double>> hull_pts;

        if (pts.size() >= 3)
            {
            // compute convex hull
            typedef quickhull::Vector3<ShortReal> vec;

            quickhull::QuickHull<ShortReal> qh;
            std::vector<vec> qh_pts;
            for (auto it = pts.begin(); it != pts.end(); ++it)
                qh_pts.push_back(vec(it->x, it->y, it->z));
            auto hull = qh.getConvexHull(qh_pts, true, false);
            auto indexBuffer = hull.getIndexBuffer();
            auto vertexBuffer = hull.getVertexBuffer();

            ShortReal hull_area(0.0);
            vec hull_centroid(0.0, 0.0, 0.0);

            for (unsigned int i = 0; i < vertexBuffer.size(); ++i)
                hull_pts.push_back(
                    vec3<double>(vertexBuffer[i].x, vertexBuffer[i].y, vertexBuffer[i].z));

            for (unsigned int i = 0; i < indexBuffer.size(); i += 3)
                {
                // triangle vertices
                vec p = vertexBuffer[indexBuffer[i]];
                vec q = vertexBuffer[indexBuffer[i + 1]];
                vec r = vertexBuffer[indexBuffer[i + 2]];

                vec centroid = ShortReal(1. / 3.) * (p + q + r);
                vec cross = (q - p).crossProduct(r - p);
                ShortReal area = ShortReal(0.5) * sqrt(cross.dotProduct(cross));
                hull_area += area;
                hull_centroid += area * centroid;

                ShortReal fac = area / ShortReal(12.0);
                m(0, 0)
                    += fac * (9.0 * centroid.x * centroid.x + p.x * p.x + q.x * q.x + r.x * r.x);
                m(0, 1)
                    += fac * (9.0 * centroid.x * centroid.y + p.x * p.y + q.x * q.y + r.x * r.y);
                m(0, 2)
                    += fac * (9.0 * centroid.x * centroid.z + p.x * p.z + q.x * q.z + r.x * r.z);
                m(1, 0)
                    += fac * (9.0 * centroid.y * centroid.x + p.y * p.x + q.y * q.x + r.y * r.x);
                m(1, 1)
                    += fac * (9.0 * centroid.y * centroid.y + p.y * p.y + q.y * q.y + r.y * r.y);
                m(1, 2)
                    += fac * (9.0 * centroid.y * centroid.z + p.y * p.z + q.y * q.z + r.y * r.z);
                m(2, 0)
                    += fac * (9.0 * centroid.z * centroid.x + p.z * p.x + q.z * q.x + r.z * r.x);
                m(2, 1)
                    += fac * (9.0 * centroid.z * centroid.y + p.z * p.y + q.z * q.y + r.z * r.y);
                m(2, 2)
                    += fac * (9.0 * centroid.z * centroid.z + p.z * p.z + q.z * q.z + r.z * r.z);
                }

            hull_centroid /= hull_area;
            m(0, 0) = m(0, 0) / hull_area - hull_centroid.x * hull_centroid.x;
            m(0, 1) = m(0, 1) / hull_area - hull_centroid.x * hull_centroid.y;
            m(0, 2) = m(0, 2) / hull_area - hull_centroid.x * hull_centroid.z;
            m(1, 0) = m(1, 0) / hull_area - hull_centroid.y * hull_centroid.x;
            m(1, 1) = m(1, 1) / hull_area - hull_centroid.y * hull_centroid.y;
            m(1, 2) = m(1, 2) / hull_area - hull_centroid.y * hull_centroid.z;
            m(2, 0) = m(2, 0) / hull_area - hull_centroid.z * hull_centroid.x;
            m(2, 1) = m(2, 1) / hull_area - hull_centroid.z * hull_centroid.y;
            m(2, 2) = m(2, 2) / hull_area - hull_centroid.z * hull_centroid.z;
            }
        else
            {
            // degenerate case
            for (unsigned int i = 0; i < n; ++i)
                {
                vec3<ShortReal> dr = pts[i] - mean;

                m(0, 0) += dr.x * dr.x / (double)n;
                m(1, 0) += dr.y * dr.x / (double)n;
                m(2, 0) += dr.z * dr.x / (double)n;

                m(0, 1) += dr.x * dr.y / (double)n;
                m(1, 1) += dr.y * dr.y / (double)n;
                m(2, 1) += dr.z * dr.y / (double)n;

                m(0, 2) += dr.x * dr.z / (double)n;
                m(1, 2) += dr.y * dr.z / (double)n;
                m(2, 2) += dr.z * dr.z / (double)n;
                }
            }

        // compute normalized eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;

        es.compute(m);

        rotmat3<ShortReal> r;

        if (es.info() != Eigen::Success)
            {
            // numerical issue, set r to identity matrix
            r.row0 = vec3<ShortReal>(1, 0, 0);
            r.row1 = vec3<ShortReal>(0, 1, 0);
            r.row2 = vec3<ShortReal>(0, 0, 1);
            }
        else
            {
            // get the orthonormal basis
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(es.eigenvectors());
            Eigen::MatrixXd eigenvec_ortho = qr.householderQ();

            r.row0 = vec3<ShortReal>(ShortReal(eigenvec_ortho(0, 0)),
                                     ShortReal(eigenvec_ortho(0, 1)),
                                     ShortReal(eigenvec_ortho(0, 2)));
            r.row1 = vec3<ShortReal>(ShortReal(eigenvec_ortho(1, 0)),
                                     ShortReal(eigenvec_ortho(1, 1)),
                                     ShortReal(eigenvec_ortho(1, 2)));
            r.row2 = vec3<ShortReal>(ShortReal(eigenvec_ortho(2, 0)),
                                     ShortReal(eigenvec_ortho(2, 1)),
                                     ShortReal(eigenvec_ortho(2, 2)));
            }

        if (pts.size() >= 3)
            {
            bool done = false;
            vec3<double> cur_axis[3];
            cur_axis[0] = vec3<double>(r.row0.x, r.row1.x, r.row2.x);
            cur_axis[1] = vec3<double>(r.row0.y, r.row1.y, r.row2.y);
            cur_axis[2] = vec3<double>(r.row0.z, r.row1.z, r.row2.z);

            double min_V = DBL_MAX;
            unsigned int min_axis = 0;
            vec2<double> min_axes_2d[2];

            // iteratively improve OBB
            while (!done)
                {
                bool updated_axes = false;

                // test if a projection normal to any axis reduces the volume of the bounding box
                for (unsigned int test_axis = 0; test_axis < 3; ++test_axis)
                    {
                    // project normal to test_axis
                    std::vector<vec2<double>> proj_2d(hull_pts.size());
                    for (unsigned int i = 0; i < hull_pts.size(); ++i)
                        {
                        unsigned k = 0;
                        for (unsigned int j = 0; j < 3; j++)
                            {
                            if (j != test_axis)
                                {
                                if (k++ == 0)
                                    proj_2d[i].x = dot(cur_axis[j], hull_pts[i]);
                                else
                                    proj_2d[i].y = dot(cur_axis[j], hull_pts[i]);
                                }
                            }
                        }

                    vec2<double> new_axes_2d[2];
                    vec2<double> c;
                    double area = MinAreaRect(&proj_2d.front(),
                                              (unsigned int)hull_pts.size(),
                                              c,
                                              new_axes_2d);

                    // find extent along test_axis
                    double proj_min = DBL_MAX;
                    double proj_max = -DBL_MAX;
                    for (unsigned int i = 0; i < hull_pts.size(); ++i)
                        {
                        double proj = dot(hull_pts[i], cur_axis[test_axis]);

                        if (proj > proj_max)
                            proj_max = proj;
                        if (proj < proj_min)
                            proj_min = proj;
                        }
                    double extent = proj_max - proj_min;

                    // bounding box volume
                    double V = extent * area;
                    double eps_rel(1e-6); // convergence criterion
                    if (V < min_V && (min_V - V) > eps_rel * min_V)
                        {
                        min_V = V;
                        min_axes_2d[0] = new_axes_2d[0];
                        min_axes_2d[1] = new_axes_2d[1];
                        min_axis = test_axis;
                        updated_axes = true;
                        }
                    } // end loop over test axis

                if (updated_axes)
                    {
                    vec3<double> new_axis[3];

                    // test axis stays the same
                    new_axis[min_axis] = cur_axis[min_axis];

                    // rotate axes
                    for (unsigned int j = 0; j < 3; j++)
                        {
                        if (j != min_axis)
                            {
                            for (unsigned int l = j + 1; l < 3; l++)
                                if (l != min_axis)
                                    {
                                    new_axis[l] = min_axes_2d[0].x * cur_axis[j]
                                                  + min_axes_2d[0].y * cur_axis[l];
                                    new_axis[j] = min_axes_2d[1].x * cur_axis[j]
                                                  + min_axes_2d[1].y * cur_axis[l];
                                    }
                            }
                        }

                    // update axes
                    for (unsigned int j = 0; j < 3; j++)
                        cur_axis[j] = new_axis[j];
                    }
                else
                    {
                    // local minimum reached
                    done = true;
                    }
                }

            // update rotation matrix
            r.row0 = cur_axis[0];
            r.row1 = cur_axis[1];
            r.row2 = cur_axis[2];
            r = transpose(r);
            }

        // final axes
        vec3<ShortReal> axis[3];
        axis[0] = vec3<ShortReal>(r.row0.x, r.row1.x, r.row2.x);
        axis[1] = vec3<ShortReal>(r.row0.y, r.row1.y, r.row2.y);
        axis[2] = vec3<ShortReal>(r.row0.z, r.row1.z, r.row2.z);

        vec3<ShortReal> proj_min = vec3<ShortReal>(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3<ShortReal> proj_max = vec3<ShortReal>(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        // project points onto axes
        for (unsigned int i = 0; i < n; ++i)
            {
            vec3<ShortReal> proj;
            proj.x = dot(pts[i] - mean, axis[0]);
            proj.y = dot(pts[i] - mean, axis[1]);
            proj.z = dot(pts[i] - mean, axis[2]);

            if (proj.x + vertex_radii[i] > proj_max.x)
                proj_max.x = proj.x + vertex_radii[i];
            if (proj.y + vertex_radii[i] > proj_max.y)
                proj_max.y = proj.y + vertex_radii[i];
            if (proj.z + vertex_radii[i] > proj_max.z)
                proj_max.z = proj.z + vertex_radii[i];

            if (proj.x - vertex_radii[i] < proj_min.x)
                proj_min.x = proj.x - vertex_radii[i];
            if (proj.y - vertex_radii[i] < proj_min.y)
                proj_min.y = proj.y - vertex_radii[i];
            if (proj.z - vertex_radii[i] < proj_min.z)
                proj_min.z = proj.z - vertex_radii[i];
            }

        res.center = mean;

        res.center += ShortReal(0.5) * (proj_max.x + proj_min.x) * axis[0];
        res.center += ShortReal(0.5) * (proj_max.y + proj_min.y) * axis[1];
        res.center += ShortReal(0.5) * (proj_max.z + proj_min.z) * axis[2];

        res.lengths = ShortReal(0.5) * (proj_max - proj_min);

        // sort by decreasing length, so split can occur along longest axis
        if (res.lengths.x < res.lengths.y)
            {
            std::swap(r.row0.x, r.row0.y);
            std::swap(r.row1.x, r.row1.y);
            std::swap(r.row2.x, r.row2.y);
            std::swap(res.lengths.x, res.lengths.y);
            }

        if (res.lengths.y < res.lengths.z)
            {
            std::swap(r.row0.y, r.row0.z);
            std::swap(r.row1.y, r.row1.z);
            std::swap(r.row2.y, r.row2.z);
            std::swap(res.lengths.y, res.lengths.z);
            }

        if (res.lengths.x < res.lengths.y)
            {
            std::swap(r.row0.x, r.row0.y);
            std::swap(r.row1.x, r.row1.y);
            std::swap(r.row2.x, r.row2.y);
            std::swap(res.lengths.x, res.lengths.y);
            }

        // make sure coordinate system is proper
        if (r.det() < ShortReal(0.0))
            {
            // swap column two and three
            std::swap(r.row0.y, r.row0.z);
            std::swap(r.row1.y, r.row1.z);
            std::swap(r.row2.y, r.row2.z);
            std::swap(res.lengths.y, res.lengths.z);
            }

        res.rotation = quat<ShortReal>(r);
        }
    else
        {
        res.lengths.x = res.lengths.y = res.lengths.z
            = ritter_iterative(pts, res.center, vertex_radii);
        res.rotation = quat<ShortReal>();
        res.is_sphere = 1;
        }

    return res;
    }
#endif
    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef DEFAULT_MASK
#endif //__OBB_H__
