// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/AABB.h"

#include <algorithm>
#include <cfloat>

#ifndef __OBB_H__
#define __OBB_H__

#ifndef NVCC
#include "hoomd/extern/Eigen/Dense"
#include "hoomd/extern/Eigen/Eigenvalues"
#endif

/*! \file OBB.h
    \brief Basic OBB routines
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE __attribute__((always_inline))
#endif

namespace hpmc
{

namespace detail
{

/*! \addtogroup overlap
    @{
*/

//! Axis aligned bounding box
/*! An OBB represents a bounding volume defined by an axis-aligned bounding box. It is stored as plain old data
    with a lower and upper bound. This is to make the most common operation of OBB overlap testing fast.

    Do not access data members directly. OBB uses SSE and AVX optimizations and the internal data format changes.
    It also changes between the CPU and GPU. Instead, use the accessor methods getLower(), getUpper() and getPosition().

    Operations are provided as free functions to perform the following operations:

    - merge()
    - overlap()
*/
struct OBB
    {
    vec3<OverlapReal> lengths; // half-axes
    vec3<OverlapReal> center;
    rotmat3<OverlapReal> rotation;

    //! Default construct a 0 OBB
    DEVICE OBB() {}

    //! Construct an OBB from a sphere
    /*! \param _position Position of the sphere
        \param radius Radius of the sphere
    */
    DEVICE OBB(const vec3<OverlapReal>& _position, OverlapReal radius)
        {
        lengths = vec3<OverlapReal>(radius,radius,radius);
        center = _position;
        }

    DEVICE OBB(const detail::AABB& aabb)
        {
        lengths = OverlapReal(0.5)*(vec3<OverlapReal>(aabb.getUpper())-vec3<OverlapReal>(aabb.getLower()));
        center = aabb.getPosition();
        }

    //! Construct an OBB from an AABB
    //! Get the OBB's position
    DEVICE vec3<OverlapReal> getPosition() const
        {
        return center;
        }

    //! Get list of OBB corners
    std::vector<vec3<OverlapReal> > getCorners() const
        {
        std::vector< vec3<OverlapReal> > corners(8);

        rotmat3<OverlapReal> r(transpose(rotation));
        corners[0] = center + r.row0*lengths.x + r.row1*lengths.y + r.row2*lengths.z;
        corners[1] = center - r.row0*lengths.x + r.row1*lengths.y + r.row2*lengths.z;
        corners[2] = center + r.row0*lengths.x - r.row1*lengths.y + r.row2*lengths.z;
        corners[3] = center - r.row0*lengths.x - r.row1*lengths.y + r.row2*lengths.z;
        corners[4] = center + r.row0*lengths.x + r.row1*lengths.y - r.row2*lengths.z;
        corners[5] = center - r.row0*lengths.x + r.row1*lengths.y - r.row2*lengths.z;
        corners[6] = center + r.row0*lengths.x - r.row1*lengths.y - r.row2*lengths.z;
        corners[7] = center - r.row0*lengths.x - r.row1*lengths.y - r.row2*lengths.z;
        return corners;
        }

    //! Rotate OBB, then translate the given vector
    DEVICE void affineTransform(const quat<OverlapReal>& q, const vec3<OverlapReal>& v)
        {
        center = ::rotate(q,center) + v;
        rotation = rotmat3<OverlapReal>(q) * rotation;
        }

    } __attribute__((aligned(32)));

//! Check if two OBBs overlap
/*! \param a First OBB
    \param b Second OBB

    \param exact If true, report exact overlaps
    Otherwise, false positives may be reported (which do not hurt
    since this is used in broad phase), which can improve performance

    \returns true when the two OBBs overlap, false otherwise
*/
DEVICE inline bool overlap(const OBB& a, const OBB& b, bool exact=false)
    {
    // rotate B in A's coordinate frame
    rotmat3<OverlapReal> r = transpose(a.rotation) * b.rotation;

    // translation vector
    vec3<OverlapReal> t = b.center - a.center;

    // rotate translation into A's frame
    t = transpose(a.rotation)*t;

    // compute common subexpressions. Add in epsilon term to counteract
    // arithmetic errors when two edges are parallel and their cross prodcut is (near) null
    const OverlapReal eps(1e-3); // can be large, because false positives don't harm

    OverlapReal rabs[3][3];
    rabs[0][0] = fabs(r.row0.x) + eps;
    rabs[0][1] = fabs(r.row0.y) + eps;
    rabs[0][2] = fabs(r.row0.z) + eps;

    // test axes L = a0, a1, a2
    OverlapReal ra, rb;
    ra = a.lengths.x;
    rb = b.lengths.x * rabs[0][0] + b.lengths.y * rabs[0][1] + b.lengths.z*rabs[0][2];
    if (fabs(t.x) > ra + rb) return false;

    rabs[1][0] = fabs(r.row1.x) + eps;
    rabs[1][1] = fabs(r.row1.y) + eps;
    rabs[1][2] = fabs(r.row1.z) + eps;

    ra = a.lengths.y;
    rb = b.lengths.x * rabs[1][0] + b.lengths.y * rabs[1][1] + b.lengths.z*rabs[1][2];
    if (fabs(t.y) > ra + rb) return false;

    rabs[2][0] = fabs(r.row2.x) + eps;
    rabs[2][1] = fabs(r.row2.y) + eps;
    rabs[2][2] = fabs(r.row2.z) + eps;

    ra = a.lengths.z;
    rb = b.lengths.x * rabs[2][0] + b.lengths.y * rabs[2][1] + b.lengths.z*rabs[2][2];
    if (fabs(t.z) > ra + rb) return false;

    // test axes L = b0, b1, b2
    ra = a.lengths.x * rabs[0][0] + a.lengths.y * rabs[1][0] + a.lengths.z*rabs[2][0];
    rb = b.lengths.x;
    if (fabs(t.x*r.row0.x+t.y*r.row1.x+t.z*r.row2.x) > ra + rb) return false;

    ra = a.lengths.x * rabs[0][1] + a.lengths.y * rabs[1][1] + a.lengths.z*rabs[2][1];
    rb = b.lengths.y;
    if (fabs(t.x*r.row0.y+t.y*r.row1.y+t.z*r.row2.y) > ra + rb) return false;

    ra = a.lengths.x * rabs[0][2] + a.lengths.y * rabs[1][2] + a.lengths.z*rabs[2][2];
    rb = b.lengths.z;
    if (fabs(t.x*r.row0.z+t.y*r.row1.z+t.z*r.row2.z) > ra + rb) return false;

    if (!exact) return true; // if exactness is not required, skip some tests

    // test axis L = A0 x B0
    ra = a.lengths.y * rabs[2][0] + a.lengths.z*rabs[1][0];
    rb = b.lengths.y * rabs[0][2] + b.lengths.z*rabs[0][1];
    if (fabs(t.z*r.row1.x-t.y*r.row2.x) > ra + rb) return false;

    // test axis L = A0 x B1
    ra = a.lengths.y * rabs[2][1] + a.lengths.z*rabs[1][1];
    rb = b.lengths.x * rabs[0][2] + b.lengths.z*rabs[0][0];
    if (fabs(t.z*r.row1.y-t.y*r.row2.y) > ra + rb) return false;

    // test axis L = A0 x B2
    ra = a.lengths.y * rabs[2][2] + a.lengths.z*rabs[1][2];
    rb = b.lengths.x * rabs[0][1] + b.lengths.y*rabs[0][0];
    if (fabs(t.z*r.row1.z-t.y*r.row2.z) > ra + rb) return false;

    // test axis L = A1 x B0
    ra = a.lengths.x * rabs[2][0] + a.lengths.z*rabs[0][0];
    rb = b.lengths.y * rabs[1][2] + b.lengths.z*rabs[1][1];
    if (fabs(t.x*r.row2.x - t.z*r.row0.x) > ra + rb) return false;

    // test axis L = A1 x B1
    ra = a.lengths.x * rabs[2][1] + a.lengths.z * rabs[0][1];
    rb = b.lengths.x * rabs[1][2] + b.lengths.z * rabs[1][0];
    if (fabs(t.x*r.row2.y - t.z*r.row0.y) > ra + rb) return false;

    // test axis L = A1 x B2
    ra = a.lengths.x * rabs[2][2] + a.lengths.z * rabs[0][2];
    rb = b.lengths.x * rabs[1][1] + b.lengths.y * rabs[1][0];
    if (fabs(t.x*r.row2.z - t.z * r.row0.z) > ra + rb) return false;

    // test axis L = A2 x B0
    ra = a.lengths.x * rabs[1][0] + a.lengths.y * rabs[0][0];
    rb = b.lengths.y * rabs[2][2] + b.lengths.z * rabs[2][1];
    if (fabs(t.y * r.row0.x - t.x * r.row1.x) > ra + rb) return false;

    // test axis L = A2 x B1
    ra = a.lengths.x * rabs[1][1] + a.lengths.y * rabs[0][1];
    rb = b.lengths.x * rabs[2][2] + b.lengths.z * rabs[2][0];
    if (fabs(t.y * r.row0.y - t.x * r.row1.y) > ra + rb) return false;

    // test axis L = A2 x B2
    ra = a.lengths.x * rabs[1][2] + a.lengths.y * rabs[0][2];
    rb = b.lengths.x * rabs[2][1] + b.lengths.y * rabs[2][0];
    if (fabs(t.y*r.row0.z - t.x * r.row1.z) > ra + rb) return false;

    // no separating axis found, the OBBs must be intersecting
    return true;
    }

#ifndef NVCC
DEVICE inline OBB compute_obb(const std::vector< vec3<OverlapReal> >& pts, OverlapReal vertex_radius)
    {
    // compute mean
    OBB res;
    vec3<OverlapReal> mean = vec3<OverlapReal>(0,0,0);

    unsigned int n = pts.size();
    for (unsigned int i = 0; i < n; ++i)
        {
        mean += pts[i]/(OverlapReal)n;
        }

    // compute covariance matrix
    Eigen::MatrixXd m(3,3);

    m(0,0) = m(0,1) = m(0,2) = m(1,0) = m(1,1) = m(1,2) = m(2,0) = m(2,1) = m(2,2) = 0.0;

    for (unsigned int i = 0; i < n; ++i)
        {
        vec3<OverlapReal> dr = pts[i] - mean;

        m(0,0) += dr.x * dr.x/OverlapReal(n);
        m(1,0) += dr.y * dr.x/OverlapReal(n);
        m(2,0) += dr.z * dr.x/OverlapReal(n);

        m(0,1) += dr.x * dr.y/OverlapReal(n);
        m(1,1) += dr.y * dr.y/OverlapReal(n);
        m(2,1) += dr.z * dr.y/OverlapReal(n);

        m(0,2) += dr.x * dr.z/OverlapReal(n);
        m(1,2) += dr.y * dr.z/OverlapReal(n);
        m(2,2) += dr.z * dr.z/OverlapReal(n);
        }

    // compute normalized eigenvectors
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    es.compute(m);
    Eigen::MatrixXcd eigen_vec = es.eigenvectors();
    Eigen::VectorXcd eigen_val = es.eigenvalues();

    rotmat3<OverlapReal> r;

    r.row0 = vec3<OverlapReal>(eigen_vec(0,0).real(),eigen_vec(0,1).real(),eigen_vec(0,2).real());
    r.row1 = vec3<OverlapReal>(eigen_vec(1,0).real(),eigen_vec(1,1).real(),eigen_vec(1,2).real());
    r.row2 = vec3<OverlapReal>(eigen_vec(2,0).real(),eigen_vec(2,1).real(),eigen_vec(2,2).real());

    // sort by descending eigenvalue, so split can occur along axis with largest covariance
    if (eigen_val(0).real() < eigen_val(1).real())
        {
        std::swap(r.row0.x,r.row0.y);
        std::swap(r.row1.x,r.row1.y);
        std::swap(r.row2.x,r.row2.y);
        std::swap(eigen_val(1),eigen_val(0));
        }

    if (eigen_val(1).real() < eigen_val(2).real())
        {
        std::swap(r.row0.y,r.row0.z);
        std::swap(r.row1.y,r.row1.z);
        std::swap(r.row2.y,r.row2.z);
        std::swap(eigen_val(1),eigen_val(2));
        }

    if (eigen_val(0).real() < eigen_val(1).real())
        {
        std::swap(r.row0.x,r.row0.y);
        std::swap(r.row1.x,r.row1.y);
        std::swap(r.row2.x,r.row2.y);
        std::swap(eigen_val(1),eigen_val(0));
        }

    vec3<OverlapReal> axis[3];
    axis[0] = vec3<OverlapReal>(r.row0.x, r.row1.x, r.row2.x);
    axis[1] = vec3<OverlapReal>(r.row0.y, r.row1.y, r.row2.y);
    axis[2] = vec3<OverlapReal>(r.row0.z, r.row1.z, r.row2.z);

    res.rotation = r;

    vec3<OverlapReal> proj_min = vec3<OverlapReal>(FLT_MAX,FLT_MAX,FLT_MAX);
    vec3<OverlapReal> proj_max = vec3<OverlapReal>(-FLT_MAX,-FLT_MAX,-FLT_MAX);

    // project points onto axes
    for (unsigned int i = 0; i < n; ++i)
        {
        vec3<OverlapReal> proj;
        proj.x = dot(pts[i]-mean, axis[0]);
        proj.y = dot(pts[i]-mean, axis[1]);
        proj.z = dot(pts[i]-mean, axis[2]);

        if (proj.x > proj_max.x) proj_max.x = proj.x;
        if (proj.y > proj_max.y) proj_max.y = proj.y;
        if (proj.z > proj_max.z) proj_max.z = proj.z;

        if (proj.x < proj_min.x) proj_min.x = proj.x;
        if (proj.y < proj_min.y) proj_min.y = proj.y;
        if (proj.z < proj_min.z) proj_min.z = proj.z;
        }

    res.center = mean;
    res.center += OverlapReal(0.5)*(proj_max.x + proj_min.x)*axis[0];
    res.center += OverlapReal(0.5)*(proj_max.y + proj_min.y)*axis[1];
    res.center += OverlapReal(0.5)*(proj_max.z + proj_min.z)*axis[2];

    res.lengths = OverlapReal(0.5)*(proj_max - proj_min);

    res.lengths.x += vertex_radius;
    res.lengths.y += vertex_radius;
    res.lengths.z += vertex_radius;

    return res;
    }
#endif
}; // end namespace detail

}; // end namespace hpmc

#undef DEVICE
#endif //__OBB_H__
