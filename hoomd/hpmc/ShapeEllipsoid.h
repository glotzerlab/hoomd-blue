// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "ShapeSphere.h" //< For the base template of test_overlap
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iomanip>
#include <iostream>
#endif

#define ELLIPSOID_OVERLAP_ERROR 2
#define ELLIPSOID_OVERLAP_TRUE 1
#define ELLIPSOID_OVERLAP_FALSE 0

#if HOOMD_LONGREAL_SIZE == 32
#define ELLIPSOID_OVERLAP_PRECISION 1e-3
#else
#define ELLIPSOID_OVERLAP_PRECISION 1e-6
#endif

namespace hoomd
    {
namespace hpmc
    {
/** Ellipsoid parameters

    Define an ellipsoid shape by the three principal semi-axes.
*/
struct EllipsoidParams : ShapeParams
    {
    /// Principle semi-axis of the ellipsoid in the x-direction
    ShortReal x;

    /// Principle semi-axis of the ellipsoid in the y-direction
    ShortReal y;

    /// Principle semi-axis of the ellipsoid in the z-direction
    ShortReal z;

    /// True when move statistics should not be counted
    unsigned int ignore;

#ifdef ENABLE_HIP
    /// Set CUDA memory hints
    void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
    /// Default constructor
    EllipsoidParams() { }

    /// Construct from a Python dictionary
    EllipsoidParams(pybind11::dict v, bool managed = false)
        {
        ignore = v["ignore_statistics"].cast<unsigned int>();
        x = v["a"].cast<ShortReal>();
        y = v["b"].cast<ShortReal>();
        z = v["c"].cast<ShortReal>();

        if (x <= 0.0f || y <= 0.0f || z <= 0.0f)
            {
            throw std::domain_error("All semimajor axes must be nonzero!");
            }
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["a"] = x;
        v["b"] = y;
        v["c"] = z;
        v["ignore_statistics"] = ignore;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

/** Ellipsoid Polygon shape

    Implement the HPMC shape interface for ellipsoids.
*/
struct ShapeEllipsoid
    {
    /// Define the parameter type
    typedef EllipsoidParams param_type;

    //! Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeEllipsoid(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), axes(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation() const
        {
        return !(axes.x == axes.y && axes.x == axes.z);
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return axes.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the maximum of the 3 axes
        return ShortReal(2) * detail::max(axes.x, detail::max(axes.y, axes.z));
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    /** Support function of the shape (in local coordinates), used in getAABB
        @param n Vector to query support function (must be normalized)
    */
    DEVICE vec3<Scalar> sfunc(vec3<Scalar> n) const
        {
        vec3<Scalar> numerator(axes.x * axes.x * n.x, axes.y * axes.y * n.y, axes.z * axes.z * n.z);
        vec3<Scalar> dvec(axes.x * n.x, axes.y * n.y, axes.z * n.z);
        return numerator / fast::sqrt(dot(dvec, dvec));
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        ShortReal max_axis = detail::max(axes.x, detail::max(axes.y, axes.z));

        // generate a tight fitting AABB
        // ShortReal min_axis = min(axes.x, min(axes.y, axes.z));

        // use support function of the ellipsoid to determine the furthest extent in each direction
        // vec3<Scalar> e_x(1,0,0);
        // vec3<Scalar> e_y(0,1,0);
        // vec3<Scalar> e_z(0,0,1);
        // vec3<Scalar> s_x = rotate(orientation, sfunc(rotate(conj(orientation),e_x)));
        // vec3<Scalar> s_y = rotate(orientation, sfunc(rotate(conj(orientation),e_y)));
        // vec3<Scalar> s_z = rotate(orientation, sfunc(rotate(conj(orientation),e_z)));

        // // translate out from the position by the furthest extent
        // vec3<Scalar> upper(pos.x + s_x.x, pos.y + s_y.y, pos.z + s_z.z);
        // // the furthest extent is symmetrical
        // vec3<Scalar> lower(pos.x - s_x.x, pos.y - s_y.y, pos.z - s_z.z);

        // return hoomd::detail::AABB(lower, upper);
        // ^^^^^^^^^ The above method is slow, just use the circumsphere
        return hoomd::detail::AABB(pos, max_axis);
        }

    /** Returns true if this shape splits the overlap check over several threads of a warp using
        threadIdx.x
    */
    HOSTDEVICE static bool isParallel()
        {
        return false;
        }

    /// Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return false;
        }

    /// Orientation of the shape
    quat<Scalar> orientation;

    /// Shape parameters
    const EllipsoidParams& axes;
    };

namespace detail
    {
/** Compute a matrix representation of the ellipsoid
    @param M output matrix
    @param pos Position of the ellipsoid
    @param orientation Orientation of the ellipsoid
    @param axes Major axes of the ellipsoid

    @pre M has 10 elements
*/
DEVICE inline void compute_ellipsoid_matrix(ShortReal* M,
                                            const vec3<ShortReal>& pos,
                                            const quat<ShortReal>& orientation,
                                            const EllipsoidParams& axes)
    {
    // calculate rotation matrix
    rotmat3<ShortReal> R(orientation);

    // calculate ellipsoid matrix
    ShortReal a = ShortReal(1.0) / (axes.x * axes.x);
    ShortReal b = ShortReal(1.0) / (axes.y * axes.y);
    ShortReal c = ShortReal(1.0) / (axes.z * axes.z);
    // ...rotation part
    // M[i][j] = a * R[i][0] * R[j][0] + b * R[i][1] * R[j][1] + c * R[i][2] * R[j][2];
    M[0] = a * R.row0.x * R.row0.x + b * R.row0.y * R.row0.y + c * R.row0.z * R.row0.z;
    M[1] = a * R.row1.x * R.row0.x + b * R.row1.y * R.row0.y + c * R.row1.z * R.row0.z;
    M[2] = a * R.row1.x * R.row1.x + b * R.row1.y * R.row1.y + c * R.row1.z * R.row1.z;
    M[3] = a * R.row2.x * R.row0.x + b * R.row2.y * R.row0.y + c * R.row2.z * R.row0.z;
    M[4] = a * R.row2.x * R.row1.x + b * R.row2.y * R.row1.y + c * R.row2.z * R.row1.z;
    M[5] = a * R.row2.x * R.row2.x + b * R.row2.y * R.row2.y + c * R.row2.z * R.row2.z;

    // calculateTranslationPart(x, M);
    // precalculation
    ShortReal M0x0 = M[0] * pos.x;
    ShortReal M1x0 = M[1] * pos.x;
    ShortReal M1x1 = M[1] * pos.y;
    ShortReal M2x1 = M[2] * pos.y;
    ShortReal M3x0 = M[3] * pos.x;
    ShortReal M3x2 = M[3] * pos.z;
    ShortReal M4x1 = M[4] * pos.y;
    ShortReal M4x2 = M[4] * pos.z;
    ShortReal M5x2 = M[5] * pos.z;

    // ...translation part
    // M[i][3] = M[3][i] = -M[i][0] * x[0] - M[i][1] * x[1] - M[i][2] * x[2];
    M[6] = -M0x0 - M1x1 - M3x2;
    M[7] = -M1x0 - M2x1 - M4x2;
    M[8] = -M3x0 - M4x1 - M5x2;
    // ...mixed part
    // M[3][3] = -1.0 + M[0][0] * x[0] * x[0] + M[1][1] * x[1] * x[1] + M[2][2] * x[2] * x[2] +
    //           2.0 * (M[0][1] * x[0] * x[1] + M[1][2] * x[1] * x[2] + M[2][0] * x[2] * x[0]);
    M[9] = ShortReal(-1.0) + pos.x * (M0x0 + ShortReal(2.0) * M1x1)
           + pos.y * (M2x1 + ShortReal(2.0) * M4x2) + pos.z * (M5x2 + ShortReal(2.0) * M3x0);
    }

/** Checks for overlap between two ellipsoids

    @param M1 Matrix representing ellipsoid 1 in the check
    @param M2 Matrix representing ellipsoid 2 in the check
    @returns true when the two ellipsoids overlap

    @pre Both M1 and M2 are 10 elements
*/
DEVICE inline int test_overlap_ellipsoids(ShortReal* M1, ShortReal* M2)
    {
    // FIRST: calculate the coefficients a4, a3, a2, a1, a0 of the
    // characteristic polynomial that interpolates between M1 and M2
    // This means to look for solutions of det(M1 * x + M2) = 0

    // (i) Bottom 2x2 squares:
    // {{ M1[2][i] * x + M2[2][i],  M1[2][j] * x + M2[2][j] },
    //  { M1[3][i] * x + M2[3][i],  M1[3][j] * x + M2[3][j] }}
    // Use: M = [M00, M10, M11, M20, M21, M22, M30, M31, M32, M33]

    double s01_2 = M1[3] * M1[7] - M1[4] * M1[6];
    double s01_1 = M1[3] * M2[7] - M1[4] * M2[6] + M2[3] * M1[7] - M2[4] * M1[6];
    double s01_0 = M2[3] * M2[7] - M2[4] * M2[6];

    double s02_2 = M1[3] * M1[8] - M1[5] * M1[6];
    double s02_1 = M1[3] * M2[8] - M1[5] * M2[6] + M2[3] * M1[8] - M2[5] * M1[6];
    double s02_0 = M2[3] * M2[8] - M2[5] * M2[6];

    double s03_2 = M1[3] * M1[9] - M1[8] * M1[6];
    double s03_1 = M1[3] * M2[9] - M1[8] * M2[6] + M2[3] * M1[9] - M2[8] * M1[6];
    double s03_0 = M2[3] * M2[9] - M2[8] * M2[6];

    double s12_2 = M1[4] * M1[8] - M1[5] * M1[7];
    double s12_1 = M1[4] * M2[8] - M1[5] * M2[7] + M2[4] * M1[8] - M2[5] * M1[7];
    double s12_0 = M2[4] * M2[8] - M2[5] * M2[7];

    double s13_2 = M1[4] * M1[9] - M1[8] * M1[7];
    double s13_1 = M1[4] * M2[9] - M1[8] * M2[7] + M2[4] * M1[9] - M2[8] * M1[7];
    double s13_0 = M2[4] * M2[9] - M2[8] * M2[7];

    double s23_2 = M1[5] * M1[9] - M1[8] * M1[8];
    double s23_1 = M1[5] * M2[9] - M1[8] * M2[8] + M2[5] * M1[9] - M2[8] * M1[8];
    double s23_0 = M2[5] * M2[9] - M2[8] * M2[8];

    // (ii) Bottom 3x3 parts:

    double t0_3 = M1[2] * s23_2 - M1[4] * s13_2 + M1[7] * s12_2;
    double t0_2 = M1[2] * s23_1 - M1[4] * s13_1 + M1[7] * s12_1 + M2[2] * s23_2 - M2[4] * s13_2
                  + M2[7] * s12_2;
    double t0_1 = M1[2] * s23_0 - M1[4] * s13_0 + M1[7] * s12_0 + M2[2] * s23_1 - M2[4] * s13_1
                  + M2[7] * s12_1;
    double t0_0 = M2[2] * s23_0 - M2[4] * s13_0 + M2[7] * s12_0;

    double t1_3 = M1[1] * s23_2 - M1[4] * s03_2 + M1[7] * s02_2;
    double t1_2 = M1[1] * s23_1 - M1[4] * s03_1 + M1[7] * s02_1 + M2[1] * s23_2 - M2[4] * s03_2
                  + M2[7] * s02_2;
    double t1_1 = M1[1] * s23_0 - M1[4] * s03_0 + M1[7] * s02_0 + M2[1] * s23_1 - M2[4] * s03_1
                  + M2[7] * s02_1;
    double t1_0 = M2[1] * s23_0 - M2[4] * s03_0 + M2[7] * s02_0;

    double t2_3 = M1[1] * s13_2 - M1[2] * s03_2 + M1[7] * s01_2;
    double t2_2 = M1[1] * s13_1 - M1[2] * s03_1 + M1[7] * s01_1 + M2[1] * s13_2 - M2[2] * s03_2
                  + M2[7] * s01_2;
    double t2_1 = M1[1] * s13_0 - M1[2] * s03_0 + M1[7] * s01_0 + M2[1] * s13_1 - M2[2] * s03_1
                  + M2[7] * s01_1;
    double t2_0 = M2[1] * s13_0 - M2[2] * s03_0 + M2[7] * s01_0;

    double t3_3 = M1[1] * s12_2 - M1[2] * s02_2 + M1[4] * s01_2;
    double t3_2 = M1[1] * s12_1 - M1[2] * s02_1 + M1[4] * s01_1 + M2[1] * s12_2 - M2[2] * s02_2
                  + M2[4] * s01_2;
    double t3_1 = M1[1] * s12_0 - M1[2] * s02_0 + M1[4] * s01_0 + M2[1] * s12_1 - M2[2] * s02_1
                  + M2[4] * s01_1;
    double t3_0 = M2[1] * s12_0 - M2[2] * s02_0 + M2[4] * s01_0;

    // (iii) Full 4x4 matrix:

    double a4 = M1[0] * t0_3 - M1[1] * t1_3 + M1[3] * t2_3 - M1[6] * t3_3;
    double a3 = M1[0] * t0_2 - M1[1] * t1_2 + M1[3] * t2_2 - M1[6] * t3_2 + M2[0] * t0_3
                - M2[1] * t1_3 + M2[3] * t2_3 - M2[6] * t3_3;
    double a2 = M1[0] * t0_1 - M1[1] * t1_1 + M1[3] * t2_1 - M1[6] * t3_1 + M2[0] * t0_2
                - M2[1] * t1_2 + M2[3] * t2_2 - M2[6] * t3_2;
    double a1 = M1[0] * t0_0 - M1[1] * t1_0 + M1[3] * t2_0 - M1[6] * t3_0 + M2[0] * t0_1
                - M2[1] * t1_1 + M2[3] * t2_1 - M2[6] * t3_1;
    double a0 = M2[0] * t0_0 - M2[1] * t1_0 + M2[3] * t2_0 - M2[6] * t3_0;

    // SECOND: analyze the polynomial for overlaps
    // a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0 = 0
    // We are looking for one (at least one) positive root of this polynomial.
    // If there is one, then the ellipsoids overlap!
    // a4 and a0 are always positive, so check if the other coefficients are positive too

    // NOTE [BEN]: Above comment appears to be an error.
    // In the implementation, it appears as though the we're actually searching for
    // roots between -infinity and 0, so a1 and a3 flip sign. All of these sign changes cancel
    // out in the sturm theorem check

    //(i) Descartes rule of signs check (if all coeffs are <0, then overlap)
    if (a1 < 0.0 && a2 < 0.0 && a3 < 0.0)
        return ELLIPSOID_OVERLAP_TRUE;

    // (ii) Sturm theorem/Sturm sequence:
    double a4Inv = 1.0 / a4;
    double a = a3 * a4Inv;
    double b = a2 * a4Inv;
    double c = a1 * a4Inv;
    double d = a0 * a4Inv;

    double e0 = 3 * a * a - 8 * b;
    if (fabs(e0) > ELLIPSOID_OVERLAP_PRECISION * detail::max(fabs(3 * a * a), fabs(8 * b)))
        {
        if (e0 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        }
    double f0 = 2 * a * b - 12 * c;
    double g0 = a * c - 16 * d;
    double h0 = 3 * a * e0 * f0 - 2 * b * e0 * e0 + 4 * (g0 * e0 - f0 * f0);
    if (fabs(h0) > ELLIPSOID_OVERLAP_PRECISION
                       * detail::max(detail::max(fabs(3 * a * e0 * f0), fabs(2 * b * e0 * e0)),
                                     4 * (g0 * e0 - f0 * f0)))
        {
        if (h0 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        }

    double k0 = 3 * a * g0 * e0 - c * e0 * e0 - 4 * f0 * g0;
    // nominally, we need to compute l = f0 * k0 * h0 - (h0 * h0 * g0 + e0 * k0 * k0) and compare it
    // to 0 but the two terms are often VERY close. Instead, compute the ratio of the terms. If it
    // is 1 within some tolerance, assume no overlap. Otherwise, do the check of l vs 0
    a = f0 * k0 * h0;
    b = h0 * h0 * g0;
    c = e0 * k0 * k0;
    double l0 = a - (b + c);

    if (fabs(l0)
        > ELLIPSOID_OVERLAP_PRECISION * detail::max(detail::max(fabs(a), fabs(b)), fabs(c)))
        {
        if (l0 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        return ELLIPSOID_OVERLAP_FALSE;
        }
    // reverse order of the check
    // (ii) Sturm theorem/Sturm sequence:
    a4Inv = 1.0 / a0;
    a = a1 * a4Inv;
    b = a2 * a4Inv;
    c = a3 * a4Inv;
    d = a4 * a4Inv;

    double e1 = 3 * a * a - 8 * b;

    // precision check
    if (fabs(e1) > ELLIPSOID_OVERLAP_PRECISION * detail::max(fabs(3 * a * a), fabs(8 * b)))
        {
        if (e1 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        }
    double f1 = 2 * a * b - 12 * c;
    double g1 = a * c - 16 * d;
    double h1 = 3 * a * e1 * f1 - 2 * b * e1 * e1 + 4 * (g1 * e1 - f1 * f1);

    // precision check
    if (fabs(h1) > ELLIPSOID_OVERLAP_PRECISION
                       * detail::max(detail::max(fabs(3 * a * e1 * f1), fabs(2 * b * e1 * e1)),
                                     4 * (g1 * e1 - f1 * f1)))
        {
        if (h1 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        }

    double k1 = 3 * a * g1 * e1 - c * e1 * e1 - 4 * f1 * g1;
    // nominally, we need to compute l1 = f1 * k1 * h1 - (h1 * h1 * g1 + e1 * k1 * k1) and compare
    // it to 0 but the two terms are often VERY close. Instead, compute the ratio of the terms. If
    // it is 1 within some tolerance, assume no overlap. Otherwise, do the check of l1 vs 0
    a = f1 * k1 * h1;
    b = h1 * h1 * g1;
    c = e1 * k1 * k1;
    double l1 = a - (b + c);

    // precision check
    if (fabs(l1)
        > ELLIPSOID_OVERLAP_PRECISION * detail::max(detail::max(fabs(a), fabs(b)), fabs(c)))
        {
        if (l1 < 0.0)
            return ELLIPSOID_OVERLAP_TRUE;
        return ELLIPSOID_OVERLAP_FALSE;
        }

    return ELLIPSOID_OVERLAP_ERROR;
    }

    }; // end namespace detail

/** Ellipsoid overlap test

    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a Shape a
    @param b Shape b
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius Additional sphere radius to sweep the shapes with
    @returns true if the two particles overlap
*/
template<>
DEVICE inline bool test_overlap<ShapeEllipsoid, ShapeEllipsoid>(const vec3<Scalar>& r_ab,
                                                                const ShapeEllipsoid& a,
                                                                const ShapeEllipsoid& b,
                                                                unsigned int& err)
    {
    // matrix representations of the two ellipsoids
    vec3<ShortReal> dr(r_ab);

    // shortcut if ellipsoids are actually spheres
    if (a.axes.x == a.axes.y && a.axes.x == a.axes.z && b.axes.x == b.axes.y
        && b.axes.x == b.axes.z)
        {
        ShortReal ab = a.axes.x + b.axes.x;
        return (dot(dr, dr) <= ab * ab);
        }

    ShortReal Ma[10], Mb[10];
    detail::compute_ellipsoid_matrix(Ma,
                                     vec3<ShortReal>(0, 0, 0),
                                     quat<ShortReal>(a.orientation),
                                     a.axes);
    detail::compute_ellipsoid_matrix(Mb, dr, quat<ShortReal>(b.orientation), b.axes);

    int ret_val = detail::test_overlap_ellipsoids(Ma, Mb);
    if (ret_val == ELLIPSOID_OVERLAP_ERROR)
        {
        err++;
        }

    return ret_val == ELLIPSOID_OVERLAP_TRUE;
    }

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeEllipsoid& ellipsoid)
    {
    std::ostringstream shapedef;
    shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << ellipsoid.axes.x
             << ", \"b\": " << ellipsoid.axes.y << ", \"c\": " << ellipsoid.axes.z << "}";
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
