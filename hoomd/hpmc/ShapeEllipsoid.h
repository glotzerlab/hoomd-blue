// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap

#ifndef __SHAPE_ELLIPSOID_H__
#define __SHAPE_ELLIPSOID_H__

/*! \file ShapeEllipsoid.h
    \brief Defines the ellipsoid shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#include <iomanip>
#endif

#define ELLIPSOID_OVERLAP_ERROR 2
#define ELLIPSOID_OVERLAP_TRUE 1
#define ELLIPSOID_OVERLAP_FALSE 0

#ifdef SINGLE_PRECISION
#define ELLIPSOID_OVERLAP_PRECISION 1e-3
#else
#define ELLIPSOID_OVERLAP_PRECISION 1e-6
#endif

namespace hpmc
{

//! Ellipsoid shape template
/*! ShapeEllipsoid implements IntegratorHPMC's shape protocol.

    The parameter defining an ellipsoid is a OverlapReal4. First three components list the major axis in that direction.
    The last component (w) is a ignore flag for overlaps. If w!=0, for both particles in overlap check, then overlaps
    between those particles will be ignored.

    \ingroup shape
*/
struct ell_params : param_base
    {
    OverlapReal x;                      //!< x semiaxis of the ellipsoid
    OverlapReal y;                      //!< y semiaxis of the ellipsoid
    OverlapReal z;                      //!< z semiaxis of the ellipsoid
    unsigned int ignore;                //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                        //   First bit is ignore overlaps, Second bit is ignore statistics

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // default implementation does nothing
        }
    #endif
    } __attribute__((aligned(32)));

struct ShapeEllipsoid
    {
    //! Define the parameter type
    typedef ell_params param_type;

    //! Initialize a polygon
    DEVICE ShapeEllipsoid(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), axes(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const { return !(axes.x==axes.y&&axes.x==axes.z); }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return axes.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the maximum of the 3 axes
        return OverlapReal(2)*detail::max(axes.x, detail::max(axes.y, axes.z));
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << axes.x <<
                    ", \"b\": " << axes.y <<
                    ", \"c\": " << axes.z <<
                    "}";
        return shapedef.str();
        }
    #endif

    //! Support function of the shape (in local coordinates), used in getAABB
    /*! \param n Vector to query support function (must be normalized)
    */
    DEVICE vec3<Scalar> sfunc(vec3<Scalar> n) const
        {
        vec3<Scalar> numerator(axes.x*axes.x*n.x, axes.y*axes.y*n.y, axes.z*axes.z*n.z);
        vec3<Scalar> dvec(axes.x*n.x, axes.y*n.y, axes.z*n.z);
        return numerator / fast::sqrt(dot(dvec, dvec));
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        OverlapReal max_axis = detail::max(axes.x, detail::max(axes.y, axes.z));

        // generate a tight fitting AABB
        // OverlapReal min_axis = min(axes.x, min(axes.y, axes.z));

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

        // return detail::AABB(lower, upper);
        // ^^^^^^^^^ The above method is slow, just use the circumsphere
        return detail::AABB(pos, max_axis);
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the polygon

    ell_params axes;     //!< Radii of major axesI
    };

namespace detail
{

//! Compute a matrix representation of the ellipsoid
/*! \param M output matrix
    \param pos Position of the ellipsoid
    \param orientation Orientation of the ellipsoid
    \param axes Major axes of the ellipsoid

    \pre M has 10 elements

    \ingroup overlap
*/
DEVICE inline void compute_ellipsoid_matrix(OverlapReal *M,
                                            const vec3<OverlapReal>& pos,
                                            const quat<OverlapReal>& orientation,
                                            const ell_params& axes)
    {
    // This code is copied from incsim. TODO there may be licensing issues with including this, but since we aren't
    // planning on releasing hpmc any time soon it doesn't matter

    // calculate rotation matrix
    rotmat3<OverlapReal> R(orientation);

    // calculate ellipsoid matrix
    OverlapReal a = OverlapReal(1.0) / (axes.x * axes.x);
    OverlapReal b = OverlapReal(1.0) / (axes.y * axes.y);
    OverlapReal c = OverlapReal(1.0) / (axes.z * axes.z);
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
    OverlapReal M0x0 = M[0] * pos.x;
    OverlapReal M1x0 = M[1] * pos.x;
    OverlapReal M1x1 = M[1] * pos.y;
    OverlapReal M2x1 = M[2] * pos.y;
    OverlapReal M3x0 = M[3] * pos.x;
    OverlapReal M3x2 = M[3] * pos.z;
    OverlapReal M4x1 = M[4] * pos.y;
    OverlapReal M4x2 = M[4] * pos.z;
    OverlapReal M5x2 = M[5] * pos.z;

    // ...translation part
    // M[i][3] = M[3][i] = -M[i][0] * x[0] - M[i][1] * x[1] - M[i][2] * x[2];
    M[6] = -M0x0 - M1x1 - M3x2;
    M[7] = -M1x0 - M2x1 - M4x2;
    M[8] = -M3x0 - M4x1 - M5x2;
    // ...mixed part
    // M[3][3] = -1.0 + M[0][0] * x[0] * x[0] + M[1][1] * x[1] * x[1] + M[2][2] * x[2] * x[2] +
    //           2.0 * (M[0][1] * x[0] * x[1] + M[1][2] * x[1] * x[2] + M[2][0] * x[2] * x[0]);
    M[9] = OverlapReal(-1.0) + pos.x * (M0x0 + OverlapReal(2.0) * M1x1) +
                               pos.y * (M2x1 + OverlapReal(2.0) * M4x2) +
                               pos.z * (M5x2 + OverlapReal(2.0) * M3x0);
    }

//! Checks for overlap between two ellipsoids
/*! \param M1 Matrix representing ellipsoid 1 in the check
    \param M2 Matrix representing ellipsoid 2 in the check
    \returns true when the two ellipsoids overlap

    \pre Both M1 and M2 are 10 elements

    This code is copied from incsim. TODO there may be licensing issues with including this, but since we aren't
    planning on releasing hpmc any time soon it doesn't matter

    \ingroup overlap
*/
DEVICE inline int test_overlap_ellipsoids(OverlapReal *M1, OverlapReal *M2)
    {
    // FIRST: calculate the coefficients a4, a3, a2, a1, a0 of the
    // characteristic polynomial that interpolates between M1 and M2
    // This means to look for solutions of det(M1 * x + M2) = 0

    // (i) Bottom 2x2 squares:
    // {{ M1[2][i] * x + M2[2][i],  M1[2][j] * x + M2[2][j] },
    //  { M1[3][i] * x + M2[3][i],  M1[3][j] * x + M2[3][j] }}
    // Use: M = [M00, M10, M11, M20, M21, M22, M30, M31, M32, M33]

    double s01_2 = M1[3] * M1[7] - M1[4] * M1[6];
    double s01_1 = M1[3] * M2[7] - M1[4] * M2[6] +
                        M2[3] * M1[7] - M2[4] * M1[6];
    double s01_0 = M2[3] * M2[7] - M2[4] * M2[6];

    double s02_2 = M1[3] * M1[8] - M1[5] * M1[6];
    double s02_1 = M1[3] * M2[8] - M1[5] * M2[6] +
                        M2[3] * M1[8] - M2[5] * M1[6];
    double s02_0 = M2[3] * M2[8] - M2[5] * M2[6];

    double s03_2 = M1[3] * M1[9] - M1[8] * M1[6];
    double s03_1 = M1[3] * M2[9] - M1[8] * M2[6] +
                        M2[3] * M1[9] - M2[8] * M1[6];
    double s03_0 = M2[3] * M2[9] - M2[8] * M2[6];

    double s12_2 = M1[4] * M1[8] - M1[5] * M1[7];
    double s12_1 = M1[4] * M2[8] - M1[5] * M2[7] +
                        M2[4] * M1[8] - M2[5] * M1[7];
    double s12_0 = M2[4] * M2[8] - M2[5] * M2[7];

    double s13_2 = M1[4] * M1[9] - M1[8] * M1[7];
    double s13_1 = M1[4] * M2[9] - M1[8] * M2[7] +
                        M2[4] * M1[9] - M2[8] * M1[7];
    double s13_0 = M2[4] * M2[9] - M2[8] * M2[7];

    double s23_2 = M1[5] * M1[9] - M1[8] * M1[8];
    double s23_1 = M1[5] * M2[9] - M1[8] * M2[8] +
                        M2[5] * M1[9] - M2[8] * M1[8];
    double s23_0 = M2[5] * M2[9] - M2[8] * M2[8];

    // (ii) Bottom 3x3 parts:

    double t0_3 = M1[2] * s23_2 - M1[4] * s13_2 + M1[7] * s12_2;
    double t0_2 = M1[2] * s23_1 - M1[4] * s13_1 + M1[7] * s12_1 +
                       M2[2] * s23_2 - M2[4] * s13_2 + M2[7] * s12_2;
    double t0_1 = M1[2] * s23_0 - M1[4] * s13_0 + M1[7] * s12_0 +
                       M2[2] * s23_1 - M2[4] * s13_1 + M2[7] * s12_1;
    double t0_0 = M2[2] * s23_0 - M2[4] * s13_0 + M2[7] * s12_0;

    double t1_3 = M1[1] * s23_2 - M1[4] * s03_2 + M1[7] * s02_2;
    double t1_2 = M1[1] * s23_1 - M1[4] * s03_1 + M1[7] * s02_1 +
                       M2[1] * s23_2 - M2[4] * s03_2 + M2[7] * s02_2;
    double t1_1 = M1[1] * s23_0 - M1[4] * s03_0 + M1[7] * s02_0 +
                       M2[1] * s23_1 - M2[4] * s03_1 + M2[7] * s02_1;
    double t1_0 = M2[1] * s23_0 - M2[4] * s03_0 + M2[7] * s02_0;

    double t2_3 = M1[1] * s13_2 - M1[2] * s03_2 + M1[7] * s01_2;
    double t2_2 = M1[1] * s13_1 - M1[2] * s03_1 + M1[7] * s01_1 +
                       M2[1] * s13_2 - M2[2] * s03_2 + M2[7] * s01_2;
    double t2_1 = M1[1] * s13_0 - M1[2] * s03_0 + M1[7] * s01_0 +
                       M2[1] * s13_1 - M2[2] * s03_1 + M2[7] * s01_1;
    double t2_0 = M2[1] * s13_0 - M2[2] * s03_0 + M2[7] * s01_0;

    double t3_3 = M1[1] * s12_2 - M1[2] * s02_2 + M1[4] * s01_2;
    double t3_2 = M1[1] * s12_1 - M1[2] * s02_1 + M1[4] * s01_1 +
                       M2[1] * s12_2 - M2[2] * s02_2 + M2[4] * s01_2;
    double t3_1 = M1[1] * s12_0 - M1[2] * s02_0 + M1[4] * s01_0 +
                       M2[1] * s12_1 - M2[2] * s02_1 + M2[4] * s01_1;
    double t3_0 = M2[1] * s12_0 - M2[2] * s02_0 + M2[4] * s01_0;

    // (iii) Full 4x4 matrix:

    double a4 = M1[0] * t0_3 - M1[1] * t1_3 + M1[3] * t2_3 - M1[6] * t3_3;
    double a3 = M1[0] * t0_2 - M1[1] * t1_2 + M1[3] * t2_2 - M1[6] * t3_2 +
                     M2[0] * t0_3 - M2[1] * t1_3 + M2[3] * t2_3 - M2[6] * t3_3;
    double a2 = M1[0] * t0_1 - M1[1] * t1_1 + M1[3] * t2_1 - M1[6] * t3_1 +
                     M2[0] * t0_2 - M2[1] * t1_2 + M2[3] * t2_2 - M2[6] * t3_2;
    double a1 = M1[0] * t0_0 - M1[1] * t1_0 + M1[3] * t2_0 - M1[6] * t3_0 +
                     M2[0] * t0_1 - M2[1] * t1_1 + M2[3] * t2_1 - M2[6] * t3_1;
    double a0 = M2[0] * t0_0 - M2[1] * t1_0 + M2[3] * t2_0 - M2[6] * t3_0;

    // SECOND: analyze the polynomial for overlaps
    // a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0 = 0
    // We are looking for one (at least one) positive root of this polynomial.
    // If there is one, then the ellipsoids overlap!
    // a4 and a0 are always positive, so check if the other coefficients are positive too

    //NOTE [BEN]: Above comment appears to be an error.
    //In the implementation, it appears as though the we're actually searching for
    //roots between -infinity and 0, so a1 and a3 flip sign. All of these sign changes cancel
    //out in the sturm theorem check

    //(i) Descartes rule of signs check (if all coeffs are <0, then overlap)
    if (a1 < 0.0 && a2 < 0.0 && a3 < 0.0) return ELLIPSOID_OVERLAP_TRUE;

    // (ii) Sturm theorem/Sturm sequence:
    double a4Inv = 1.0 / a4;
    double a = a3 * a4Inv;
    double b = a2 * a4Inv;
    double c = a1 * a4Inv;
    double d = a0 * a4Inv;

    double e0 = 3 * a * a - 8 * b;
    if (fabs(e0)>ELLIPSOID_OVERLAP_PRECISION*detail::max(fabs(3*a*a),fabs(8*b)))
        {
        if (e0 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        }
    double f0 = 2 * a * b - 12 * c;
    double g0 = a * c - 16 * d;
    double h0 = 3 * a * e0 * f0 - 2 * b * e0 * e0 + 4 * (g0 * e0 - f0 * f0);
    if (fabs(h0)>ELLIPSOID_OVERLAP_PRECISION*detail::max(detail::max(fabs(3 * a * e0 * f0),fabs(2 * b * e0 * e0)), 4 * (g0 * e0 - f0 * f0)))
        {
        if (h0 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        }

    double k0 = 3 * a * g0 * e0 - c * e0 * e0 - 4 * f0 * g0;
    // nominally, we need to compute l = f0 * k0 * h0 - (h0 * h0 * g0 + e0 * k0 * k0) and compare it to 0
    // but the two terms are often VERY close. Instead, compute the ratio of the terms. If it is 1 within
    // some tolerance, assume no overlap. Otherwise, do the check of l vs 0
    a = f0 * k0 * h0;
    b = h0 * h0 * g0;
    c = e0 * k0 * k0;
    double l0 =  a - (b + c);

    if (fabs(l0)>ELLIPSOID_OVERLAP_PRECISION*detail::max(detail::max(fabs(a),fabs(b)),fabs(c)))
        {
        if (l0 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        return ELLIPSOID_OVERLAP_FALSE;
        }
    //reverse order of the check
    // (ii) Sturm theorem/Sturm sequence:
    a4Inv = 1.0 / a0;
    a = a1 * a4Inv;
    b = a2 * a4Inv;
    c = a3 * a4Inv;
    d = a4 * a4Inv;

    double e1 = 3 * a * a - 8 * b;

    //precision check
    if (fabs(e1)>ELLIPSOID_OVERLAP_PRECISION*detail::max(fabs(3*a*a),fabs(8*b)))
        {
        if (e1 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        }
    double f1 = 2 * a * b - 12 * c;
    double g1 = a * c - 16 * d;
    double h1 = 3 * a * e1 * f1 - 2 * b * e1 * e1 + 4 * (g1 * e1 - f1 * f1);

    //precision check
    if (fabs(h1)>ELLIPSOID_OVERLAP_PRECISION*detail::max(detail::max(fabs(3 * a * e1 * f1),fabs(2 * b * e1 * e1)), 4 * (g1 * e1 - f1 * f1)))
        {
        if (h1 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        }

    double k1 = 3 * a * g1 * e1 - c * e1 * e1 - 4 * f1 * g1;
    // nominally, we need to compute l1 = f1 * k1 * h1 - (h1 * h1 * g1 + e1 * k1 * k1) and compare it to 0
    // but the two terms are often VERY close. Instead, compute the ratio of the terms. If it is 1 within
    // some tolerance, assume no overlap. Otherwise, do the check of l1 vs 0
    a = f1 * k1 * h1;
    b = h1 * h1 * g1;
    c = e1 * k1 * k1;
    double l1 = a - (b +c);

    //precision check
    if (fabs(l1)>ELLIPSOID_OVERLAP_PRECISION*detail::max(detail::max(fabs(a),fabs(b)),fabs(c)))
        {
        if (l1 < 0.0) return ELLIPSOID_OVERLAP_TRUE;
        return ELLIPSOID_OVERLAP_FALSE;
        }

    return ELLIPSOID_OVERLAP_ERROR;
    }

}; // end namespace detail

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeEllipsoid& a,
    const ShapeEllipsoid &b)
    {
    //otherwise actually check circumspheres for earl exit
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

//! Ellipsoid overlap test
/*!
    \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a Shape a
    \param b Shape b
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true if the two particles overlap

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeEllipsoid,ShapeEllipsoid>(const vec3<Scalar>& r_ab,
                                                               const ShapeEllipsoid& a,
                                                               const ShapeEllipsoid& b,
                                                               unsigned int& err)
    {

    // matrix representations of the two ellipsoids
    vec3<OverlapReal> dr(r_ab);

    //shortcut if ellipsoids are actually spheres
    if(a.axes.x==a.axes.y && a.axes.x==a.axes.z && b.axes.x==b.axes.y && b.axes.x==b.axes.z)
       {
       OverlapReal ab = a.axes.x + b.axes.x;
       return (dot(dr,dr) <= ab*ab);
       }

    OverlapReal Ma[10], Mb[10];
    detail::compute_ellipsoid_matrix(Ma, vec3<OverlapReal>(0,0,0), quat<OverlapReal>(a.orientation), a.axes);
    detail::compute_ellipsoid_matrix(Mb, dr, quat<OverlapReal>(b.orientation), b.axes);

    int ret_val = detail::test_overlap_ellipsoids(Ma, Mb);
    if (ret_val == ELLIPSOID_OVERLAP_ERROR)
        {
        err++;
        }

    return ret_val == ELLIPSOID_OVERLAP_TRUE;
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_ELLIPSOID_H__
