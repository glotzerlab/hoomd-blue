// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "XenoCollide2D.h"

#ifndef __SHAPE_CONVEX_POLYGON_H__
#define __SHAPE_CONVEX_POLYGON_H__

/*! \file ShapeConvexPolygon.h
    \brief Defines the convex polygon shape
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
#if defined (__SSE__)
#include <immintrin.h>
#endif
#endif

namespace hpmc
{

namespace detail
{

//! maximum number of vertices that can be stored (must be a multiple of 8)
//! Note that vectorized methods using this struct will assume unused coordinates are set to zero.
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_POLY2D_VERTS=64;

//! Data structure for polygon vertices
/*! \ingroup hpmc_data_structs */
struct poly2d_verts : param_base
    {
    //! Default constructor initializes zero values.
    DEVICE poly2d_verts()
        : N(0),
          diameter(OverlapReal(0)),
          sweep_radius(OverlapReal(0)),
          ignore(0)
        {
        for (unsigned int i=0; i < MAX_POLY2D_VERTS; i++)
            {
            x[i] = y[i] = OverlapReal(0);
            }
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        // default implementation does nothing
        }
    #endif

    OverlapReal x[MAX_POLY2D_VERTS];    //!< X coordinate of vertices
    OverlapReal y[MAX_POLY2D_VERTS];    //!< Y coordinate of vertices
    unsigned int N;                     //!< Number of vertices
    OverlapReal diameter;               //!< Precomputed diameter
    OverlapReal sweep_radius;           //!< Radius of the sphere sweep (used for spheropolygons)
    unsigned int ignore;                //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                        //   First bit is ignore overlaps, Second bit is ignore statistics
    } __attribute__((aligned(32)));

//! Support function for ShapeConvexPolygon
/*! SupportFuncConvexPolygon is a functor that computes the support function for ShapeConvexPolygon. For a given
    input vector in local coordinates, it finds the vertex most in that direction.

    \ingroup minkowski
*/
class SupportFuncConvexPolygon
    {
    public:
        //! Construct a support function for a convex polygon
        /*! Note that for performance it is assumed that unused vertices (beyond N) have already been set to zero.
            \param _verts Polygon vertices
        */
        DEVICE SupportFuncConvexPolygon(const poly2d_verts& _verts)
            : verts(_verts)
            {
            }

        //! Compute the support function
        /*! \param n Normal vector input (in the local frame)
            \returns Local coords of the point furthest in the direction of n
        */
        DEVICE vec2<OverlapReal> operator() (const vec2<OverlapReal>& n) const
            {
            OverlapReal max_dot = -(verts.diameter * verts.diameter);
            unsigned int max_idx = 0;

            if (verts.N > 0)
                {
                #if !defined(NVCC) && defined(__AVX__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
                // process dot products with AVX 8 at a time on the CPU when working with more than 4 verts
                __m256 nx_v = _mm256_broadcast_ss(&n.x);
                __m256 ny_v = _mm256_broadcast_ss(&n.y);
                __m256 max_dot_v = _mm256_broadcast_ss(&max_dot);
                float d_s[MAX_POLY2D_VERTS] __attribute__((aligned(32)));

                for (size_t i = 0; i < verts.N; i+=8)
                    {
                    __m256 x_v = _mm256_load_ps(verts.x + i);
                    __m256 y_v = _mm256_load_ps(verts.y + i);

                    __m256 d_v = _mm256_add_ps(_mm256_mul_ps(nx_v, x_v), _mm256_mul_ps(ny_v, y_v));

                    // determine a maximum in each of the 8 channels as we go
                    max_dot_v = _mm256_max_ps(max_dot_v, d_v);
                    _mm256_store_ps(d_s + i, d_v);
                    }

                // find the maximum of the 8 channels
                // http://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
                max_dot_v = _mm256_max_ps(max_dot_v, _mm256_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(2, 1, 0, 3)));
                max_dot_v = _mm256_max_ps(max_dot_v, _mm256_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(1, 0, 3, 2)));
                // shuffles work only within the two 128b segments, so right now we have two separate max values
                // swap the left and right hand sides and max again to get the final max
                max_dot_v = _mm256_max_ps(max_dot_v, _mm256_permute2f128_ps(max_dot_v, max_dot_v, 1));

                // loop again and find the max. The reason this is in a 2nd loop is because branch mis-predictions
                // and the extra max calls kill performance if this is in the first loop
                // Use BSF to find the first index of the max element
                // https://software.intel.com/en-us/forums/topic/285956
                for (unsigned int i = 0; i < verts.N; i+=8)
                    {
                    __m256 d_v = _mm256_load_ps(d_s + i);

                    int id = __builtin_ffs(_mm256_movemask_ps(_mm256_cmp_ps(max_dot_v, d_v, 0)));

                    if (id)
                        {
                        max_idx = i + id - 1;
                        break;
                        }
                    }

                #elif !defined(NVCC) && defined(__SSE__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
                // process dot products with SSE 4 at a time on the CPU
                __m128 nx_v = _mm_load_ps1(&n.x);
                __m128 ny_v = _mm_load_ps1(&n.y);
                __m128 max_dot_v = _mm_load_ps1(&max_dot);
                float d_s[MAX_POLY2D_VERTS] __attribute__((aligned(16)));

                for (unsigned int i = 0; i < verts.N; i+=4)
                    {
                    __m128 x_v = _mm_load_ps(verts.x + i);
                    __m128 y_v = _mm_load_ps(verts.y + i);

                    __m128 d_v = _mm_add_ps(_mm_mul_ps(nx_v, x_v), _mm_mul_ps(ny_v, y_v));

                    // determine a maximum in each of the 4 channels as we go
                    max_dot_v = _mm_max_ps(max_dot_v, d_v);

                    _mm_store_ps(d_s + i, d_v);
                    }

                // find the maximum of the 4 channels
                // http://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
                max_dot_v = _mm_max_ps(max_dot_v, _mm_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(2, 1, 0, 3)));
                max_dot_v = _mm_max_ps(max_dot_v, _mm_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(1, 0, 3, 2)));

                // loop again and find the max. The reason this is in a 2nd loop is because branch mis-predictions
                // and the extra max calls kill performance if this is in the first loop
                // Use BSF to find the first index of the max element
                // https://software.intel.com/en-us/forums/topic/285956
                for (unsigned int i = 0; i < verts.N; i+=4)
                    {
                    __m128 d_v = _mm_load_ps(d_s + i);

                    int id = __builtin_ffs(_mm_movemask_ps(_mm_cmpeq_ps(max_dot_v, d_v)));

                    if (id)
                        {
                        max_idx = i + id - 1;
                        break;
                        }
                    }
                #else


                OverlapReal max_dot0 = dot(n, vec2<OverlapReal>(verts.x[0], verts.y[0]));
                unsigned int max_idx0 = 0;
                OverlapReal max_dot1 = dot(n, vec2<OverlapReal>(verts.x[1], verts.y[1]));
                unsigned int max_idx1 = 1;
                OverlapReal max_dot2 = dot(n, vec2<OverlapReal>(verts.x[2], verts.y[2]));
                unsigned int max_idx2 = 2;
                OverlapReal max_dot3 = dot(n, vec2<OverlapReal>(verts.x[3], verts.y[3]));
                unsigned int max_idx3 = 3;

                for (unsigned int i = 4; i < verts.N; i+=4)
                    {
                    OverlapReal d0 = dot(n, vec2<OverlapReal>(verts.x[i], verts.y[i]));
                    OverlapReal d1 = dot(n, vec2<OverlapReal>(verts.x[i+1], verts.y[i+1]));
                    OverlapReal d2 = dot(n, vec2<OverlapReal>(verts.x[i+2], verts.y[i+2]));
                    OverlapReal d3 = dot(n, vec2<OverlapReal>(verts.x[i+3], verts.y[i+3]));

                    if (d0 > max_dot0)
                        {
                        max_dot0 = d0;
                        max_idx0 = i;
                        }
                    if (d1 > max_dot1)
                        {
                        max_dot1 = d1;
                        max_idx1 = i+1;
                        }
                    if (d2 > max_dot2)
                        {
                        max_dot2 = d2;
                        max_idx2 = i+2;
                        }
                    if (d3 > max_dot3)
                        {
                        max_dot3 = d3;
                        max_idx3 = i+3;
                        }
                    }

                max_dot = max_dot0;
                max_idx = max_idx0;

                if (max_dot1 > max_dot)
                    {
                    max_dot = max_dot1;
                    max_idx = max_idx1;
                    }
                if (max_dot2 > max_dot)
                    {
                    max_dot = max_dot2;
                    max_idx = max_idx2;
                    }
                if (max_dot3 > max_dot)
                    {
                    max_dot = max_dot3;
                    max_idx = max_idx3;
                    }
                #endif
                }

            return vec2<OverlapReal>(verts.x[max_idx], verts.y[max_idx]);
            }

    private:
        const poly2d_verts& verts;      //!< Vertices of the polygon
    };

}; // end namespace detail

//! Convex Polygon shape template
/*! ShapeConvexPolygon implements IntegratorHPMC's shape protocol. It serves at the simplest example of an orientable
    shape for HPMC.

    The parameter defining a polygon is a structure containing a list of N vertices. They are assumed to be listed
    in counter-clockwise order and centered on 0,0. In fact, it is **required** that the origin is inside the shape,
    and it is best if the origin is the center of mass.

    \ingroup shape
*/
struct ShapeConvexPolygon
    {
    //! Define the parameter type
    typedef detail::poly2d_verts param_type;

    //! Initialize a polygon
    DEVICE ShapeConvexPolygon(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() { return true; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const{ return verts.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-circle radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << verts.sweep_radius << ", \"vertices\": [";
        for (unsigned int i = 0; i < verts.N-1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
            }
        shapedef << "[" << verts.x[verts.N-1] << ", " << verts.y[verts.N-1] << "]]}";
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // generate a tight AABB
        // detail::SupportFuncConvexPolygon sfunc(verts);

        // // // use support function of the to determine the furthest extent in each direction
        // quat<OverlapReal> o(orientation);
        // vec2<OverlapReal> e_x(rotate(conj(o), vec2<OverlapReal>(1,0)));
        // vec2<OverlapReal> e_y(rotate(conj(o), vec2<OverlapReal>(0,1)));

        // vec2<OverlapReal> s_x = rotate(o, sfunc(e_x));
        // vec2<OverlapReal> s_y = rotate(o, sfunc(e_y));
        // vec2<OverlapReal> s_neg_x = rotate(o, sfunc(-e_x));
        // vec2<OverlapReal> s_neg_y = rotate(o, sfunc(-e_y));

        // // translate out from the position by the furthest extents
        // vec3<Scalar> upper(pos.x + s_x.x, pos.y + s_y.y, 0.1);
        // vec3<Scalar> lower(pos.x + s_neg_x.x, pos.y + s_neg_y.y, -0.1);

        // return detail::AABB(lower, upper);
        // ^^^^^^^^^^ The above method is slow, just use the bounding sphere
        return detail::AABB(pos, verts.diameter/Scalar(2));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the polygon

    const detail::poly2d_verts& verts;     //!< Vertices
    };

namespace detail
{

//! Test if all vertices in a polygon are outside of a given line
/*! \param verts Vertices of the polygon
    \param p Point on the line
    \param n Outward pointing normal
    \returns true when all vertices in the polygon are on the outside of the given line

    \note \a p and \a n are specified *in the polygon's reference frame!*

    \ingroup overlap
*/
DEVICE inline bool is_outside(const poly2d_verts& verts, const vec2<OverlapReal>& p, const vec2<OverlapReal>& n)
    {
    bool outside = true;

    // for each vertex in the polygon
    // check if n dot (v[i]-p) < 0
    // distribute: (n dot v[i] - n dot p) < 0
    OverlapReal ndotp = dot(n,p);
    #pragma unroll 3
    for (unsigned int i = 0; i < verts.N; i++)
        {
        if ((dot(n,vec2<OverlapReal>(verts.x[i], verts.y[i])) - ndotp) <= OverlapReal(0.0))
            {
            return false;       // runs faster on the cpu with an early return
            }
        }

    // if we get here, all points are outside
    return outside;
    }

//! Tests if any edge in a separates polygons a and b
/*! \param a First polygon
    \param b Second polygon
    \param ab_t Vector pointing from a's center to b's center, rotated by conj(qb) (see description for why)
    \param ab_r quaternion that rotates from *a*'s orientation into *b*'s.
    \returns true if any edge in *a* separates shapes *a* and *b*

    Shape *a* is at the origin. (in other words, we are solving this in the frame of *a*). Normal vectors can be rotated
    from the frame of a to b simply by rotating by ab_r, which is equal to conj(b.orientation) * a.orientation.
    This comes from the following

        - first, go from the frame of *a* into the space frame (qa))
        - then, go into back into the *b* frame (conj(qb))

    Transforming points from one frame into another takes a bit more care. The easiest way to think about it is this:

        - Rotate from the *a* frame into the space frame (rotate by *qa*).
        - Then translate into a frame with *b*'s origin at the center (subtract ab_t).
        - Then rotate into the *b* frame (rotate by conj(*qb*))

    Putting that all together, we get: \f$ q_b^* \cdot (q_a \vec{v} q_a^* - \vec{a}) a_b \f$. That's a lot of quats to
    store and a lot of rotations to do. Distributing gives \f$ q_b^* q_a \vec{v} q_a^* q_b - q_b^* \vec{a} q_b \f$.
    The first rotation is by the already computed ab_r! The 2nd only needs to be computed once

    \note Only edges in a are checked. This function must be called twice for a full separating planes overlap test

    \ingroup overlap
*/
DEVICE inline bool find_separating_plane(const poly2d_verts& a,
                                         const poly2d_verts& b,
                                         const vec2<OverlapReal>& ab_t,
                                         const quat<OverlapReal>& ab_r)
    {
    bool separating = false;

    rotmat2<OverlapReal> R(ab_r);

    // loop through all the edges in polygon a and check if they separate it from polygon b
    unsigned int prev = a.N-1;
    for (unsigned int cur = 0; cur < a.N; cur++)
        {
        // find a point and a vector describing a line
        vec2<OverlapReal> p = vec2<OverlapReal>(a.x[cur], a.y[cur]);
        vec2<OverlapReal> line = p - vec2<OverlapReal>(a.x[prev], a.y[prev]);

        // construct an outward pointing vector perpendicular to that line (assumes counter-clockwise ordering!)
        vec2<OverlapReal> n(line.y, -line.x);

        // transform into b's coordinate system
        // p = rotate(ab_r, p) - ab_t;
        // n = rotate(ab_r, n);
        p = R * p - ab_t;
        n = R * n;

        // is this a separating plane?
        if (is_outside(b, p, n))
            {
            return true;        // runs faster on the cpu with the early return
            }

        // save previous vertex for next iteration
        prev = cur;
        }

    // if we get here, there is no separating plane
    return separating;
    }

//! Test the overlap of two polygons via separating planes
/*! \param a First polygon
    \param b Second polygon
    \param ab_t Vector pointing from a's center to b's center, in the space frame
    \param qa Orientation of first polygon
    \param qb Orientation of second polygon
    \returns true when the two polygons overlap

    \pre Polygon vertices are in **counter-clockwise** order
    \pre The shape is convex and contains no internal vertices

    \ingroup overlap
*/
DEVICE inline bool test_overlap_separating_planes(const poly2d_verts& a,
                                                  const poly2d_verts& b,
                                                  const vec2<OverlapReal>& ab_t,
                                                  const quat<OverlapReal>& qa,
                                                  const quat<OverlapReal>& qb)
    {
    // construct a quaternion that rotates from a's coordinate system into b's
    quat<OverlapReal> ab_r = conj(qb) * qa;

    // see if we can find a separating plane from a's edges, or from b's edges, or else the shapes overlap
    if (find_separating_plane(a, b, rotate(conj(qb), ab_t), ab_r))
        return false;

    if (find_separating_plane(b, a, rotate(conj(qa), -ab_t), conj(ab_r)))
        return false;

    return true;
    }

}; // end namespace detail

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeConvexPolygon& a,
    const ShapeConvexPolygon &b)
    {
    vec2<OverlapReal> dr(r_ab.x, r_ab.y);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }


//! Convex polygon overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeConvexPolygon,ShapeConvexPolygon>(const vec3<Scalar>& r_ab,
                                                                       const ShapeConvexPolygon& a,
                                                                       const ShapeConvexPolygon& b,
                                                                       unsigned int& err)
    {
    vec2<OverlapReal> dr(r_ab.x,r_ab.y);
    #ifdef NVCC
    return detail::xenocollide_2d(detail::SupportFuncConvexPolygon(a.verts),
                                  detail::SupportFuncConvexPolygon(b.verts),
                                  dr,
                                  quat<OverlapReal>(a.orientation),
                                  quat<OverlapReal>(b.orientation),
                                  err);
    #else
    return detail::test_overlap_separating_planes(a.verts,
                                                  b.verts,
                                                  dr,
                                                  quat<OverlapReal>(a.orientation),
                                                  quat<OverlapReal>(b.orientation));
    #endif
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_CONVEX_POLYGON_H__
