// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "ShapeSphere.h" //< For the base template of test_overlap
#include "XenoCollide2D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#if !defined(__HIPCC__) && defined(__SSE__)
#include <immintrin.h>
#endif
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/** maximum number of vertices that can be stored (must be a multiple of 8)
    Note that vectorized methods using this struct assume unused coordinates are set to zero.
*/
const unsigned int MAX_POLY2D_VERTS = 64;

/** Polygon parameters

    Define the parameters of a polygon for HPMC shape overlap checks. Polygons are defined by N
    vertices in a counter-clockwise winding order. The polygon's diameter is precomputed from the
    vertex farthest from the origin. Polygons may have sweep radius greater than 0 which makes them
    rounded polygons (see ShapeSpheroPolygon). Coordinates are stored with x and y in separate
    arrays to support vector intrinsics on the CPU.

    These parameters are used in ShapeConvexPolygon, ShapeSimplePolygon, and ShapeSpheroPolygon.
*/
struct PolygonVertices : ShapeParams
    {
    /// X coordinates of the vertices
    ShortReal x[MAX_POLY2D_VERTS];

    /// Y coordinates of the vertice
    ShortReal y[MAX_POLY2D_VERTS];

    /// Number of vertices
    unsigned int N;

    /// Precomputed diameter
    ShortReal diameter;

    /// Rounding radius
    ShortReal sweep_radius;

    /// True when move statistics should not be counted
    unsigned int ignore;

    /// Default constructor initializes zero values.
    DEVICE PolygonVertices() : N(0), diameter(ShortReal(0)), sweep_radius(ShortReal(0)), ignore(0)
        {
        for (unsigned int i = 0; i < MAX_POLY2D_VERTS; i++)
            {
            x[i] = y[i] = ShortReal(0);
            }
        }

#ifdef ENABLE_HIP
    /// Set CUDA memory hint
    void set_memory_hint() const { }
#endif

#ifndef __HIPCC__

    /// Construct from a Python dictionary
    PolygonVertices(pybind11::dict v, bool managed = false)
        {
        pybind11::list verts = v["vertices"];
        if (len(verts) > MAX_POLY2D_VERTS)
            throw std::runtime_error("Too many polygon vertices");

        N = (unsigned int)len(verts);
        ignore = v["ignore_statistics"].cast<unsigned int>();
        sweep_radius = v["sweep_radius"].cast<ShortReal>();

        // extract the verts from the python list and compute the radius on the way
        ShortReal radius_sq = ShortReal(0.0);
        for (unsigned int i = 0; i < len(verts); i++)
            {
            pybind11::list verts_i = verts[i];
            if (len(verts_i) != 2)
                throw std::runtime_error("Each vertex must have 2 elements");

            vec2<ShortReal> vert = vec2<ShortReal>(pybind11::cast<ShortReal>(verts_i[0]),
                                                   pybind11::cast<ShortReal>(verts_i[1]));
            x[i] = vert.x;
            y[i] = vert.y;
            radius_sq = max(radius_sq, dot(vert, vert));
            }

        // zero memory for unused vertices
        for (unsigned int i = (unsigned int)len(verts); i < MAX_POLY2D_VERTS; i++)
            {
            x[i] = 0;
            y[i] = 0;
            }

        // set the diameter
        diameter = 2 * (sqrt(radius_sq) + sweep_radius);
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        pybind11::list verts;
        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::list vert;
            vert.append(x[i]);
            vert.append(y[i]);
            verts.append(pybind11::tuple(vert));
            }

        v["vertices"] = verts;
        v["ignore_statistics"] = ignore;
        v["sweep_radius"] = sweep_radius;
        return v;
        }

#endif
    } __attribute__((aligned(32)));

/** Support function for ShapeConvexPolygon

    SupportFuncConvexPolygon is a functor that computes the support function for ShapeConvexPolygon.
    For a given input vector in local coordinates, it finds the vertex most in that direction.

    @todo Make a minkowski namespace
*/
class SupportFuncConvexPolygon
    {
    public:
    /** Construct a support function for a convex polygon

        @param _verts Polygon vertices

        Note that for performance it is assumed that unused vertices (beyond N) have already
        been set to zero.
    */
    DEVICE SupportFuncConvexPolygon(const PolygonVertices& _verts) : verts(_verts) { }

    /** Compute the support function

        @param n Normal vector input (in the local frame)
        @returns Local coords of the point furthest in the direction of n
    */
    DEVICE vec2<ShortReal> operator()(const vec2<ShortReal>& n) const
        {
        ShortReal max_dot = -(verts.diameter * verts.diameter);
        unsigned int max_idx = 0;

        if (verts.N > 0)
            {
#if !defined(__HIPCC__) && defined(__AVX__) && HOOMD_SHORTREAL_SIZE == 32
            // process dot products with AVX 8 at a time on the CPU when working with more than 4
            // verts
            __m256 nx_v = _mm256_broadcast_ss(&n.x);
            __m256 ny_v = _mm256_broadcast_ss(&n.y);
            __m256 max_dot_v = _mm256_broadcast_ss(&max_dot);
            float d_s[MAX_POLY2D_VERTS] __attribute__((aligned(32)));

            for (size_t i = 0; i < verts.N; i += 8)
                {
                __m256 x_v = _mm256_load_ps(verts.x + i);
                __m256 y_v = _mm256_load_ps(verts.y + i);

                __m256 d_v = _mm256_add_ps(_mm256_mul_ps(nx_v, x_v), _mm256_mul_ps(ny_v, y_v));

                // determine a maximum in each of the 8 channels as we go
                max_dot_v = _mm256_max_ps(max_dot_v, d_v);
                _mm256_store_ps(d_s + i, d_v);
                }

            // find the maximum of the 8 channels
            // https://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
            max_dot_v
                = _mm256_max_ps(max_dot_v,
                                _mm256_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(2, 1, 0, 3)));
            max_dot_v
                = _mm256_max_ps(max_dot_v,
                                _mm256_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(1, 0, 3, 2)));
            // shuffles work only within the two 128b segments, so right now we have two separate
            // max values swap the left and right hand sides and max again to get the final max
            max_dot_v = _mm256_max_ps(max_dot_v, _mm256_permute2f128_ps(max_dot_v, max_dot_v, 1));

            // loop again and find the max. The reason this is in a 2nd loop is because branch
            // mis-predictions and the extra max calls kill performance if this is in the first loop
            // Use BSF to find the first index of the max element
            // https://software.intel.com/en-us/forums/topic/285956
            for (unsigned int i = 0; i < verts.N; i += 8)
                {
                __m256 d_v = _mm256_load_ps(d_s + i);

                int id = __builtin_ffs(_mm256_movemask_ps(_mm256_cmp_ps(max_dot_v, d_v, 0)));

                if (id)
                    {
                    max_idx = i + id - 1;
                    break;
                    }
                }

#elif !defined(__HIPCC__) && defined(__SSE__) && HOOMD_SHORTREAL_SIZE == 32
            // process dot products with SSE 4 at a time on the CPU
            __m128 nx_v = _mm_load_ps1(&n.x);
            __m128 ny_v = _mm_load_ps1(&n.y);
            __m128 max_dot_v = _mm_load_ps1(&max_dot);
            float d_s[MAX_POLY2D_VERTS] __attribute__((aligned(16)));

            for (unsigned int i = 0; i < verts.N; i += 4)
                {
                __m128 x_v = _mm_load_ps(verts.x + i);
                __m128 y_v = _mm_load_ps(verts.y + i);

                __m128 d_v = _mm_add_ps(_mm_mul_ps(nx_v, x_v), _mm_mul_ps(ny_v, y_v));

                // determine a maximum in each of the 4 channels as we go
                max_dot_v = _mm_max_ps(max_dot_v, d_v);

                _mm_store_ps(d_s + i, d_v);
                }

            // find the maximum of the 4 channels
            // https://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
            max_dot_v = _mm_max_ps(max_dot_v,
                                   _mm_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(2, 1, 0, 3)));
            max_dot_v = _mm_max_ps(max_dot_v,
                                   _mm_shuffle_ps(max_dot_v, max_dot_v, _MM_SHUFFLE(1, 0, 3, 2)));

            // loop again and find the max. The reason this is in a 2nd loop is because branch
            // mis-predictions and the extra max calls kill performance if this is in the first loop
            // Use BSF to find the first index of the max element
            // https://software.intel.com/en-us/forums/topic/285956
            for (unsigned int i = 0; i < verts.N; i += 4)
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

            // implementation without vector intrinsics
            ShortReal max_dot0 = dot(n, vec2<ShortReal>(verts.x[0], verts.y[0]));
            unsigned int max_idx0 = 0;
            ShortReal max_dot1 = dot(n, vec2<ShortReal>(verts.x[1], verts.y[1]));
            unsigned int max_idx1 = 1;
            ShortReal max_dot2 = dot(n, vec2<ShortReal>(verts.x[2], verts.y[2]));
            unsigned int max_idx2 = 2;
            ShortReal max_dot3 = dot(n, vec2<ShortReal>(verts.x[3], verts.y[3]));
            unsigned int max_idx3 = 3;

            for (unsigned int i = 4; i < verts.N; i += 4)
                {
                ShortReal d0 = dot(n, vec2<ShortReal>(verts.x[i], verts.y[i]));
                ShortReal d1 = dot(n, vec2<ShortReal>(verts.x[i + 1], verts.y[i + 1]));
                ShortReal d2 = dot(n, vec2<ShortReal>(verts.x[i + 2], verts.y[i + 2]));
                ShortReal d3 = dot(n, vec2<ShortReal>(verts.x[i + 3], verts.y[i + 3]));

                if (d0 > max_dot0)
                    {
                    max_dot0 = d0;
                    max_idx0 = i;
                    }
                if (d1 > max_dot1)
                    {
                    max_dot1 = d1;
                    max_idx1 = i + 1;
                    }
                if (d2 > max_dot2)
                    {
                    max_dot2 = d2;
                    max_idx2 = i + 2;
                    }
                if (d3 > max_dot3)
                    {
                    max_dot3 = d3;
                    max_idx3 = i + 3;
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

        return vec2<ShortReal>(verts.x[max_idx], verts.y[max_idx]);
        }

    private:
    /// Vertices of the polygon
    const PolygonVertices& verts;
    };

    }; // end namespace detail

/** Convex Polygon shape

    Implement the HPMC shape interface for convex polygons.
*/
struct ShapeConvexPolygon
    {
    /// Define the parameter type
    typedef detail::PolygonVertices param_type;

    //! Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeConvexPolygon(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation()
        {
        return true;
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return verts.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // Generate the AABB of a bounding sphere, computing tight fitting AABBs is slow.
        return hoomd::detail::AABB(pos, verts.diameter / Scalar(2));
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
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
    const detail::PolygonVertices& verts;
    };

namespace detail
    {
/** Test if all vertices in a polygon are outside of a given line

    @param verts Vertices of the polygon.
    @param p Point on the line.
    @param n Outward pointing normal.
    @returns true when all vertices in the polygon are on the outside of the given line.

    `p` and `n` are specified *in the polygon's reference frame!*

    \todo make overlap check namespace
*/
DEVICE inline bool
is_outside(const PolygonVertices& verts, const vec2<ShortReal>& p, const vec2<ShortReal>& n)
    {
    bool outside = true;

    // for each vertex in the polygon
    // check if n dot (v[i]-p) < 0
    // distribute: (n dot v[i] - n dot p) < 0
    ShortReal ndotp = dot(n, p);
#pragma unroll 3
    for (unsigned int i = 0; i < verts.N; i++)
        {
        if ((dot(n, vec2<ShortReal>(verts.x[i], verts.y[i])) - ndotp) <= ShortReal(0.0))
            {
            return false; // runs faster on the cpu with an early return
            }
        }

    // if we get here, all points are outside
    return outside;
    }

/** Tests if any edge in a separates polygons a and b

    @param a First polygon
    @param b Second polygon
    @param ab_t Vector pointing from a's center to b's center, rotated by conj(qb)
                (see description for why)
    @param ab_r quaternion that rotates from *a*'s orientation into *b*'s.
    @returns true if any edge in *a* separates shapes *a* and *b*

    Shape *a* is at the origin. (in other words, we are solving this in the frame of *a*). Normal
    vectors can be rotated from the frame of *a* to *b* simply by rotating by *ab_r*, which is equal
    to `conj(b.orientation) * a.orientation`. This comes from the following

    - first, go from the frame of *a* into the space frame (`qa`))
    - then, go into back into the *b* frame (`conj(qb)`)

    Transforming points from one frame into another takes a bit more care. The easiest way to think
    about it is this:

    - Rotate from the *a* frame into the space frame (rotate by `qa`).
    - Then translate into a frame with *b*'s origin at the center (subtract `ab_t`).
    - Then rotate into the *b* frame (rotate by `conj(qb)`)

    Putting that all together, we get: \f$ q_b^* \cdot (q_a \vec{v} q_a^* - \vec{a}) a_b \f$. That's
    a lot of quats to store and a lot of rotations to do. Distributing gives \f$ q_b^* q_a \vec{v}
    q_a^* q_b - q_b^* \vec{a} q_b \f$. The first rotation is by the already computed `ab_r`! The 2nd
    only needs to be computed once

    @note Only edges in *a* are checked. This function must be called twice for a full separating
    planes overlap test
*/
DEVICE inline bool find_separating_plane(const PolygonVertices& a,
                                         const PolygonVertices& b,
                                         const vec2<ShortReal>& ab_t,
                                         const quat<ShortReal>& ab_r)
    {
    bool separating = false;

    rotmat2<ShortReal> R(ab_r);

    // loop through all the edges in polygon a and check if they separate it from polygon b
    unsigned int prev = a.N - 1;
    for (unsigned int cur = 0; cur < a.N; cur++)
        {
        // find a point and a vector describing a line
        vec2<ShortReal> p = vec2<ShortReal>(a.x[cur], a.y[cur]);
        vec2<ShortReal> line = p - vec2<ShortReal>(a.x[prev], a.y[prev]);

        // construct an outward pointing vector perpendicular to that line (assumes
        // counter-clockwise ordering!)
        vec2<ShortReal> n(line.y, -line.x);

        // transform into b's coordinate system
        // p = rotate(ab_r, p) - ab_t;
        // n = rotate(ab_r, n);
        p = R * p - ab_t;
        n = R * n;

        // is this a separating plane?
        if (is_outside(b, p, n))
            {
            return true; // runs faster on the cpu with the early return
            }

        // save previous vertex for next iteration
        prev = cur;
        }

    // if we get here, there is no separating plane
    return separating;
    }

/** Test the overlap of two polygons via separating planes

    @param a First polygon
    @param b Second polygon
    @param ab_t Vector pointing from *a*'s center to *b*'s center, in the space frame
    @param qa Orientation of first polygon
    @param qb Orientation of second polygon
    @returns true when the two polygons overlap

    @pre Polygon vertices are in **counter-clockwise** order
    @pre The shape is convex and contains no internal vertices
f*/
DEVICE inline bool test_overlap_separating_planes(const PolygonVertices& a,
                                                  const PolygonVertices& b,
                                                  const vec2<ShortReal>& ab_t,
                                                  const quat<ShortReal>& qa,
                                                  const quat<ShortReal>& qb)
    {
    // construct a quaternion that rotates from a's coordinate system into b's
    quat<ShortReal> ab_r = conj(qb) * qa;

    // see if we can find a separating plane from a's edges, or from b's edges, or else the shapes
    // overlap
    if (find_separating_plane(a, b, rotate(conj(qb), ab_t), ab_r))
        return false;

    if (find_separating_plane(b, a, rotate(conj(qa), -ab_t), conj(ab_r)))
        return false;

    return true;
    }

    }; // end namespace detail

/** Convex polygon overlap test

    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a first shape
    @param b second shape
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius Additional radius to sweep both shapes by
    @returns *true* when *a* and *b* overlap, and false when they are disjoint
*/
template<>
DEVICE inline bool test_overlap<ShapeConvexPolygon, ShapeConvexPolygon>(const vec3<Scalar>& r_ab,
                                                                        const ShapeConvexPolygon& a,
                                                                        const ShapeConvexPolygon& b,
                                                                        unsigned int& err)
    {
    vec2<ShortReal> dr(ShortReal(r_ab.x), ShortReal(r_ab.y));
#ifdef __HIPCC__
    return detail::xenocollide_2d(detail::SupportFuncConvexPolygon(a.verts),
                                  detail::SupportFuncConvexPolygon(b.verts),
                                  dr,
                                  quat<ShortReal>(a.orientation),
                                  quat<ShortReal>(b.orientation),
                                  err);
#else
    return detail::test_overlap_separating_planes(a.verts,
                                                  b.verts,
                                                  dr,
                                                  quat<ShortReal>(a.orientation),
                                                  quat<ShortReal>(b.orientation));
#endif
    }

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeConvexPolygon& poly)
    {
    std::ostringstream shapedef;
    const auto& verts = poly.verts;
    shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << verts.sweep_radius
             << ", \"vertices\": [";
    if (verts.N != 0)
        {
        for (unsigned int i = 0; i < verts.N - 1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
            }
        shapedef << "[" << verts.x[verts.N - 1] << ", " << verts.y[verts.N - 1] << "]]}";
        }
    else
        {
        shapedef << "[0, 0]]}";
        }
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd
#undef DEVICE
#undef HOSTDEVICE
