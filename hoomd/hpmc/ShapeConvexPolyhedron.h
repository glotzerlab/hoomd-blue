// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "ShapeSphere.h" //< For the base template of test_overlap
#include "XenoCollide3D.h"
#include "XenoSweep3D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/OBB.h"

#include <cfloat>

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
/** Convex polyhedron vertices

    Define the parameters of a convex polyhedron for HPMC shape overlap checks. Convex polyhedra are
    defined by the convex hull of N vertices. The polyhedrons's diameter is precomputed from the
    vertex farthest from the origin. Convex polyhedra may have sweep radius greater than 0 which
    makes them rounded convex polyhedra. Coordinates are stored with x, y, and z in separate arrays
    to support vector intrinsics on the CPU. These arrays are stored in ManagedArray to support
    arbitrary numbers of verticles.
*/
struct PolyhedronVertices : ShapeParams
    {
    static constexpr unsigned int MAX_VERTS = 4096;
    
    /// Default constructor initializes zero values.
    DEVICE PolyhedronVertices()
        : n_hull_verts(0), N(0), diameter(ShortReal(0)), sweep_radius(ShortReal(0)), ignore(0)
        {
        }

#ifndef __HIPCC__
    /** Initialize with a given number of vertices
     */
    PolyhedronVertices(unsigned int _N, bool managed = false) : ignore(0)
        {
        std::vector<vec3<ShortReal>> v(_N, vec3<ShortReal>(0, 0, 0));
        setVerts(v, 0, managed);
        }

    PolyhedronVertices(const std::vector<vec3<ShortReal>>& verts,
                       ShortReal sweep_radius_,
                       unsigned int ignore_,
                       bool managed = false)
        : x((unsigned int)verts.size(), managed), y((unsigned int)verts.size(), managed),
          z((unsigned int)verts.size(), managed), n_hull_verts(0), N((unsigned int)verts.size()),
          diameter(0.0), sweep_radius(sweep_radius_), ignore(ignore_)
        {
        setVerts(verts, sweep_radius_, managed);
        }

    /** Set the shape vertices

        Sets new vertices for the shape. Maintains the same managed memory state.

        @param verts Vertices to set
        @param sweep_radius_ Sweep radius
    */
    void setVerts(const std::vector<vec3<ShortReal>>& verts,
                  ShortReal sweep_radius_,
                  bool managed = false)
        {
        N = (unsigned int)verts.size();

        if (N > MAX_VERTS)
            {
            throw std::domain_error("Too many vertices in polyhedron.");
            }
        
        diameter = 0;
        sweep_radius = sweep_radius_;

        unsigned int align_size = 8; // for AVX
        unsigned int N_align = ((N + align_size - 1) / align_size) * align_size;
        x = ManagedArray<ShortReal>(N_align, managed, 32); // 32byte alignment for AVX
        y = ManagedArray<ShortReal>(N_align, managed, 32);
        z = ManagedArray<ShortReal>(N_align, managed, 32);
        for (unsigned int i = 0; i < N_align; ++i)
            {
            x[i] = y[i] = z[i] = ShortReal(0.0);
            }

        // copy the verts over from the std vector and compute the radius on the way
        ShortReal radius_sq = ShortReal(0.0);
        for (unsigned int i = 0; i < verts.size(); i++)
            {
            const auto& vert = verts[i];
            x[i] = vert.x;
            y[i] = vert.y;
            z[i] = vert.z;
            radius_sq = max(radius_sq, dot(vert, vert));
            }
        // set the diameter
        diameter = 2 * (sqrt(radius_sq) + sweep_radius);

        if (N >= 3)
            {
            // compute convex hull of vertices
            std::vector<quickhull::Vector3<ShortReal>> qh_pts;
            for (unsigned int i = 0; i < N; i++)
                {
                quickhull::Vector3<ShortReal> vert;
                vert.x = x[i];
                vert.y = y[i];
                vert.z = z[i];
                qh_pts.push_back(vert);
                }

            quickhull::QuickHull<ShortReal> qh;
            // argument 2: CCW orientation of triangles viewed from outside
            // argument 3: use existing vertex list
            auto hull = qh.getConvexHull(qh_pts, false, true);
            auto indexBuffer = hull.getIndexBuffer();

            hull_verts = ManagedArray<unsigned int>((unsigned int)indexBuffer.size(), managed);
            n_hull_verts = (unsigned int)indexBuffer.size();

            for (unsigned int i = 0; i < indexBuffer.size(); i++)
                hull_verts[i] = (unsigned int)indexBuffer[i];
            }

        if (N >= 1)
            {
            std::vector<ShortReal> vertex_radii(N, sweep_radius);
            std::vector<vec3<ShortReal>> pts(N);
            for (unsigned int i = 0; i < N; ++i)
                pts[i] = vec3<ShortReal>(x[i], y[i], z[i]);

            obb = detail::compute_obb(pts, vertex_radii, managed);
            }
        }

    /// Construct from a Python dictionary
    PolyhedronVertices(pybind11::dict v, bool managed = false)
        : PolyhedronVertices((unsigned int)pybind11::len(v["vertices"]), managed)
        {
        pybind11::list verts = v["vertices"];
        ignore = v["ignore_statistics"].cast<unsigned int>();

        // extract the verts from the python list
        std::vector<vec3<ShortReal>> vert_vector;
        for (unsigned int i = 0; i < pybind11::len(verts); i++)
            {
            pybind11::list verts_i = verts[i];
            if (len(verts_i) != 3)
                throw std::runtime_error("Each vertex must have 3 elements");
            vec3<ShortReal> vert = vec3<ShortReal>(pybind11::cast<ShortReal>(verts_i[0]),
                                                   pybind11::cast<ShortReal>(verts_i[1]),
                                                   pybind11::cast<ShortReal>(verts_i[2]));
            vert_vector.push_back(vert);
            }

        setVerts(vert_vector, v["sweep_radius"].cast<float>(), managed);
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
            vert.append(z[i]);
            pybind11::tuple vert_tuple = pybind11::tuple(vert);
            verts.append(vert_tuple);
            }
        v["vertices"] = verts;
        v["ignore_statistics"] = ignore;
        v["sweep_radius"] = sweep_radius;
        return v;
        }

#endif

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
        {
        x.load_shared(ptr, available_bytes);
        y.load_shared(ptr, available_bytes);
        z.load_shared(ptr, available_bytes);
        hull_verts.load_shared(ptr, available_bytes);
        }

    /** Determine size of the shared memory allocation

        @param ptr Pointer to increment
        @param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        x.allocate_shared(ptr, available_bytes);
        y.allocate_shared(ptr, available_bytes);
        z.allocate_shared(ptr, available_bytes);
        hull_verts.allocate_shared(ptr, available_bytes);
        }

#ifdef ENABLE_HIP
    void set_memory_hint() const
        {
        x.set_memory_hint();
        y.set_memory_hint();
        z.set_memory_hint();
        hull_verts.set_memory_hint();
        }
#endif

    /// X coordinate of vertices
    ManagedArray<ShortReal> x;

    /// Y coordinate of vertices
    ManagedArray<ShortReal> y;

    /// Z coordinate of vertices
    ManagedArray<ShortReal> z;

    /// TODO: store aligned copy of d here for use in the support function.

    /** List of triangles hull_verts[3*i], hull_verts[3*i+1], hull_verts[3*i+2] making up the convex
        hull
    */
    ManagedArray<unsigned int> hull_verts;

    /// Number of vertices in the convex hull
    unsigned int n_hull_verts;

    /// Number of vertices
    unsigned int N;

    /// Circumsphere diameter
    ShortReal diameter;

    /// Radius of the sphere sweep (used for spheropolyhedra)
    ShortReal sweep_radius;

    /// True when move statistics should not be counted
    unsigned int ignore;

    /// Tight fitting bounding box
    detail::OBB obb;
    } __attribute__((aligned(32)));

/** Support function for ShapePolyhedron

    SupportFuncPolyhedron is a functor that computes the support function for ShapePolyhedron. For a
    given input vector in local coordinates, it finds the vertex most in that direction.
*/
class SupportFuncConvexPolyhedron
    {
    public:
    /** Construct a support function for a convex polyhedron

        @param _verts Polyhedron vertices

        Note that for performance it is assumed that unused vertices (beyond N) have already
        been set to zero.
    */
    DEVICE inline SupportFuncConvexPolyhedron(const PolyhedronVertices& _verts,
                                       ShortReal extra_sweep_radius = ShortReal(0.0))
        : verts(_verts), sweep_radius(extra_sweep_radius)
        {
        }

    /** Compute the support function

        @param n Normal vector input (in the local frame)
        @returns Local coords of the point furthest in the direction of n
    */
    DEVICE inline __attribute__((always_inline)) vec3<ShortReal> operator()(const vec3<ShortReal>& n) const
        {
        ShortReal max_dot = -(verts.diameter * verts.diameter);
        unsigned int max_idx = 0;

        if (verts.N > 0)
            {
#if !defined(__HIPCC__) && defined(__AVX__) && HOOMD_SHORTREAL_SIZE == 32
            // process dot products with AVX 8 at a time on the CPU when working with more than
            // 4 verts
            __m256 nx_v = _mm256_broadcast_ss(&n.x);
            __m256 ny_v = _mm256_broadcast_ss(&n.y);
            __m256 nz_v = _mm256_broadcast_ss(&n.z);
            __m256 max_dot_v = _mm256_broadcast_ss(&max_dot);
            float d_s[PolyhedronVertices::MAX_VERTS] __attribute__((aligned(32)));

            for (unsigned int i = 0; i < verts.N; i += 8)
                {
                __m256 x_v = _mm256_load_ps(verts.x.get() + i);
                __m256 y_v = _mm256_load_ps(verts.y.get() + i);
                __m256 z_v = _mm256_load_ps(verts.z.get() + i);

                __m256 d_v = _mm256_add_ps(
                    _mm256_mul_ps(nx_v, x_v),
                    _mm256_add_ps(_mm256_mul_ps(ny_v, y_v), _mm256_mul_ps(nz_v, z_v)));

                // determine a maximum in each of the 8 channels as we go
                max_dot_v = _mm256_max_ps(max_dot_v, d_v);

                _mm256_store_ps(d_s + i, d_v);
                }

            // find the maximum of the 8 channels
            // http://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
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
            __m128 nz_v = _mm_load_ps1(&n.z);
            __m128 max_dot_v = _mm_load_ps1(&max_dot);
            float d_s[PolyhedronVertices::MAX_VERTS] __attribute__((aligned(16)));

            for (unsigned int i = 0; i < verts.N; i += 4)
                {
                __m128 x_v = _mm_load_ps(verts.x.get() + i);
                __m128 y_v = _mm_load_ps(verts.y.get() + i);
                __m128 z_v = _mm_load_ps(verts.z.get() + i);

                __m128 d_v = _mm_add_ps(_mm_mul_ps(nx_v, x_v),
                                        _mm_add_ps(_mm_mul_ps(ny_v, y_v), _mm_mul_ps(nz_v, z_v)));

                // determine a maximum in each of the 4 channels as we go
                max_dot_v = _mm_max_ps(max_dot_v, d_v);

                _mm_store_ps(d_s + i, d_v);
                }

            // find the maximum of the 4 channels
            // http://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
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

            // if no AVX or SSE, or running in double precision, fall back on serial computation
            // this code path also triggers on the GPU

            ShortReal max_dot0 = dot(n, vec3<ShortReal>(verts.x[0], verts.y[0], verts.z[0]));
            unsigned int max_idx0 = 0;
            ShortReal max_dot1 = dot(n, vec3<ShortReal>(verts.x[1], verts.y[1], verts.z[1]));
            unsigned int max_idx1 = 1;
            ShortReal max_dot2 = dot(n, vec3<ShortReal>(verts.x[2], verts.y[2], verts.z[2]));
            unsigned int max_idx2 = 2;
            ShortReal max_dot3 = dot(n, vec3<ShortReal>(verts.x[3], verts.y[3], verts.z[3]));
            unsigned int max_idx3 = 3;

            for (unsigned int i = 4; i < verts.N; i += 4)
                {
                const ShortReal* verts_x = verts.x.get() + i;
                const ShortReal* verts_y = verts.y.get() + i;
                const ShortReal* verts_z = verts.z.get() + i;
                ShortReal d0 = dot(n, vec3<ShortReal>(verts_x[0], verts_y[0], verts_z[0]));
                ShortReal d1 = dot(n, vec3<ShortReal>(verts_x[1], verts_y[1], verts_z[1]));
                ShortReal d2 = dot(n, vec3<ShortReal>(verts_x[2], verts_y[2], verts_z[2]));
                ShortReal d3 = dot(n, vec3<ShortReal>(verts_x[3], verts_y[3], verts_z[3]));

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

            vec3<ShortReal> v(verts.x[max_idx], verts.y[max_idx], verts.z[max_idx]);
            if (sweep_radius != ShortReal(0.0))
                return v + (sweep_radius * fast::rsqrt(dot(n, n))) * n;
            else
                return v;
            } // end if(verts.N > 0)
        else
            {
            if (sweep_radius != ShortReal(0.0))
                return (sweep_radius * fast::rsqrt(dot(n, n))) * n;
            else
                return vec3<ShortReal>(0.0, 0.0, 0.0); // No verts!
            }
        }

    private:
    const PolyhedronVertices& verts; //!< Vertices of the polyhedron
    const ShortReal sweep_radius;    //!< Extra sweep radius
    };

/** Geometric primitives for closest point calculation

    From Real Time Collision Detection (Christer Ericson)
    https://doi.org/10.1201/b14581
*/
DEVICE inline vec3<ShortReal> closestPointOnTriangle(const vec3<ShortReal>& p,
                                                     const vec3<ShortReal>& a,
                                                     const vec3<ShortReal>& b,
                                                     const vec3<ShortReal>& c)
    {
    vec3<ShortReal> ab = b - a;
    vec3<ShortReal> ac = c - a;
    vec3<ShortReal> ap = p - a;

    ShortReal d1 = dot(ab, ap);
    ShortReal d2 = dot(ac, ap);
    if (d1 <= ShortReal(0.0) && d2 <= ShortReal(0.0))
        return a; // barycentric coordinates (1,0,0)

    // Check if P in vertex region outside B
    vec3<ShortReal> bp = p - b;
    ShortReal d3 = dot(ab, bp);
    ShortReal d4 = dot(ac, bp);
    if (d3 >= ShortReal(0.0) && d4 <= d3)
        return b; // barycentric coordinates (0,1,0)

    // Check if P in edge region of AB, if so return projection of P onto AB
    ShortReal vc = d1 * d4 - d3 * d2;
    if (vc <= ShortReal(0.0) && d1 >= ShortReal(0.0) && d3 <= ShortReal(0.0))
        {
        ShortReal v = d1 / (d1 - d3);
        return a + v * ab; // barycentric coordinates (1-v,v,0)
        }

    // Check if P in vertex region outside C
    vec3<ShortReal> cp = p - c;
    ShortReal d5 = dot(ab, cp);
    ShortReal d6 = dot(ac, cp);
    if (d6 >= ShortReal(0.0) && d5 <= d6)
        return c; // barycentric coordinates (0,0,1)

    // Check if P in edge region of AC, if so return projection of P onto AC
    ShortReal vb = d5 * d2 - d1 * d6;
    if (vb <= ShortReal(0.0) && d2 >= ShortReal(0.0) && d6 <= ShortReal(0.0))
        {
        ShortReal w = d2 / (d2 - d6);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
        }
    // Check if P in edge region of BC, if so return projection of P onto BC
    ShortReal va = d3 * d6 - d5 * d4;
    if (va <= ShortReal(0.0) && (d4 - d3) >= ShortReal(0.0) && (d5 - d6) >= ShortReal(0.0))
        {
        ShortReal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
        }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    ShortReal denom = ShortReal(1.0) / (va + vb + vc);
    ShortReal v = vb * denom;
    ShortReal w = vc * denom;
    return a + ab * v + ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
    }

/// Test if point p lies outside plane through abc
DEVICE inline bool PointOutsideOfPlane(const vec3<ShortReal>& p,
                                       const vec3<ShortReal>& a,
                                       const vec3<ShortReal>& b,
                                       const vec3<ShortReal>& c)
    {
    return dot(p - a, cross(b - a, c - a)) >= ShortReal(0.0);
    }

/// Find the point on a segment closest to point p
DEVICE inline vec3<ShortReal> ClosestPtPointSegment(const vec3<ShortReal>& c,
                                                    const vec3<ShortReal>& a,
                                                    const vec3<ShortReal>& b,
                                                    ShortReal& t)
    {
    vec3<ShortReal> ab = b - a;

    vec3<ShortReal> d;

    // Project c onto ab, but deferring divide by dot(ab,ab)
    t = dot(c - a, ab);

    if (t <= ShortReal(0.0))
        {
        // c projects outside the [a,b] interval, on the a side; clamp to a
        t = ShortReal(0.0);
        d = a;
        }
    else
        {
        ShortReal denom = dot(ab, ab);
        if (t >= denom)
            {
            // c project outside the [a,b] interval, on the b side; clamp to b
            t = ShortReal(1.0);
            d = b;
            }
        else
            {
            // c projects inside the [a,b] interval' must do deferred divide now
            t = t / denom;
            d = a + t * ab;
            }
        }

    return d;
    }

/** Projection function for ShapeConvexPolyhedron

    ProjectionFuncConvexPolyhedron is a functor that computes the projection function for
    ShapePolyhedron. For a given input point in local coordinates, it finds the vertex closest to
    that point.
*/
class ProjectionFuncConvexPolyhedron
    {
    public:
    /** Construct a projection function for a convex polyhedron

        @param _verts Polyhedron vertices
    */
    DEVICE ProjectionFuncConvexPolyhedron(const PolyhedronVertices& _verts,
                                          ShortReal extra_sweep_radius = ShortReal(0.0))
        : verts(_verts), sweep_radius(extra_sweep_radius)
        {
        }

    /** Compute the projection

        @param p Point to compute the projection for
        @returns Local coords of the point in the shape closest to p
    */
    DEVICE vec3<ShortReal> operator()(const vec3<ShortReal>& p) const
        {
        // Find the point on the convex hull closest to p
        vec3<ShortReal> closest_p = p;
        ShortReal closest_dsq(FLT_MAX);

        // iterate over triangles of convex hull to find closest point on every face
        unsigned int n_hull_verts = verts.n_hull_verts;

        if (n_hull_verts > 0)
            {
            for (unsigned int i = 0; i < n_hull_verts; i += 3)
                {
                unsigned int k = verts.hull_verts[i];
                unsigned int l = verts.hull_verts[i + 1];
                unsigned int m = verts.hull_verts[i + 2];

                vec3<ShortReal> a(verts.x[k], verts.y[k], verts.z[k]);
                vec3<ShortReal> b(verts.x[l], verts.y[l], verts.z[l]);
                vec3<ShortReal> c(verts.x[m], verts.y[m], verts.z[m]);

                // is the point on the outside of the plane?
                /* For this to work correctly in the degenerate case, i.e. planar facet and 1d line,
                   we require that the convex hull always is a complete mesh
                 */
                if (PointOutsideOfPlane(p, a, b, c))
                    {
                    vec3<ShortReal> q = closestPointOnTriangle(p, a, b, c);
                    ShortReal dsq = dot(p - q, p - q);

                    if (dsq < closest_dsq)
                        {
                        closest_p = q;
                        closest_dsq = dsq;
                        }
                    }
                }
            }
        else
            {
            // handle special cases
            if (verts.N == 0)
                {
                // return the origin;
                closest_p = vec3<ShortReal>(0, 0, 0);
                }
            else if (verts.N == 1)
                {
                // return the only point there is
                closest_p = vec3<ShortReal>(verts.x[0], verts.y[0], verts.z[0]);
                }
            else if (verts.N == 2)
                {
                // line segment
                ShortReal t;
                closest_p
                    = ClosestPtPointSegment(p,
                                            vec3<ShortReal>(verts.x[0], verts.y[0], verts.z[0]),
                                            vec3<ShortReal>(verts.x[1], verts.y[1], verts.z[1]),
                                            t);
                }
            }

        if (sweep_radius != 0.0 && (p.x != closest_p.x || p.y != closest_p.y || p.z != closest_p.z))
            {
            // point is on the surface, see if we have to project further out
            vec3<ShortReal> del = p - closest_p;
            ShortReal dsq = dot(del, del);
            if (dsq > sweep_radius * sweep_radius)
                {
                // add the sphere radius in direction of closest approach, or the closest point
                // inside the sphere
                ShortReal d = fast::sqrt(dsq);
                return closest_p + sweep_radius / d * del;
                }
            else
                return p;
            }

        // pt is inside base shape
        return closest_p;
        }

    private:
    const PolyhedronVertices& verts; //!< Vertices of the polyhedron
    const ShortReal sweep_radius;    //!< extra sphere sweep radius
    };

    }; // end namespace detail

/** Convex Polyhedron shape

    Implement the HPMC shape interface for convex polyhedra.
*/
struct ShapeConvexPolyhedron
    {
    /// Define the parameter type
    typedef detail::PolyhedronVertices param_type;

    //! Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeConvexPolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation() const
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
        return hoomd::detail::AABB(pos, getCircumsphereDiameter() / Scalar(2));
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        detail::OBB obb = verts.obb;
        obb.affineTransform(orientation, pos);
        return obb;
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
        return true;
        }

    /// Orientation of the shape
    quat<Scalar> orientation;

    /// Shape parameters
    const detail::PolyhedronVertices& verts;
    };

/** Convex polyhedron overlap test

    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a first shape
    @param b second shape
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius_a Radius of a sphere to sweep shape a by
    @param sweep_radius_b Radius of a sphere to sweep shape b by
    @returns true when *a* and *b* overlap, and false when they are disjoint
*/
template<>
DEVICE inline __attribute__((always_inline)) bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeConvexPolyhedron& a,
                                const ShapeConvexPolyhedron& b,
                                unsigned int& err)
    {
    vec3<ShortReal> dr(r_ab);

    ShortReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    return detail::xenocollide_3d(detail::SupportFuncConvexPolyhedron(a.verts),
                                  detail::SupportFuncConvexPolyhedron(b.verts),
                                  rotate(conj(quat<ShortReal>(a.orientation)), dr),
                                  conj(quat<ShortReal>(a.orientation))
                                      * quat<ShortReal>(b.orientation),
                                  DaDb / ShortReal(2.0),
                                  err);

    /*
    return detail::gjke_3d(detail::SupportFuncConvexPolyhedron(a.verts),
                           detail::SupportFuncConvexPolyhedron(b.verts),
                           vec3<Scalar>(dr.x, dr.y, dr.z),
                           a.orientation,
                           b.orientation,
                           DaDb/2.0,
                           err);
    */
    }

//! Convex polyhedron sweep distance
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
DEVICE inline ShortReal sweep_distance(const vec3<Scalar>& r_ab,
                                       const ShapeConvexPolyhedron& a,
                                       const ShapeConvexPolyhedron& b,
                                       const vec3<Scalar>& direction,
                                       unsigned int& err,
                                       vec3<Scalar>& collisionPlaneVector)
    {
    vec3<ShortReal> dr(r_ab);
    vec3<ShortReal> to(direction);

    vec3<ShortReal> csp(collisionPlaneVector);

    ShortReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    ShortReal distance = detail::xenosweep_3d(detail::SupportFuncConvexPolyhedron(a.verts),
                                              detail::SupportFuncConvexPolyhedron(b.verts),
                                              rotate(conj(quat<ShortReal>(a.orientation)), dr),
                                              conj(quat<ShortReal>(a.orientation))
                                                  * quat<ShortReal>(b.orientation),
                                              rotate(conj(quat<ShortReal>(a.orientation)), to),
                                              DaDb / ShortReal(2.0),
                                              err,
                                              csp);
    collisionPlaneVector = vec3<Scalar>(rotate(quat<ShortReal>(a.orientation), csp));

    return distance;
    }

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeConvexPolyhedron& poly)
    {
    std::ostringstream shapedef;
    const auto& verts = poly.verts;
    shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": " << verts.sweep_radius
             << ", \"vertices\": [";
    if (verts.N != 0)
        {
        for (unsigned int i = 0; i < verts.N - 1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << ", " << verts.z[i] << "], ";
            }
        shapedef << "[" << verts.x[verts.N - 1] << ", " << verts.y[verts.N - 1] << ", "
                 << verts.z[verts.N - 1] << "]]}";
        }
    else
        {
        shapedef << "[0, 0, 0]]}";
        }
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
