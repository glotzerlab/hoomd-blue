// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "XenoCollide3D.h"
#include "MAP3D.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/hpmc/OBB.h"

#include <cfloat>

#ifndef __SHAPE_CONVEX_POLYHEDRON_H__
#define __SHAPE_CONVEX_POLYHEDRON_H__

/*! \file ShapeConvexPolyhedron.h
    \brief Defines the convex polyhedron shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
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

//! maximum number of vertices that can be stored (must be multiple of 8)
/*! \ingroup hpmc_data_structs */

//! Data structure for polyhedron vertices
//! Note that vectorized methods using this struct will assume unused coordinates are set to zero.
/*! \ingroup hpmc_data_structs */
struct poly3d_verts : param_base

    {
    //! Default constructor initializes zero values.
    DEVICE poly3d_verts()
        : n_hull_verts(0),
          N(0),
          diameter(OverlapReal(0)),
          sweep_radius(OverlapReal(0)),
          ignore(0)
        { }

    #ifndef __HIPCC__
    //! Shape constructor
    poly3d_verts(unsigned int _N, bool _managed)
        : n_hull_verts(0), N(_N), diameter(0.0), sweep_radius(0.0), ignore(0)
        {
        unsigned int align_size = 8; //for AVX
        unsigned int N_align =((N + align_size - 1)/align_size)*align_size;
        x = ManagedArray<OverlapReal>(N_align,_managed, 32); // 32byte alignment for AVX
        y = ManagedArray<OverlapReal>(N_align,_managed, 32);
        z = ManagedArray<OverlapReal>(N_align,_managed, 32);
        for (unsigned int i = 0; i <  N_align; ++i)
            {
            x[i] = y[i] = z[i] = OverlapReal(0.0);
            }
        }
    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char *& ptr, unsigned int &available_bytes)
        {
        x.load_shared(ptr,available_bytes);
        y.load_shared(ptr,available_bytes);
        z.load_shared(ptr,available_bytes);
        hull_verts.load_shared(ptr,available_bytes);
        }

    //! Determine size of a shared memory allocation
    /*! \param ptr Pointer to increment
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char *& ptr, unsigned int &available_bytes) const
        {
        x.allocate_shared(ptr,available_bytes);
        y.allocate_shared(ptr,available_bytes);
        z.allocate_shared(ptr,available_bytes);
        hull_verts.allocate_shared(ptr,available_bytes);
        }

    #ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        x.set_memory_hint();
        y.set_memory_hint();
        z.set_memory_hint();
        hull_verts.set_memory_hint();
        }
    #endif

    ManagedArray<OverlapReal> x;        //!< X coordinate of vertices
    ManagedArray<OverlapReal> y;        //!< Y coordinate of vertices
    ManagedArray<OverlapReal> z;        //!< Z coordinate of vertices

    ManagedArray<unsigned int> hull_verts;  //!< List of triangles hull_verts[3*i], hull_verts[3*i+1], hull_verts[3*i+2] making up the convex hull
    unsigned int n_hull_verts;              //!< Number of vertices in the convex hull

    unsigned int N;                         //!< Number of vertices
    OverlapReal diameter;                   //!< Circumsphere diameter
    OverlapReal sweep_radius;               //!< Radius of the sphere sweep (used for spheropolyhedra)
    unsigned int ignore;                    //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                            //   First bit is ignore overlaps, Second bit is ignore statistics

    detail::OBB obb;                        //!< Tight fitting bounding box
    } __attribute__((aligned(32)));

//! Support function for ShapePolyhedron
/*! SupportFuncPolyhedron is a functor that computes the support function for ShapePolyhedron. For a given
    input vector in local coordinates, it finds the vertex most in that direction.

    \ingroup minkowski
*/

class SupportFuncConvexPolyhedron
    {
    public:
        //! Construct a support function for a convex polyhedron
        /*! \param _verts Polyhedron vertices
            Note that for performance it is assumed that unused vertices (beyond N) have already been set to zero.
        */
        DEVICE SupportFuncConvexPolyhedron(const poly3d_verts& _verts,
            OverlapReal extra_sweep_radius=OverlapReal(0.0))
            : verts(_verts), sweep_radius(extra_sweep_radius)
            {
            }

        //! Compute the support function
        /*! \param n Normal vector input (in the local frame)
            \returns Local coords of the point furthest in the direction of n
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& n) const
            {
            OverlapReal max_dot = -(verts.diameter * verts.diameter);
            unsigned int max_idx = 0;

            if (verts.N > 0)
                {
                #if !defined(__HIPCC__) && defined(__AVX__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
                // process dot products with AVX 8 at a time on the CPU when working with more than 4 verts
                __m256 nx_v = _mm256_broadcast_ss(&n.x);
                __m256 ny_v = _mm256_broadcast_ss(&n.y);
                __m256 nz_v = _mm256_broadcast_ss(&n.z);
                __m256 max_dot_v = _mm256_broadcast_ss(&max_dot);
                float d_s[verts.x.size()] __attribute__((aligned(32)));

                for (unsigned int i = 0; i < verts.N; i+=8)
                    {
                    __m256 x_v = _mm256_load_ps(verts.x.get() + i);
                    __m256 y_v = _mm256_load_ps(verts.y.get() + i);
                    __m256 z_v = _mm256_load_ps(verts.z.get() + i);

                    __m256 d_v = _mm256_add_ps(_mm256_mul_ps(nx_v, x_v), _mm256_add_ps(_mm256_mul_ps(ny_v, y_v), _mm256_mul_ps(nz_v, z_v)));

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
                #elif !defined(__HIPCC__) && defined(__SSE__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
                // process dot products with SSE 4 at a time on the CPU
                __m128 nx_v = _mm_load_ps1(&n.x);
                __m128 ny_v = _mm_load_ps1(&n.y);
                __m128 nz_v = _mm_load_ps1(&n.z);
                __m128 max_dot_v = _mm_load_ps1(&max_dot);
                float d_s[verts.x.size()] __attribute__((aligned(16)));

                for (unsigned int i = 0; i < verts.N; i+=4)
                    {
                    __m128 x_v = _mm_load_ps(verts.x.get() + i);
                    __m128 y_v = _mm_load_ps(verts.y.get() + i);
                    __m128 z_v = _mm_load_ps(verts.z.get() + i);

                    __m128 d_v = _mm_add_ps(_mm_mul_ps(nx_v, x_v), _mm_add_ps(_mm_mul_ps(ny_v, y_v), _mm_mul_ps(nz_v, z_v)));

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

                // if no AVX or SSE, or running in double precision, fall back on serial computation
                // this code path also triggers on the GPU

                OverlapReal max_dot0 = dot(n, vec3<OverlapReal>(verts.x[0], verts.y[0], verts.z[0]));
                unsigned int max_idx0 = 0;
                OverlapReal max_dot1 = dot(n, vec3<OverlapReal>(verts.x[1], verts.y[1], verts.z[1]));
                unsigned int max_idx1 = 1;
                OverlapReal max_dot2 = dot(n, vec3<OverlapReal>(verts.x[2], verts.y[2], verts.z[2]));
                unsigned int max_idx2 = 2;
                OverlapReal max_dot3 = dot(n, vec3<OverlapReal>(verts.x[3], verts.y[3], verts.z[3]));
                unsigned int max_idx3 = 3;

                for (unsigned int i = 4; i < verts.N; i+=4)
                    {
                    const OverlapReal *verts_x = verts.x.get() + i;
                    const OverlapReal *verts_y = verts.y.get() + i;
                    const OverlapReal *verts_z = verts.z.get() + i;
                    OverlapReal d0 = dot(n, vec3<OverlapReal>(verts_x[0], verts_y[0], verts_z[0]));
                    OverlapReal d1 = dot(n, vec3<OverlapReal>(verts_x[1], verts_y[1], verts_z[1]));
                    OverlapReal d2 = dot(n, vec3<OverlapReal>(verts_x[2], verts_y[2], verts_z[2]));
                    OverlapReal d3 = dot(n, vec3<OverlapReal>(verts_x[3], verts_y[3], verts_z[3]));

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

                vec3<OverlapReal> v(verts.x[max_idx], verts.y[max_idx], verts.z[max_idx]);
                if (sweep_radius != OverlapReal(0.0))
                    return v + (sweep_radius * fast::rsqrt(dot(n,n))) * n;
                else
                    return v;
                } // end if(verts.N > 0)
            else
                {
                if (sweep_radius != OverlapReal(0.0))
                    return (sweep_radius * fast::rsqrt(dot(n,n))) * n;
                else
                    return vec3<OverlapReal>(0.0, 0.0, 0.0); // No verts!
                }
            }

    private:
        const poly3d_verts& verts;      //!< Vertices of the polyhedron
        const OverlapReal sweep_radius; //!< Extra sweep radius
    };

/*!
 *  Geometric primitives for closest point calculation
 */

// From Real Time Collision Detection (Christer Ericson)
// https://doi.org/10.1201/b14581
DEVICE inline vec3<OverlapReal> closestPointOnTriangle(const vec3<OverlapReal>& p,
     const vec3<OverlapReal>& a, const vec3<OverlapReal>& b, const vec3<OverlapReal>& c)
    {
    vec3<OverlapReal> ab = b - a;
    vec3<OverlapReal> ac = c - a;
    vec3<OverlapReal> ap = p - a;

    OverlapReal d1 = dot(ab, ap);
    OverlapReal d2 = dot(ac, ap);
    if (d1 <= OverlapReal(0.0) && d2 <= OverlapReal(0.0)) return a; // barycentric coordinates (1,0,0)

    // Check if P in vertex region outside B
    vec3<OverlapReal> bp = p - b;
    OverlapReal d3 = dot(ab, bp);
    OverlapReal d4 = dot(ac, bp);
    if (d3 >= OverlapReal(0.0) && d4 <= d3) return b; // barycentric coordinates (0,1,0)

    // Check if P in edge region of AB, if so return projection of P onto AB
    OverlapReal vc = d1*d4 - d3*d2;
    if (vc <= OverlapReal(0.0) && d1 >= OverlapReal(0.0) && d3 <= OverlapReal(0.0))
        {
        OverlapReal v = d1 / (d1 - d3);
        return a + v * ab; // barycentric coordinates (1-v,v,0)
        }

    // Check if P in vertex region outside C
    vec3<OverlapReal> cp = p - c;
    OverlapReal d5 = dot(ab, cp);
    OverlapReal d6 = dot(ac, cp);
    if (d6 >= OverlapReal(0.0) && d5 <= d6) return c; // barycentric coordinates (0,0,1)

    // Check if P in edge region of AC, if so return projection of P onto AC
    OverlapReal vb = d5*d2 - d1*d6;
    if (vb <= OverlapReal(0.0) && d2 >= OverlapReal(0.0) && d6 <= OverlapReal(0.0))
        {
        OverlapReal w = d2 / (d2 - d6);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
        }
    // Check if P in edge region of BC, if so return projection of P onto BC
    OverlapReal va = d3*d6 - d5*d4;
    if (va <= OverlapReal(0.0) && (d4 - d3) >= OverlapReal(0.0) && (d5 - d6) >= OverlapReal(0.0))
        {
        OverlapReal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
        }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    OverlapReal denom = OverlapReal(1.0) / (va + vb + vc);
    OverlapReal v = vb * denom;
    OverlapReal w = vc * denom;
    return a + ab*v+ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
    }

// Test if point p lies outside plane through abc
DEVICE inline bool PointOutsideOfPlane(const vec3<OverlapReal>& p,
     const vec3<OverlapReal>& a, const vec3<OverlapReal>& b, const vec3<OverlapReal>& c)
    {
    return dot(p-a,cross(b-a,c-a)) >= OverlapReal(0.0);
    }

//! Find the point on a segment closest to point p
DEVICE inline vec3<OverlapReal> ClosestPtPointSegment(const vec3<OverlapReal>& c, const vec3<OverlapReal>& a,
    const vec3<OverlapReal>& b, OverlapReal& t)
    {
    vec3<OverlapReal> ab = b - a;

    vec3<OverlapReal> d;

    // Project c onto ab, but deferring divide by dot(ab,ab)
    t = dot(c -a, ab);

    if (t <= OverlapReal(0.0))
        {
        // c projects outside the [a,b] interval, on the a side; clamp to a
        t = OverlapReal(0.0);
        d = a;
        }
    else
        {
        OverlapReal denom = dot(ab,ab);
        if (t >= denom)
            {
            // c project outside the [a,b] interval, on the b side; clamp to b
            t = OverlapReal(1.0);
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

//! Projection function for ShapeConvexPolyhedron
/*! ProjectionFuncConvexPolyhedron is a functor that computes the projection function for ShapePolyhedron. For a given
    input point in local coordinates, it finds the vertex closest to that point.

    \ingroup minkowski
*/

class ProjectionFuncConvexPolyhedron
    {
    public:
        //! Construct a projection function for a convex polyhedron
        /*! \param _verts Polyhedron vertices
        */
        DEVICE ProjectionFuncConvexPolyhedron(const poly3d_verts& _verts,
            OverlapReal extra_sweep_radius=OverlapReal(0.0))
            : verts(_verts), sweep_radius(extra_sweep_radius)
            {
            }

        //! Compute the projection
        /*! \param p Point to compute the projection for
            \returns Local coords of the point in the shape closest to p
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& p) const
            {
            //! Find the point on the convex hull closest to p
            vec3<OverlapReal> closest_p = p;
            OverlapReal closest_dsq(FLT_MAX);

            // iterate over triangles of convex hull to find closest point on every face
            unsigned int n_hull_verts = verts.n_hull_verts;

            if (n_hull_verts > 0)
                {
                for (unsigned int i = 0; i < n_hull_verts; i+=3)
                    {
                    unsigned int k = verts.hull_verts[i];
                    unsigned int l = verts.hull_verts[i+1];
                    unsigned int m = verts.hull_verts[i+2];

                    vec3<OverlapReal> a(verts.x[k], verts.y[k], verts.z[k]);
                    vec3<OverlapReal> b(verts.x[l], verts.y[l], verts.z[l]);
                    vec3<OverlapReal> c(verts.x[m], verts.y[m], verts.z[m]);

                    // is the point on the outside of the plane?
                    /* For this to work correctly in the degenerate case, i.e. planar facet and 1d line, we
                       require that the convex hull always is a complete mesh
                     */
                    if (PointOutsideOfPlane(p, a, b, c))
                        {
                        vec3<OverlapReal> q = closestPointOnTriangle(p, a, b, c);
                        OverlapReal dsq = dot(p-q,p-q);

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
                    closest_p = vec3<OverlapReal>(0,0,0);
                    }
                else if (verts.N == 1)
                    {
                    // return the only point there is
                    closest_p = vec3<OverlapReal>(verts.x[0], verts.y[0], verts.z[0]);
                    }
                else if (verts.N == 2)
                    {
                    // line segment
                    OverlapReal t;
                    closest_p = ClosestPtPointSegment(p,
                        vec3<OverlapReal>(verts.x[0], verts.y[0], verts.z[0]),
                        vec3<OverlapReal>(verts.x[1], verts.y[1], verts.z[1]), t);
                    }
                }

            if (sweep_radius != 0.0 &&
                (p.x != closest_p.x || p.y != closest_p.y || p.z != closest_p.z))
                {
                // point is on the surface, see if we have to project further out
                vec3<OverlapReal> del = p - closest_p;
                OverlapReal dsq = dot(del,del);
                if (dsq > sweep_radius*sweep_radius)
                    {
                    // add the sphere radius in direction of closest approach, or the closest point inside the sphere
                    OverlapReal d = fast::sqrt(dsq);
                    return closest_p + sweep_radius/d*del;
                    }
                else
                    return p;
                }

            // pt is inside base shape
            return closest_p;
            }

    private:
        const poly3d_verts& verts;      //!< Vertices of the polyhedron
        const OverlapReal sweep_radius; //!< extra sphere sweep radius
    };


}; // end namespace detail

//! Convex Polyhedron shape template
/*! ShapeConvexPolyhedron implements IntegratorHPMC's shape protocol.

    The parameter defining a polyhedron is a structure containing a list of N vertices, centered on 0,0. In fact, it is
    **required** that the origin is inside the shape, and it is best if the origin is the center of mass.

    \ingroup shape
*/
struct ShapeConvexPolyhedron
    {
    //! Define the parameter type
    typedef detail::poly3d_verts param_type;

    //! Initialize a polyhedron
    DEVICE ShapeConvexPolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const { return true; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return verts.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": " << verts.sweep_radius << ", \"vertices\": [";
        for (unsigned int i = 0; i < verts.N-1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << ", " << verts.z[i] << "], ";
            }
        shapedef << "[" << verts.x[verts.N-1] << ", " << verts.y[verts.N-1] << ", " << verts.z[verts.N-1] << "]]}";
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // generate a tight AABB around the polyhedron
        // detail::SupportFuncConvexPolyhedron sfunc(verts);

        // // use support function of the to determine the furthest extent in each direction
        // quat<OverlapReal> o(orientation);
        // vec3<OverlapReal> e_x(1,0,0);
        // vec3<OverlapReal> e_y(0,1,0);
        // vec3<OverlapReal> e_z(0,0,1);
        // vec3<OverlapReal> s_x = rotate(o, sfunc(rotate(conj(o),e_x)));
        // vec3<OverlapReal> s_y = rotate(o, sfunc(rotate(conj(o),e_y)));
        // vec3<OverlapReal> s_z = rotate(o, sfunc(rotate(conj(o),e_z)));
        // vec3<OverlapReal> s_neg_x = rotate(o, sfunc(rotate(conj(o),-e_x)));
        // vec3<OverlapReal> s_neg_y = rotate(o, sfunc(rotate(conj(o),-e_y)));
        // vec3<OverlapReal> s_neg_z = rotate(o, sfunc(rotate(conj(o),-e_z)));

        // // translate out from the position by the furthest extents
        // vec3<Scalar> upper(pos.x + s_x.x, pos.y + s_y.y, pos.z + s_z.z);
        // vec3<Scalar> lower(pos.x + s_neg_x.x, pos.y + s_neg_y.y, pos.z + s_neg_z.z);

        // return detail::AABB(lower, upper);

        // ^^^^^^^ The above method is slow, just use a box that bounds the circumsphere
        return detail::AABB(pos, getCircumsphereDiameter()/Scalar(2));
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        detail::OBB obb = verts.obb;
        obb.affineTransform(orientation, pos);
        return obb;
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    //! Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return true;
        }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_verts& verts;     //!< Vertices
    };

//! Convex polyhedron overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \param sweep_radius_a Radius of a sphere to sweep shape a by
    \param sweep_radius_b Radius of a sphere to sweep shape b by
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                 const ShapeConvexPolyhedron& a,
                                 const ShapeConvexPolyhedron& b,
                                 unsigned int& err,
                                 Scalar sweep_radius_a,
                                 Scalar sweep_radius_b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    return detail::xenocollide_3d(detail::SupportFuncConvexPolyhedron(a.verts,sweep_radius_a),
                                  detail::SupportFuncConvexPolyhedron(b.verts,sweep_radius_b),
                                  rotate(conj(quat<OverlapReal>(a.orientation)), dr),
                                  conj(quat<OverlapReal>(a.orientation))* quat<OverlapReal>(b.orientation),
                                  DaDb/2.0,
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

//! Test for the overlap of a third convex polyhedron with the intersection of two convex polyhedra
/*! \param a First shape to test
    \param b Second shape to test
    \param c Third shape to test
    \param ab_t Position of second shape relative to first
    \param ac_t Position of third shape relative to first
    \param err Output variable that is incremented upon non-convergence
    \param sweep_radius Radius of a sphere to sweep all shapes by
*/
template<>
DEVICE inline bool test_overlap_intersection(const ShapeConvexPolyhedron& a,
    const ShapeConvexPolyhedron& b, const ShapeConvexPolyhedron& c,
    const vec3<Scalar>& ab_t, const vec3<Scalar>& ac_t, unsigned int &err,
    Scalar sweep_radius_a, Scalar sweep_radius_b, Scalar sweep_radius_c)
    {
    return detail::map_three(a,b,c,
        detail::SupportFuncConvexPolyhedron(a.verts,sweep_radius_a),
        detail::SupportFuncConvexPolyhedron(b.verts,sweep_radius_b),
        detail::SupportFuncConvexPolyhedron(c.verts,sweep_radius_c),
        detail::ProjectionFuncConvexPolyhedron(a.verts,sweep_radius_a),
        detail::ProjectionFuncConvexPolyhedron(b.verts,sweep_radius_b),
        detail::ProjectionFuncConvexPolyhedron(c.verts,sweep_radius_c),
        vec3<OverlapReal>(ab_t),
        vec3<OverlapReal>(ac_t),
        err);
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_CONVEX_POLYHEDRON_H__
