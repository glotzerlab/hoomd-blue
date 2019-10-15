// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "XenoCollide3D.h"
#include "hoomd/ManagedArray.h"

#ifndef __SHAPE_CONVEX_POLYHEDRON_H__
#define __SHAPE_CONVEX_POLYHEDRON_H__

/*! \file ShapeConvexPolyhedron.h
    \brief Defines the convex polyhedron shape
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

//! maximum number of vertices that can be stored (must be multiple of 8)
/*! \ingroup hpmc_data_structs */

//! Data structure for polyhedron vertices
//! Note that vectorized methods using this struct will assume unused coordinates are set to zero.
/*! \ingroup hpmc_data_structs */
struct poly3d_verts : param_base

    {
    //! Default constructor initializes zero values.
    DEVICE poly3d_verts()
        : N(0),
          diameter(OverlapReal(0)),
          sweep_radius(OverlapReal(0)),
          ignore(0)
        { }

    #ifndef NVCC
    //! Shape constructor
    poly3d_verts(unsigned int _N, bool _managed)
        : N(_N), diameter(0.0), sweep_radius(0.0), ignore(0)
        {
        unsigned int align_size = 8; //for AVX
        unsigned int N_align =((N + align_size - 1)/align_size)*align_size;
        x = ManagedArray<OverlapReal>(N_align,_managed);
        y = ManagedArray<OverlapReal>(N_align,_managed);
        z = ManagedArray<OverlapReal>(N_align,_managed);
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
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        x.load_shared(ptr,available_bytes);
        y.load_shared(ptr,available_bytes);
        z.load_shared(ptr,available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        x.attach_to_stream(stream);
        y.attach_to_stream(stream);
        z.attach_to_stream(stream);
        }
    #endif

    ManagedArray<OverlapReal> x;        //!< X coordinate of vertices
    ManagedArray<OverlapReal> y;        //!< Y coordinate of vertices
    ManagedArray<OverlapReal> z;        //!< Z coordinate of vertices
    unsigned int N;                         //!< Number of vertices
    OverlapReal diameter;                   //!< Circumsphere diameter
    OverlapReal sweep_radius;               //!< Radius of the sphere sweep (used for spheropolyhedra)
    unsigned int ignore;                    //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                            //   First bit is ignore overlaps, Second bit is ignore statistics
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
        DEVICE SupportFuncConvexPolyhedron(const poly3d_verts& _verts)
            : verts(_verts)
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
                #if !defined(NVCC) && defined(__AVX__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
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
                #elif !defined(NVCC) && defined(__SSE__) && (defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION))
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
                return vec3<OverlapReal>(verts.x[max_idx], verts.y[max_idx], verts.z[max_idx]);
                } // end if(verts.N > 0)
            else
                {
                return vec3<OverlapReal>(0.0, 0.0, 0.0); // No verts!
                }
            }

    private:
        const poly3d_verts& verts;      //!< Vertices of the polyhedron
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

    #ifndef NVCC
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

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_verts& verts;     //!< Vertices
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeConvexPolyhedron& a,
    const ShapeConvexPolyhedron &b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

//! Convex polyhedron overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                 const ShapeConvexPolyhedron& a,
                                 const ShapeConvexPolyhedron& b,
                                 unsigned int& err)
    {
    vec3<OverlapReal> dr(r_ab);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    return detail::xenocollide_3d(detail::SupportFuncConvexPolyhedron(a.verts),
                                  detail::SupportFuncConvexPolyhedron(b.verts),
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

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_CONVEX_POLYHEDRON_H__
