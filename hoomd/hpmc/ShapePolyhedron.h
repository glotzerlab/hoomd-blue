// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include <cfloat>

#include "GPUTree.h"

#ifndef __SHAPE_POLYHEDRON_H__
#define __SHAPE_POLYHEDRON_H__

//#define DEBUG_OUTPUT

/*! \file ShapePolyhedron.h
    \brief Defines the general polyhedron shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <iostream>
#endif

// Check against zero with absolute tolerance
#define CHECK_ZERO(x, abs_tol) ((x < abs_tol && x >= 0) || (-x < abs_tol && x < 0))

namespace hpmc
{

namespace detail
{

//! maximum number of faces that can be stored
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_POLY3D_FACES=25000;

//! maximum number of vertices per face
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_POLY3D_FACE_VERTS=4;

//! Maximum number of OBB Tree nodes
const unsigned int MAX_POLY3D_NODES=5000;

//! Maximum number of faces per OBB tree leaf node
const unsigned int MAX_POLY3D_CAPACITY=2;

//! Data structure for general polytopes
/*! \ingroup hpmc_data_structs */

struct poly3d_data : param_base
    {
    poly3d_data() : n_faces(0), ignore(0) {};

    #ifndef NVCC
    //! Constructor
    poly3d_data(unsigned int nverts, unsigned int _n_faces, unsigned int _n_face_verts, bool _managed)
        : n_faces(_n_faces)
        {
        verts = poly3d_verts(nverts, _managed);
        face_offs = ManagedArray<unsigned int>(n_faces+1,_managed);
        face_verts = ManagedArray<unsigned int>(_n_face_verts, _managed);
        }
    #endif

    poly3d_verts verts;                             //!< Holds parameters of convex hull
    ManagedArray<unsigned int> face_offs;           //!< Offset of every face in the list of vertices per face
    ManagedArray<unsigned int> face_verts;          //!< Ordered vertex IDs of every face
    unsigned int n_faces;                           //!< Number of faces
    unsigned int ignore;                            //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore

     //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param load If true, copy data to pointer, otherwise increment only
        \param ptr_max Maximum address in shared memory
     */
    HOSTDEVICE void load_shared(char *& ptr, bool load, char *ptr_max) const
        {
        verts.load_shared(ptr, load, ptr_max);
        face_offs.load_shared(ptr, load, ptr_max);
        face_verts.load_shared(ptr, load, ptr_max);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        verts.attach_to_stream(stream);
        face_offs.attach_to_stream(stream);
        face_verts.attach_to_stream(stream);
        }
    #endif
    } __attribute__((aligned(32)));

}; // end namespace detail

#if defined(NVCC) && (__CUDA_ARCH__ >= 300)
//! CTA allreduce
static __device__ int warp_reduce(int val, int width)
    {
    #pragma unroll
    for (int i = 1; i < width; i *= 2)
        {
        val += __shfl_xor(val,i,width);
        }
    return val;
    }
#endif

//!  Polyhedron shape template
/*! ShapePolyhedron implements IntegragorHPMC's shape protocol.

    The parameter defining a polyhedron is a structure containing a list of n_faces faces, each representing
    a polygon, for which the vertices are stored in sorted order, giving a total number of n_verts vertices.

    \ingroup shape
*/
struct ShapePolyhedron
    {
    typedef detail::GPUTree<detail::MAX_POLY3D_CAPACITY> gpu_tree_type;

    //! Define the parameter type
    typedef struct : public param_base {
        detail::poly3d_data data;
        gpu_tree_type tree;

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param load If true, copy data to pointer, otherwise increment only
            \param ptr_max Maximum address in shared memory

         */
        HOSTDEVICE void load_shared(char *& ptr, bool load, char *ptr_max) const
            {
            tree.load_shared(ptr, load, ptr_max);
            data.load_shared(ptr, load, ptr_max);
            }

        #ifdef ENABLE_CUDA
        //! Attach managed memory to CUDA stream
        void attach_to_stream(cudaStream_t stream) const
            {
            // attach managed memory arrays to stream
            tree.attach_to_stream(stream);
            data.attach_to_stream(stream);
            }
        #endif
        }
        param_type;

    //! Initialize a polyhedron
    DEVICE ShapePolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation),
        data(_params.data), tree(_params.tree)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() { return data.verts.N > 1; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return data.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return data.verts.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    //! Return true if this is a sphero-shape
    DEVICE OverlapReal isSpheroPolyhedron() const
        {
        return data.verts.sweep_radius != OverlapReal(0.0);
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, data.verts.diameter/Scalar(2));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return true; }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_data& data;     //!< Vertices
    const gpu_tree_type &tree;           //!< Tree for particle features
    };

DEVICE inline OverlapReal det_4x4(vec3<OverlapReal> a, vec3<OverlapReal> b, vec3<OverlapReal> c, vec3<OverlapReal> d)
    {
    return dot(cross(c,d),b-a)+dot(cross(a,b),d-c);
    }

// From Real Time Collision Detection (Christer Ericson)
DEVICE inline vec3<OverlapReal> closestPointToTriangle(const vec3<OverlapReal>& p,
     const vec3<OverlapReal>& a, const vec3<OverlapReal>& b, const vec3<OverlapReal>& c)
    {
    vec3<OverlapReal> ab = b - a;
    vec3<OverlapReal> ac = c - a;
    vec3<OverlapReal> ap = p - a;

    OverlapReal d1 = dot(ab, ap);
    OverlapReal d2 = dot(ac, ap);
    if (d1 <= OverlapReal(0.0) && d2 <= OverlapReal(0.0)) return a; // barycentric coordiantes (1,0,0)

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

    #if 0
    vec3<OverlapReal> bc = c - b;
    // Compute parameteric position s for projection P' of P on AB<
    // P' = A + s*AB, s = snom/(snom+denom)
    OverlapReal snom = dot(p-a, ab);
    OverlapReal sdenom = dot(p-b,a-b);

    // Compute parametric position for projection P' of P on AC,
    // P' = A + t*AC, s= tnom/(tnom+tdenom)
    OverlapReal tnom = dot(p-a, ac);
    OverlapReal tdenom = dot(p-c, a-c);


    if (snom <= OverlapReal(0.0) && tnom <= 0.0) return a; // Vertex region early out

    // Compute parametric position u for projection P' of P on BC,
    // P' = B + u*BC, u = unom/(unom+udenom)
    OverlapReal unom = dot(p-b, bc);
    OverlapReal udenom = dot(p-c,b-c);

    if (sdenom <= OverlapReal(0.0) && unom <= OverlapReal(0.0)) return b; // Vertex region early out
    if (tdenom <= OverlapReal(0.0) && udenom <= OverlapReal(0.0)) return c; // Vertex region early out

    // P is outside (or on) AB if the triple scalar product [N PA PB] <= 0
    vec3<OverlapReal> n = cross(b-a, c-a);
    vec3<OverlapReal> vc = dot(n, cross(a-p,b-p));

    // If P outside AB and within feature region of AB,
    // return projection of P onto AB
    if (vc <= OverlapReal(0.0) && snom >= OverlapReal(0.0) && sdenom >= OverlapReal(0.0))
        return a + snom / (snom + sdenom) * ab;

    // P is outside (or on) BC if the triple scalar product [N PB PC] <= 0
    OverlapReal va = dot(n, cross(b-p, c-p));

    // If P outside BC and within feature region of BC,
    // return projection of P onto BC
    if (va <= OverlapReal(0.0) && unom >= OverlapReal(0.0) && udenom >= OverlapReal(0.0))
        return b + unom / (unom + udenom) * bc;

    // P is outside (or on) C if the triple sclar product [N PC PA] <= 0
    OverlapReal vb = dot(n, cross(c-p, a-p));

    // If P outside CA and within feature region of CA,
    // return projection of P onto CA
    if (vb <= OverlapREal(0.0) && tnom >= OverlapReal(0.0) && tdenom >= OverlapReal(0.0))
        return a + tnom / (tnom + denom) * ac;

    // P must project inside the face region. Compute Q using barycentric coordinates
    OverlapReal u = va / (va + vb + vc);
    OverlapReal v = vb / (va + vb + vc);
    OverlapReal w = OverlapReal(1.0) - u - v; // = vc / (va + vb + vc)
    return u*a+v*b+w*c;
    #endif
    }


// Clamp n to lie within the range [min, max]
DEVICE inline OverlapReal clamp(OverlapReal n, OverlapReal min, OverlapReal max) {
    if (n < min) return min;
    if (n > max) return max;
    return n;
    }

// From Real Time Collision Detection (Christer Ericson)

// Computes closest points C1 and C2 of S1(s)=P1+s*(Q1-P1) and
// S2(t)=P2+t*(Q2-P2), returning s and t. Function result is squared
// distance between between S1(s) and S2(t)
DEVICE inline OverlapReal closestPtSegmentSegment(const vec3<OverlapReal>& p1, const vec3<OverlapReal>& q1,
    const vec3<OverlapReal>& p2, const vec3<OverlapReal>& q2, OverlapReal &s, OverlapReal &t, vec3<OverlapReal> &c1, vec3<OverlapReal> &c2)
    {
    vec3<OverlapReal> d1 = q1 - p1; // Direction vector of segment S1
    vec3<OverlapReal> d2 = q2 - p2; // Direction vector of segment S2
    vec3<OverlapReal> r = p1 - p2;
    OverlapReal a = dot(d1, d1); // Squared length of segment S1, always nonnegative
    OverlapReal e = dot(d2, d2); // Squared length of segment S2, always nonnegative
    OverlapReal f = dot(d2, r);

    const OverlapReal EPSILON(1e-6);

    // Check if either or both segments degenerate into points
    if (a <= EPSILON && e <= EPSILON)
        {
        // Both segments degenerate into points
        s = t = OverlapReal(0.0);
        c1 = p1;
        c2 = p2;
        return dot(c1 - c2, c1 - c2);
        }

    if (a <= EPSILON) {
        // First segment degenerates into a point
        s = OverlapReal(0.0);
        t = f / e; // s = 0 => t = (b*s + f) / e = f / e
        t = clamp(t, OverlapReal(0.0), OverlapReal(1.0));
        }
    else
        {
        OverlapReal c = dot(d1, r);
        if (e <= EPSILON)
            {
            // Second segment degenerates into a point
            t = OverlapReal(0.0);
            s = clamp(-c / a, OverlapReal(0.0), OverlapReal(1.0)); // t = 0 => s = (b*t - c) / a = -c / a
            }
        else
            {
            // The general nondegenerate case starts here
            OverlapReal b = dot(d1, d2);
            OverlapReal denom = a*e-b*b; // Always nonnegative
            // If segments not parallel, compute closest point on L1 to L2 and
            // clamp to segment S1. Else pick arbitrary s (here 0)
            if (denom != OverlapReal(0.0))
                {
                s = clamp((b*f - c*e) / denom, OverlapReal(0.0), OverlapReal(1.0));
                }
             else
                s = OverlapReal(0.0);

            // Compute point on L2 closest to S1(s) using
            // t = dot((P1 + D1*s) - P2,D2) / dot(D2,D2) = (b*s + f) / e
            t = (b*s + f) / e;
            // If t in [0,1] done. Else clamp t, recompute s for the new value
            // of t using s = dot((P2 + D2*t) - P1,D1) / dot(D1,D1)= (t*b - c) / a
            // and clamp s to [0, 1]
            if (t < OverlapReal(0.0))
                {
                t = OverlapReal(0.0);
                s = clamp(-c / a, OverlapReal(0.0), OverlapReal(1.0));
                }
             else
                 if (t > OverlapReal(1.0))
                    {
                    t = OverlapReal(1.0);
                    s = clamp((b - c) / a, OverlapReal(0.0), OverlapReal(1.0));
                    }
             }
        }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return dot(c1 - c2, c1 - c2);
    }

//! Test if a point lies on a line segment
/*! \param v The vertex coordinates
    \param a First point of line segment
    \param b Second point of line segment
 */
DEVICE inline bool test_vertex_line_segment_overlap(const vec3<OverlapReal>& v,
                                                    const vec3<OverlapReal>& a,
                                                    const vec3<OverlapReal>& b,
                                                    OverlapReal abs_tol)
    {
    vec3<OverlapReal> c = cross(v-a, b-a);
    OverlapReal d = dot(v - a, b - a)/dot(b-a,b-a);
    return (CHECK_ZERO(dot(c,c),abs_tol) && d >= OverlapReal(0.0) && d <= OverlapReal(1.0));
    }

//! Test for intersection of line segments in 3D
/*! \param p Support vertex of first line segment
 *  \param q Support vertex of second line segment
 *  \param a Vector between endpoints of first line segment
 *  \param b Vector between endpoints of second line segment
 *  \returns true if line segments intersect or overlap
 */
DEVICE inline bool test_line_segment_overlap(const vec3<OverlapReal>& p,
                                             const vec3<OverlapReal>& q,
                                             const vec3<OverlapReal>& a,
                                             const vec3<OverlapReal>& b,
                                             OverlapReal abs_tol)
    {
    /*
    if (det_4x4(p,q,p+a,q+b))
        return false; // line segments are skew
    */

    // 2d coordinate frame ex,ey
    vec3<OverlapReal> ex = a;
    OverlapReal mag_r = fast::sqrt(dot(ex,ex));
    ex /= mag_r;
    vec3<OverlapReal> ey = cross(ex,b);
    ey = cross(ey,ex);

    if (dot(ey,ey)) ey *= fast::rsqrt(dot(ey,ey));
    vec2<OverlapReal> r(mag_r, OverlapReal(0.0));
    vec2<OverlapReal> s(dot(b,ex),dot(b,ey));
    OverlapReal denom = (r.x*s.y-r.y*s.x);
    vec2<OverlapReal> del(dot(q - p, ex), dot(q - p, ey));

    if (CHECK_ZERO(denom,abs_tol))
        {
        // collinear or parallel?
        vec3<OverlapReal> c = cross(q-p,a);
        if (dot(c,c))
            return false; // parallel

        OverlapReal t = dot(del,r);
        OverlapReal u = -dot(del,s);
        if ((t < 0 || t > dot(r,r)) && (u < 0 || u > dot(s,s)))
            return false; // collinear, disjoint

        // collinear, overlapping
        return true;
        }

    OverlapReal t = (del.x*s.y - del.y*s.x)/denom;
    OverlapReal u = (del.x*r.y - del.y*r.x)/denom;
    if (t >= OverlapReal(0.0) && t <= OverlapReal(1.0) &&
        u >= OverlapReal(0.0) && u <= OverlapReal(1.0))
        {
        // intersection
        return true;
        }

    return false;
    }

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapePolyhedron& a,
    const ShapePolyhedron &b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    // first check overlap of circumspheres
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }


// compute shortest distance between two triangles
// Returns square of shortest distance
DEVICE inline OverlapReal shortest_distance_triangles(
    const vec3<OverlapReal> &a1,
    const vec3<OverlapReal> &b1,
    const vec3<OverlapReal> &c1,
    const vec3<OverlapReal> &a2,
    const vec3<OverlapReal> &b2,
    const vec3<OverlapReal> &c2)
    {
    // nine pairs of edges
    OverlapReal dmin_sq(FLT_MAX);

    vec3<OverlapReal> edge1, edge2;
    vec3<OverlapReal> p1, p2;
    OverlapReal s,t;

    OverlapReal dsq;
    dsq = closestPtSegmentSegment(a1,b1,a2,b2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,b1,a2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,b1,b2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,a2,b2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,a2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,b2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,a2,b2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,a2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,b2,c2, s,t,p1,p2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    // six vertex-triangle distances
    vec3<OverlapReal> p;

    p = closestPointToTriangle(a1, a2, b2, c2);
    dsq = dot(p-a1,p-a1);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    p = closestPointToTriangle(b1, a2, b2, c2);
    dsq = dot(p-b1,p-b1);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    p = closestPointToTriangle(c1, a2, b2, c2);
    dsq = dot(p-c1,p-c1);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    p = closestPointToTriangle(a2, a1, b1, c1);
    dsq = dot(p-a2,p-a2);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    p = closestPointToTriangle(b2, a1, b1, c1);
    dsq = dot(p-b2,p-b2);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    p = closestPointToTriangle(c2, a1, b1, c1);
    dsq = dot(p-c2,p-c2);
    if (dsq < dmin_sq)
        dmin_sq  = dsq;

    return dmin_sq;
    }

DEVICE inline bool test_narrow_phase_overlap( vec3<OverlapReal> r_ab,
                                              const ShapePolyhedron& a,
                                              const ShapePolyhedron& b,
                                              unsigned int cur_node_a,
                                              unsigned int cur_node_b,
                                              unsigned int &err)
    {
    // An absolute tolerance.
    // Possible improvement: make this adaptive as a function of ratios of occuring length scales
    const OverlapReal abs_tol(1e-16);

    // loop through faces of cur_node_a
    unsigned int na = a.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i< na; i++)
        {
        unsigned int iface = a.tree.getParticle(cur_node_a, i);

        // loop through faces of cur_node_b
        for (unsigned int j= 0; j< nb; j++)
            {
            unsigned int nverts_b, offs_b;
            bool intersect = false;

            unsigned int jface = b.tree.getParticle(cur_node_b, j);

            // Load number of face vertices
            unsigned int nverts_a = a.data.face_offs[iface + 1] - a.data.face_offs[iface];
            unsigned int offs_a = a.data.face_offs[iface];

            // fetch next face of particle b
            nverts_b = b.data.face_offs[jface + 1] - b.data.face_offs[jface];
            offs_b = b.data.face_offs[jface];

            unsigned int nverts_s0 = nverts_a;
            unsigned int nverts_s1 = nverts_b;

            if (nverts_a > 2 && nverts_b > 2)
                {
                // check collision between polygons
                unsigned int offs_s0 = offs_a;
                unsigned int offs_s1 = offs_b;

                vec3<OverlapReal> dr = rotate(conj(quat<OverlapReal>(b.orientation)),-r_ab);
                quat<OverlapReal> q(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation));
                // loop over edges of iface, then jface
                for (unsigned int ivertab = 0; ivertab < nverts_a+nverts_b; ivertab++)
                    {
                    const ShapePolyhedron &s0 = (ivertab < nverts_a ? a : b);
                    const ShapePolyhedron &s1 = (ivertab < nverts_a ? b : a);

                    if (ivertab == nverts_a)
                        {
                        dr = rotate(conj(quat<OverlapReal>(a.orientation)),r_ab);
                        q = conj(quat<OverlapReal>(a.orientation))*quat<OverlapReal>(b.orientation);
                        nverts_s0 = nverts_b;
                        nverts_s1 = nverts_a;
                        offs_s0 = offs_b;
                        offs_s1 = offs_a;
                        }

                    unsigned int ivert = (ivertab < nverts_a ? ivertab : ivertab - nverts_a);

                    unsigned int idx_a = s0.data.face_verts[offs_s0+ivert];
                    vec3<OverlapReal> v_a;
                    v_a.x = s0.data.verts.x[idx_a];
                    v_a.y = s0.data.verts.y[idx_a];
                    v_a.z = s0.data.verts.z[idx_a];
                    v_a = rotate(q,v_a) + dr;

                    // Load next vertex (t)
                    unsigned face_idx_next = (ivert + 1 == nverts_s0) ? 0 : ivert + 1;
                    unsigned int idx_next_a = s0.data.face_verts[offs_s0 + face_idx_next];
                    vec3<OverlapReal> v_next_a;
                    v_next_a.x = s0.data.verts.x[idx_next_a];
                    v_next_a.y = s0.data.verts.y[idx_next_a];
                    v_next_a.z = s0.data.verts.z[idx_next_a];
                    v_next_a = rotate(q,v_next_a) + dr;

                    bool collinear = false;
                    unsigned int face_idx_aux_a = face_idx_next + 1;
                    vec3<OverlapReal> v_aux_a,c;
                    do
                        {
                        face_idx_aux_a = (face_idx_aux_a == nverts_s0) ? 0 : face_idx_aux_a;
                        unsigned int idx_aux_a = s0.data.face_verts[offs_s0 + face_idx_aux_a];
                        v_aux_a.x = s0.data.verts.x[idx_aux_a];
                        v_aux_a.y = s0.data.verts.y[idx_aux_a];
                        v_aux_a.z = s0.data.verts.z[idx_aux_a];
                        v_aux_a = rotate(q,v_aux_a) + dr;
                        c = cross(v_next_a - v_a, v_aux_a - v_a);
                        collinear = CHECK_ZERO(dot(c,c),abs_tol);
                        } while(collinear && ++face_idx_aux_a < nverts_s0);

                    if (collinear)
                        {
                        err++;
                        return true;
                        }

                    bool overlap = false;

                    // Load vertex 0
                    vec3<OverlapReal> v_next_b;
                    unsigned int idx_v = s1.data.face_verts[offs_s1];
                    v_next_b.x = s1.data.verts.x[idx_v];
                    v_next_b.y = s1.data.verts.y[idx_v];
                    v_next_b.z = s1.data.verts.z[idx_v];

                    // vertex 1
                    idx_v = s1.data.face_verts[offs_s1 + 1];
                    vec3<OverlapReal> v_b;
                    v_b.x = s1.data.verts.x[idx_v];
                    v_b.y = s1.data.verts.y[idx_v];
                    v_b.z = s1.data.verts.z[idx_v];

                    // vertex 2
                    idx_v = s1.data.face_verts[offs_s1 + 2];
                    vec3<OverlapReal> v_aux_b;
                    v_aux_b.x = s1.data.verts.x[idx_v];
                    v_aux_b.y = s1.data.verts.y[idx_v];
                    v_aux_b.z = s1.data.verts.z[idx_v];

                    OverlapReal det_h = det_4x4(v_next_b, v_b, v_aux_b, v_a);
                    OverlapReal det_t = det_4x4(v_next_b, v_b, v_aux_b, v_next_a);

                    // for edge i to intersect face j, it is a necessary condition that it intersects the supporting plane
                    intersect = CHECK_ZERO(det_h,abs_tol) || CHECK_ZERO(det_t,abs_tol) || detail::signbit(det_h) != detail::signbit(det_t);

                    if (intersect)
                        {
                        unsigned int n_intersect = 0;

                        for (unsigned int jvert = 0; jvert < nverts_s1; ++jvert)
                            {
                            // Load vertex (p_i)
                            unsigned int idx_v = s1.data.face_verts[offs_s1+jvert];
                            vec3<OverlapReal> v_b;
                            v_b.x = s1.data.verts.x[idx_v];
                            v_b.y = s1.data.verts.y[idx_v];
                            v_b.z = s1.data.verts.z[idx_v];

                            // Load next vertex (p_i+1)
                            unsigned int next_vert_b = (jvert + 1 == nverts_s1) ? 0 : jvert + 1;
                            idx_v = s1.data.face_verts[offs_s1 + next_vert_b];
                            vec3<OverlapReal> v_next_b;
                            v_next_b.x = s1.data.verts.x[idx_v];
                            v_next_b.y = s1.data.verts.y[idx_v];
                            v_next_b.z = s1.data.verts.z[idx_v];

                            // compute determinants in homogeneous coordinates
                            OverlapReal det_s = det_4x4(v_a, v_next_a, v_aux_a, v_b);
                            OverlapReal det_u = det_4x4(v_a, v_next_a, v_aux_a, v_next_b);
                            OverlapReal det_v = det_4x4(v_a, v_next_a, v_b, v_next_b);

                            if (CHECK_ZERO(det_u, abs_tol) && !CHECK_ZERO(det_s,abs_tol))
                                {
                                // the endpoint of this edge touches the fictitious plane
                                // check if the next vertex of this face will be on the same side of the plane
                                unsigned int idx_aux_b = (jvert + 2 >= nverts_s1) ? jvert + 2 - nverts_s1 : jvert + 2;
                                idx_v = s1.data.face_verts[offs_s1 + idx_aux_b];
                                vec3<OverlapReal> v_aux_b;
                                v_aux_b.x = s1.data.verts.x[idx_v];
                                v_aux_b.y = s1.data.verts.y[idx_v];
                                v_aux_b.z = s1.data.verts.z[idx_v];

                                OverlapReal det_w = det_4x4(v_a, v_next_a, v_aux_a, v_aux_b);
                                if (CHECK_ZERO(det_w, abs_tol) || detail::signbit(det_w) == detail::signbit(det_s))
                                    {
                                    // the edge is reflected by the plane
                                    if (test_vertex_line_segment_overlap(v_next_b, v_next_a, v_a, abs_tol))
                                        {
                                        overlap = true;
                                        }
                                    }
                                // otherwise, ignore (to avoid double-counting of edges piercing the plane)
                                }
                            else if (CHECK_ZERO(det_v,abs_tol) || (CHECK_ZERO(det_s,abs_tol) && CHECK_ZERO(det_u,abs_tol)))
                                {
                                // iedge lies in the imaginary plane
                                if (test_line_segment_overlap(v_a, v_b, v_next_a - v_a, v_next_b - v_b,abs_tol))
                                    {
                                    overlap = true;
                                    }
                                }
                            else
                                {
                                /*
                                 * odd-parity rule
                                 */
                                int v = detail::signbit(det_v) ? -1 : 1;

                                if (CHECK_ZERO(det_s,abs_tol))
                                    {
                                    // the first point of this edge touches the fictitious plane
                                    // check if the previous vertex of this face was on the plane
                                    int idx_prev_b = ((int)jvert -1 < 0) ? nverts_s1 - 1: jvert -1;
                                    idx_v = s1.data.face_verts[offs_s1 + idx_prev_b];
                                    vec3<OverlapReal> v_prev_b;
                                    v_prev_b.x = s1.data.verts.x[idx_v];
                                    v_prev_b.y = s1.data.verts.y[idx_v];
                                    v_prev_b.z = s1.data.verts.z[idx_v];

                                    OverlapReal det_w = det_4x4(v_a, v_next_a, v_aux_a, v_prev_b);
                                    if (CHECK_ZERO(det_w, abs_tol) || detail::signbit(det_w) == detail::signbit(det_u))
                                        {
                                        // the edge is reflected by the plane
                                        if (test_vertex_line_segment_overlap(v_prev_b, v_next_a, v_a, abs_tol))
                                            {
                                            overlap = true;
                                            }
                                        }
                                    else
                                        {
                                        // b's edge contacts the supporting plane of edgei
                                        int u = detail::signbit(det_u) ? -1 : 1;
                                        if (u*v < 0)
                                            {
                                            n_intersect++;
                                            }
                                        }
                                    }
                                else
                                    {
                                    int s = detail::signbit(det_s) ? -1 : 1;
                                    int u = detail::signbit(det_u) ? -1 : 1;

                                    if (s*u < 0 && s*v>0)
                                        {
                                        n_intersect++;
                                        }
                                    }
                                }
                            }

                        overlap |= (n_intersect % 2);
                        } // end if (intersect)
                    if (overlap)
                        {
                        // overlap
                        return true;
                        }
                    } // end loop over edges
                }

            if (a.isSpheroPolyhedron() || b.isSpheroPolyhedron())
                {
                vec3<OverlapReal> dr = rotate(conj(quat<OverlapReal>(b.orientation)),-r_ab);
                quat<OverlapReal> q = conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation);

                OverlapReal dsqmin(FLT_MAX);

                // Load vertex 0 on a
                vec3<OverlapReal> a0;
                unsigned int idx_a = a.data.face_verts[offs_a];
                a0.x = a.data.verts.x[idx_a];
                a0.y = a.data.verts.y[idx_a];
                a0.z = a.data.verts.z[idx_a];
                a0 = rotate(q, a0) + dr;

                // vertex 0 on b
                unsigned int idx_b = b.data.face_verts[offs_b];
                vec3<OverlapReal> b0;
                b0.x = b.data.verts.x[idx_b];
                b0.y = b.data.verts.y[idx_b];
                b0.z = b.data.verts.z[idx_b];

                vec3<OverlapReal> b1, b2;

                if (nverts_b > 1)
                    {
                    // vertex 1 on b
                    idx_b = b.data.face_verts[offs_b + 1];
                    b1.x = b.data.verts.x[idx_b];
                    b1.y = b.data.verts.y[idx_b];
                    b1.z = b.data.verts.z[idx_b];

                    // vertex 1 on b
                    if (nverts_b == 2)
                        {
                        // degenerate
                        idx_b = b.data.face_verts[offs_b];
                        }
                    else
                        {
                        idx_b = b.data.face_verts[offs_b + 1];
                        }

                    b2.x = b.data.verts.x[idx_b];
                    b2.y = b.data.verts.y[idx_b];
                    b2.z = b.data.verts.z[idx_b];
                    }

                if (nverts_b > 1 && nverts_a == 1)
                    {
                    // optimization, test vertex against triangle b
                    vec3<OverlapReal> p;
                    p = closestPointToTriangle(a0, b0, b1, b2);
                    dsqmin = dot(p-a0,p-a0);
                    }

                vec3<OverlapReal> a1, a2;

                if (nverts_a > 1)
                    {
                    // vertex 1 on a
                    idx_a = a.data.face_verts[offs_a + 1];
                    a1.x = a.data.verts.x[idx_a];
                    a1.y = a.data.verts.y[idx_a];
                    a1.z = a.data.verts.z[idx_a];
                    a1 = rotate(q, a1) + dr;

                    // vertex 2 on a
                    if (nverts_a == 2)
                        {
                        // degenerate
                        idx_a = a.data.face_verts[offs_a];
                        }
                    else
                        {
                        idx_a = a.data.face_verts[offs_a + 2];
                        }

                    a2.x = a.data.verts.x[idx_a];
                    a2.y = a.data.verts.y[idx_a];
                    a2.z = a.data.verts.z[idx_a];
                    a2 = rotate(q, a2) + dr;
                    }

                if (nverts_a > 1 && nverts_b == 1)
                    {
                    // optimization, test vertex against triangle a
                    vec3<OverlapReal> p;
                    p = closestPointToTriangle(b0, a0, a1, a2);
                    dsqmin = dot(p-b0,p-b0);
                    }

                if (nverts_b > 1 && nverts_a > 1)
                    {
                    dsqmin = shortest_distance_triangles(a0, a1, a2, b0, b1, b2);
                    }

                if (nverts_a == 1 && nverts_b == 1)
                    {
                    // trivial case
                    dsqmin = dot(a0-b0,a0-b0);
                    }

                OverlapReal R_ab = a.data.verts.sweep_radius + b.data.verts.sweep_radius;

                if (R_ab*R_ab >= dsqmin)
                    {
                    // overlap of spherotriangles
                    return true;
                    }
                }
            } // end loop over faces of b
        } // end loop over over faces of a
    return false;
    }

#ifdef DEBUG_OUTPUT
inline void output_polys(const ShapePolyhedron& a, const ShapePolyhedron& b, quat<OverlapReal> q, const vec3<Scalar> dr)
    {
    std::cout << "shape polyV " << a.data.verts.N << " ";
    for (unsigned int i = 0; i < a.data.verts.N; ++i)
        {
        vec3<OverlapReal> v(a.data.verts.x[i], a.data.verts.y[i], a.data.verts.z[i]);
        std::cout << v.x << " " << v.y << " " << v.z << " ";
        }
    std::cout << a.data.n_faces << " ";
    for (unsigned int i = 0; i < a.data.n_faces; ++i)
        {
        unsigned int len = a.data.face_offs[i+1] - a.data.face_offs[i];
        std::cout << len << " ";
        for (unsigned int j = 0; j < len; ++j)
            {
            std::cout << a.data.face_verts[a.data.face_offs[i]+j] << " ";
            }
        }
    quat<OverlapReal> q_a(q);
    std::cout << "ffff0000 " << dr.x << " " << dr.y << " " << dr.z << " " << q_a.s << " " <<
        q_a.v.x << " " << q_a.v.y << " " << q_a.v.z << std::endl;

    std::cout << "shape polyV " << b.data.verts.N << " ";
    for (unsigned int i = 0; i < b.data.verts.N; ++i)
        {
        std::cout << b.data.verts.x[i] << " " << b.data.verts.y[i] << " " << b.data.verts.z[i] << " ";
        }
    std::cout << b.data.n_faces << " ";
    for (unsigned int i = 0; i < b.data.n_faces; ++i)
        {
        unsigned int len = b.data.face_offs[i+1] - b.data.face_offs[i];
        std::cout << len << " ";
        for (unsigned int j = 0; j < len; ++j)
            {
            std::cout << b.data.face_verts[b.data.face_offs[i]+j] << " ";
            }
        }
    std::cout << "ff00ff00 " << 0 << " " << 0 << " " << 0 << " " << 1 << " " <<
        0 << " " << 0 << " " << 0 << std::endl;
    }

inline void output_obb(const detail::OBB& obb, std::string color)
    {
    std::vector< vec3<OverlapReal> > corners(8);
    std::vector< std::pair<unsigned int, unsigned int> > edges(12);
    vec3<OverlapReal> ex(1,0,0);
    vec3<OverlapReal> ey(0,1,0);
    vec3<OverlapReal> ez(0,0,1);
    corners[0] =  ex*obb.lengths.x + ey*obb.lengths.y + ez*obb.lengths.z;
    corners[1] =  OverlapReal(-1.0)*ex*obb.lengths.x + ey*obb.lengths.y + ez*obb.lengths.z;
    corners[2] =  ex*obb.lengths.x - ey*obb.lengths.y + ez*obb.lengths.z;
    corners[3] =  OverlapReal(-1.0)* ex*obb.lengths.x - ey*obb.lengths.y + ez*obb.lengths.z;
    corners[4] =  ex*obb.lengths.x + ey*obb.lengths.y - ez*obb.lengths.z;
    corners[5] =  OverlapReal(-1.0)* ex*obb.lengths.x + ey*obb.lengths.y - ez*obb.lengths.z;
    corners[6] =  ex*obb.lengths.x - ey*obb.lengths.y - ez*obb.lengths.z;
    corners[7] =  OverlapReal(-1.0)* ex*obb.lengths.x - ey*obb.lengths.y - ez*obb.lengths.z;

    edges[0] = std::make_pair(0,1);
    edges[1] = std::make_pair(0,2);
    edges[2] = std::make_pair(0,4);
    edges[3] = std::make_pair(1,3);
    edges[4] = std::make_pair(1,5);
    edges[5] = std::make_pair(2,3);
    edges[6] = std::make_pair(2,6);
    edges[7] = std::make_pair(3,7);
    edges[8] = std::make_pair(4,5);
    edges[9] = std::make_pair(4,6);
    edges[10] = std::make_pair(5,7);
    edges[11] = std::make_pair(6,7);

    for (unsigned int i = 0; i < 12; ++i)
        {
        vec3<OverlapReal> r_a,r_b;
        r_a = obb.center + obb.rotation*corners[edges[i].first];
        r_b = obb.center + obb.rotation*corners[edges[i].second];
        std::cout << "connection 0.1 " << color << " " << r_a.x << " " << r_a.y << " " << r_a.z
            << " " << r_b.x << " " << r_b.y << " " << r_b.z << std::endl;
        }
    }
#endif

//! Polyhedron overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                 const ShapePolyhedron& a,
                                 const ShapePolyhedron& b,
                                 unsigned int& err)
    {

    // test overlap of convex hulls
    if (a.isSpheroPolyhedron() || b.isSpheroPolyhedron())
        {
        if (!test_overlap(r_ab, ShapeSpheropolyhedron(a.orientation,a.data.verts),
               ShapeSpheropolyhedron(b.orientation,b.data.verts),err)) return false;
        }
    else
        {
        if (!test_overlap(r_ab, ShapeConvexPolyhedron(a.orientation,a.data.verts),
           ShapeConvexPolyhedron(b.orientation,b.data.verts),err)) return false;
        }

    vec3<OverlapReal> dr = r_ab;
    /*
     * This overlap test checks if an edge of one polyhedron is overlapping with a face of the other
     */

    /*
     * This overlap test checks if either
     * a) an edge of one polyhedron intersects the face of the other
     * b) the center of mass of one polyhedron is contained in the other
     */

    #ifdef NVCC
    // Parallel tree traversal
    unsigned int offset = threadIdx.x;
    unsigned int stride = blockDim.x;
    #else
    unsigned int offset = 0;
    unsigned int stride = 1;
    #endif

    const detail::GPUTree<detail::MAX_POLY3D_CAPACITY>& tree_a = a.tree;
    const detail::GPUTree<detail::MAX_POLY3D_CAPACITY>& tree_b = b.tree;

    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = offset; cur_leaf_a < tree_a.getNumLeaves(); cur_leaf_a += stride)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
            // rotate and translate a's obb into b's body frame
            obb_a.affineTransform(conj(b.orientation)*a.orientation,
                rotate(conj(b.orientation),-r_ab));

            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b) && test_narrow_phase_overlap(r_ab, a, b, cur_node_a, query_node, err)) return true;
                }
            }
        }
    else
        {
        for (unsigned int cur_leaf_b = offset; cur_leaf_b < tree_b.getNumLeaves(); cur_leaf_b += stride)
            {
            unsigned int cur_node_b = tree_b.getLeafNode(cur_leaf_b);
            hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

            // rotate and translate b's obb into a's body frame
            obb_b.affineTransform(conj(a.orientation)*b.orientation,
                rotate(conj(a.orientation),r_ab));

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a) && test_narrow_phase_overlap(-r_ab, b, a, cur_node_b, query_node, err)) return true;
                }
            }
        }

    // no intersecting edge, check if one polyhedron is contained in the other

    // since the origin must be contained within each shape, a zero separation is an overlap
    const OverlapReal tol(1e-12);
    if (dot(dr,dr) < tol) return true;

    // if shape(A) == shape(B), only consider intersections
    if (&a.data == &b.data) return false;

    // a small rotation angle for perturbation
    const OverlapReal eps_angle(0.123456);

    // a relative translation amount for perturbation
    const OverlapReal eps_trans(0.456789);

    // An absolute tolerance.
    // Possible improvement: make this adaptive as a function of ratios of occuring length scales
    const OverlapReal abs_tol(1e-7);

    for (unsigned int ord = 0; ord < 2; ++ord)
        {
        // load pair of shapes
        const ShapePolyhedron &s0 = (ord == 0) ? a : b;
        const ShapePolyhedron &s1 = (ord == 0) ? b : a;

        vec3<OverlapReal> v_a,v_next_a,v_aux_a;

        // the origin vertex is (0,0,0), and must be contained in the shape
        if (ord == 0)
            {
            v_a = -dr;
            }
        else
            {
            v_a = dr;
            }

        // Check if s0 is contained in s1 by shooting a ray from one of its origin
        // to a point 2*outside the circumsphere of b in direction of the origin separation


        if (ord == 0)
            {
            v_next_a = dr*(b.getCircumsphereDiameter()*fast::rsqrt(dot(dr,dr)));
            }
        else
            {
            v_next_a = -dr*a.getCircumsphereDiameter()*fast::rsqrt(dot(dr,dr));
            }

       bool degenerate = false;
       unsigned int perturb_count = 0;
       const unsigned int MAX_PERTURB_COUNT = 10;
       do
            {
            degenerate = false;
            unsigned int n_overlap = 0;

            // load a second, non-colinear vertex
            bool collinear = false;
            unsigned int aux_idx = 0;
            do
                {
                v_aux_a.x = s0.data.verts.x[aux_idx];
                v_aux_a.y = s0.data.verts.y[aux_idx];
                v_aux_a.z = s0.data.verts.z[aux_idx];
                v_aux_a = rotate(quat<OverlapReal>(s0.orientation),v_aux_a);

                if (ord == 0)
                    {
                    v_aux_a -= dr;
                    }
                else
                    {
                    v_aux_a += dr;
                    }

                // check if collinear
                vec3<OverlapReal> c = cross(v_next_a - v_a, v_aux_a - v_a);
                collinear = CHECK_ZERO(dot(c,c),abs_tol);
                aux_idx++;
                } while(collinear && aux_idx < s0.data.verts.N);

            if (aux_idx == s0.data.verts.N)
                {
                err++;
                return true;
                }

            // loop through faces
            for (unsigned int jface = 0; jface < s1.data.n_faces; jface ++)
                {
                unsigned int nverts, offs_b;
                bool intersect = false;

                // fetch next face
                nverts = s1.data.face_offs[jface + 1] - s1.data.face_offs[jface];
                offs_b = s1.data.face_offs[jface];

                if (nverts < 3) continue;

                // Load vertex 0
                vec3<OverlapReal> v_next_b;
                unsigned int idx_v = s1.data.face_verts[offs_b];
                v_next_b.x = s1.data.verts.x[idx_v];
                v_next_b.y = s1.data.verts.y[idx_v];
                v_next_b.z = s1.data.verts.z[idx_v];
                v_next_b = rotate(quat<OverlapReal>(s1.orientation),v_next_b);

                // vertex 1
                idx_v = s1.data.face_verts[offs_b + 1];
                vec3<OverlapReal> v_b;
                v_b.x = s1.data.verts.x[idx_v];
                v_b.y = s1.data.verts.y[idx_v];
                v_b.z = s1.data.verts.z[idx_v];
                v_b = rotate(quat<OverlapReal>(s1.orientation),v_b);

                // vertex 2
                idx_v = s1.data.face_verts[offs_b + 2];
                vec3<OverlapReal> v_aux_b;
                v_aux_b.x = s1.data.verts.x[idx_v];
                v_aux_b.y = s1.data.verts.y[idx_v];
                v_aux_b.z = s1.data.verts.z[idx_v];
                v_aux_b = rotate(quat<OverlapReal>(s1.orientation),v_aux_b);

                OverlapReal det_h = det_4x4(v_next_b, v_b, v_aux_b, v_a);
                OverlapReal det_t = det_4x4(v_next_b, v_b, v_aux_b, v_next_a);

                // for edge i to intersect face j, it is a necessary condition that it intersects the supporting plane
                intersect = CHECK_ZERO(det_h,abs_tol) || CHECK_ZERO(det_t,abs_tol) || detail::signbit(det_h) != detail::signbit(det_t);

                if (intersect)
                    {
                    bool overlap = false;
                    unsigned int n_intersect = 0;

                    for (unsigned int jvert = 0; jvert < nverts; ++jvert)
                        {
                        // Load vertex (p_i)
                        unsigned int idx_v = s1.data.face_verts[offs_b+jvert];
                        vec3<OverlapReal> v_b;
                        v_b.x = s1.data.verts.x[idx_v];
                        v_b.y = s1.data.verts.y[idx_v];
                        v_b.z = s1.data.verts.z[idx_v];
                        v_b = rotate(quat<OverlapReal>(s1.orientation),v_b);

                        // Load next vertex (p_i+1)
                        unsigned int next_vert_b = (jvert + 1 == nverts) ? 0 : jvert + 1;
                        idx_v = s1.data.face_verts[offs_b + next_vert_b];
                        vec3<OverlapReal> v_next_b;
                        v_next_b.x = s1.data.verts.x[idx_v];
                        v_next_b.y = s1.data.verts.y[idx_v];
                        v_next_b.z = s1.data.verts.z[idx_v];
                        v_next_b = rotate(quat<OverlapReal>(s1.orientation),v_next_b);

                        // compute determinants in homogeneous coordinates
                        OverlapReal det_s = det_4x4(v_a, v_next_a, v_aux_a, v_b);
                        OverlapReal det_u = det_4x4(v_a, v_next_a, v_aux_a, v_next_b);
                        OverlapReal det_v = det_4x4(v_a, v_next_a, v_b, v_next_b);

                        if (CHECK_ZERO(det_u, abs_tol) ||CHECK_ZERO(det_s,abs_tol) || CHECK_ZERO(det_v,abs_tol))
                            {
                            degenerate = true;
                            break;
                            }
                        /*
                         * odd-parity rule
                         */
                        int v = detail::signbit(det_v) ? -1 : 1;

                        int s = detail::signbit(det_s) ? -1 : 1;
                        int u = detail::signbit(det_u) ? -1 : 1;

                        if (s*u < 0 && s*v>0)
                            {
                            n_intersect++;
                            }
                        } // end loop over vertices

                   if (degenerate)
                        {
                        break;
                        }

                   if (n_intersect % 2)
                        {
                        overlap = true;
                        }
                    if (overlap)
                        {
                        n_overlap++;
                        }
                    } // end if intersect
                } // end loop over faces

            if (! degenerate)
                {
                // apply odd parity rule
                if (n_overlap % 2)
                    {
                    return true;
                    }
                }
            else
                {
                // perturb the end point
                v_next_a = rotate(quat<OverlapReal>::fromAxisAngle(v_aux_a, eps_angle), v_next_a);

                // find the closest vertex to the origin of the first shape
                OverlapReal min_dist(FLT_MAX);
                vec3<OverlapReal> v_min;
                for (unsigned int i = 0; i < s0.data.verts.N; ++i)
                    {
                    vec3<OverlapReal> v;
                    v.x = s0.data.verts.x[i];
                    v.y = s0.data.verts.y[i];
                    v.z = s0.data.verts.z[i];

                    v = rotate(quat<OverlapReal>(s0.orientation), v);
                    OverlapReal d = fast::sqrt(dot(v,v));
                    if (d < min_dist)
                        {
                        min_dist = d;
                        v_min = v;
                        }
                    }

                // perturb the origin of the ray along the direction of the closest vertex
                if (ord == 0)
                    {
                    v_a += eps_trans*(v_min-v_a+dr)-dr;
                    }
                else
                    {
                    v_a += eps_trans*(v_min-v_a-dr)+dr;
                    }

                perturb_count ++;
                if (perturb_count == MAX_PERTURB_COUNT)
                    {
                    err++;
                    return true;
                    }
                }
            } while (degenerate);
        }

    return false;
    }

}; // end namespace hpmc

#endif //__SHAPE_POLYHEDRON_H__
