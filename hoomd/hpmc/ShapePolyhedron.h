// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

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

/*! \file ShapePolyhedron.h
    \brief Defines the general polyhedron shape
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
#endif

/*!  This overlap check has been optimized to the best of my ability. However, further optimizations may still be possible,
  in particular regarding the tree traversal and the type of bounding volume hierarchy. Generally, I have found
  OBB's to perform superior to AABB's and spheres, because they are tightly fitting. The tandem overlap check is also
  faster than checking all leaves against the tree on the CPU. On the GPU, leave against tree traversal may be faster due
  to the possibility of parallelizing over the leave nodes, but that also leads to longer autotuning times. Even though
  tree traversal is non-recursive, occasionally I see stack errors (= overflows) on Pascal GPUs when the shape is highly complicated.
  Then the stack frame could be increased using cudaDeviceSetLimit().

  Since GPU performance is mostly deplorable for concave polyhedra, I have not put much effort into optimizing for that code path.
  The parallel overlap check code path has been left in here for future experimentation, and can be enabled by
  uncommenting the below line.
  */

// uncomment for parallel overlap checks
//#define LEAVES_AGAINST_TREE_TRAVERSAL

namespace hpmc
{

namespace detail
{

//! Data structure for general polytopes
/*! \ingroup hpmc_data_structs */

struct poly3d_data : param_base
    {
    poly3d_data() : n_faces(0), ignore(0) {};

    #ifndef NVCC
    //! Constructor
    poly3d_data(unsigned int nverts, unsigned int _n_faces, unsigned int _n_face_verts, unsigned int n_hull_verts, bool _managed)
        : n_verts(nverts), n_faces(_n_faces), hull_only(0)
        {
        convex_hull_verts = poly3d_verts(n_hull_verts, _managed);
        verts = ManagedArray<vec3<OverlapReal> >(nverts, _managed);
        face_offs = ManagedArray<unsigned int>(n_faces+1,_managed);
        face_verts = ManagedArray<unsigned int>(_n_face_verts, _managed);
        face_overlap = ManagedArray<unsigned int>(_n_faces, _managed);
        std::fill(face_overlap.get(), face_overlap.get()+_n_faces, 1);
        }
    #endif

    GPUTree tree;                                   //!< Tree for fast locality lookups
    poly3d_verts convex_hull_verts;                 //!< Holds parameters of convex hull
    ManagedArray<vec3<OverlapReal> > verts;         //!< Vertex coordinates
    ManagedArray<unsigned int> face_offs;           //!< Offset of every face in the list of vertices per face
    ManagedArray<unsigned int> face_verts;          //!< Ordered vertex IDs of every face
    ManagedArray<unsigned int> face_overlap;        //!< Overlap mask per face
    unsigned int n_verts;                           //!< Number of vertices
    unsigned int n_faces;                           //!< Number of faces
    unsigned int ignore;                            //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
    vec3<OverlapReal> origin;                       //!< A point *inside* the surface
    unsigned int hull_only;                         //!< If 1, only the hull of the shape is considered for overlaps
    OverlapReal sweep_radius;                       //!< Radius of a sweeping sphere

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        tree.load_shared(ptr, available_bytes);
        convex_hull_verts.load_shared(ptr, available_bytes);
        verts.load_shared(ptr, available_bytes);
        face_offs.load_shared(ptr, available_bytes);
        face_verts.load_shared(ptr, available_bytes);
        face_overlap.load_shared(ptr, available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        tree.attach_to_stream(stream);
        convex_hull_verts.attach_to_stream(stream);
        verts.attach_to_stream(stream);
        face_offs.attach_to_stream(stream);
        face_verts.attach_to_stream(stream);
        face_overlap.attach_to_stream(stream);
        }
    #endif
    } __attribute__((aligned(32)));

}; // end namespace detail

//!  Polyhedron shape template
/*! ShapePolyhedron implements IntegratorHPMC's shape protocol.

    The parameter defining a polyhedron is a structure containing a list of n_faces faces, each representing
    a polygon, for which the vertices are stored in sorted order, giving a total number of n_verts vertices.

    \ingroup shape
*/
struct ShapePolyhedron
    {
    //! Define the parameter type
    typedef detail::poly3d_data param_type;

    //! Initialize a polyhedron
    DEVICE ShapePolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), data(_params), tree(_params.tree)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() { return data.n_verts > 1; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return data.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return data.convex_hull_verts.diameter;
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
        return data.sweep_radius != OverlapReal(0.0);
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        unsigned int n_verts = data.n_verts;
        unsigned int n_faces = data.n_faces;
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Mesh\", \"vertices\": [";
        for (unsigned int i = 0; i < n_verts-1; i++)
            {
            shapedef << "[" << data.verts[i].x << ", " << data.verts[i].y << ", " << data.verts[i].z << "], ";
            }
        shapedef << "[" << data.verts[n_verts-1].x << ", " << data.verts[n_verts-1].y << ", " << data.verts[n_verts-1].z << "]], \"indices\": [";
        unsigned int nverts_face, offset;
        for (unsigned int i = 0; i < n_faces; i++)
            {
            // Number of vertices of ith face
            nverts_face = data.face_offs[i + 1] - data.face_offs[i];
            offset = data.face_offs[i];
            shapedef << "[";
            for (unsigned int j = 0; j < nverts_face-1; j++)
                {
                shapedef << data.face_verts[offset+j] << ", ";
                }
            shapedef << data.face_verts[offset+nverts_face-1];
            if (i == n_faces-1)
                shapedef << "]]}";
            else
                shapedef << "], ";
            }
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, data.convex_hull_verts.diameter/Scalar(2));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
        #ifdef LEAVES_AGAINST_TREE_TRAVERSAL
        return true;
        #else
        return false;
        #endif
        }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_data& data;     //!< Vertices
    const detail::GPUTree &tree;           //!< Tree for particle features
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
DEVICE inline OverlapReal closestPtSegmentSegment(const vec3<OverlapReal> p1, const vec3<OverlapReal>& q1,
    const vec3<OverlapReal>& p2, const vec3<OverlapReal>& q2, OverlapReal &s, OverlapReal &t, vec3<OverlapReal> &c1, vec3<OverlapReal> &c2, OverlapReal abs_tol)
    {
    vec3<OverlapReal> d1 = q1 - p1; // Direction vector of segment S1
    vec3<OverlapReal> d2 = q2 - p2; // Direction vector of segment S2
    vec3<OverlapReal> r = p1 - p2;
    OverlapReal a = dot(d1, d1); // Squared length of segment S1, always nonnegative
    OverlapReal e = dot(d2, d2); // Squared length of segment S2, always nonnegative
    OverlapReal f = dot(d2, r);

    // Check if either or both segments degenerate into points
    if (CHECK_ZERO(a,abs_tol) && CHECK_ZERO(e,abs_tol))
        {
        // Both segments degenerate into points
        s = t = OverlapReal(0.0);
        c1 = p1;
        c2 = p2;
        return dot(c1 - c2, c1 - c2);
        }

    if (CHECK_ZERO(a, abs_tol)) {
        // First segment degenerates into a point
        s = OverlapReal(0.0);
        t = f / e; // s = 0 => t = (b*s + f) / e = f / e
        t = clamp(t, OverlapReal(0.0), OverlapReal(1.0));
        }
    else
        {
        OverlapReal c = dot(d1, r);
        if (CHECK_ZERO(e, abs_tol))
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
             else if (t > OverlapReal(1.0))
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
    const vec3<OverlapReal> &c2,
    OverlapReal abs_tol)
    {
    // nine pairs of edges
    OverlapReal dmin_sq(FLT_MAX);

    vec3<OverlapReal> p1, p2;
    OverlapReal s,t;

    OverlapReal dsq;
    dsq = closestPtSegmentSegment(a1,b1,a2,b2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,b1,a2,c2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,b1,b2,c2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,a2,b2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,a2,c2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(a1,c1,b2,c2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,a2,b2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,a2,c2, s,t,p1,p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    dsq = closestPtSegmentSegment(b1,c1,b2,c2, s,t,p1,p2, abs_tol);
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

#include <hoomd/extern/triangle_triangle.h>

/*! Test overlap in narrow phase

    \param dr separation vector between the particles, IN THE REFERENCE FRAME of b
    \param a first shape
    \param b second shape
    \param cur_node_a Node in a's tree to check
    \param cur_node_a Node in b's tree to check
    \param err gets incremented if there are errors (not currently implemented)
    \param abs_tol an absolute tolerance for the triangle triangle check
 */
DEVICE inline bool test_narrow_phase_overlap( vec3<OverlapReal> dr,
                                              const ShapePolyhedron& a,
                                              const ShapePolyhedron& b,
                                              unsigned int cur_node_a,
                                              unsigned int cur_node_b,
                                              unsigned int &err,
                                              OverlapReal abs_tol)
    {
    // loop through faces of cur_node_a
    unsigned int na = a.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.tree.getNumParticles(cur_node_b);

    for (unsigned int i= 0; i< na; i++)
        {
        unsigned int iface = a.tree.getParticle(cur_node_a, i);

        // Load number of face vertices
        unsigned int nverts_a = a.data.face_offs[iface + 1] - a.data.face_offs[iface];
        unsigned int offs_a = a.data.face_offs[iface];
        unsigned mask_a = a.data.face_overlap[iface];

        float U[3][3];

        quat<OverlapReal> q(conj(quat<OverlapReal>(b.orientation))*quat<OverlapReal>(a.orientation));

        if (nverts_a > 2)
            {
            for (unsigned int ivert = 0; ivert < 3; ++ivert)
                {
                unsigned int idx_a = a.data.face_verts[offs_a+ivert];
                vec3<float> v = a.data.verts[idx_a];
                v = rotate(q,v) + dr;
                U[ivert][0] = v.x; U[ivert][1] = v.y; U[ivert][2] = v.z;
                }
            }

        // loop through faces of cur_node_b
        for (unsigned int j= 0; j< nb; j++)
            {
            unsigned int nverts_b, offs_b;

            unsigned int jface = b.tree.getParticle(cur_node_b, j);

            // fetch next face of particle b
            nverts_b = b.data.face_offs[jface + 1] - b.data.face_offs[jface];
            offs_b = b.data.face_offs[jface];
            unsigned int mask_b = b.data.face_overlap[jface];

            // only check overlaps if required
            if (! (mask_a & mask_b)) continue;

            if (nverts_a > 2 && nverts_b > 2)
                {
                float V[3][3];
                for (unsigned int ivert = 0; ivert < 3; ++ivert)
                    {
                    unsigned int idx_b = b.data.face_verts[offs_b+ivert];
                    vec3<float> v = b.data.verts[idx_b];
                    V[ivert][0] = v.x; V[ivert][1] = v.y; V[ivert][2] = v.z;
                    }

                // check collision between triangles
                if (NoDivTriTriIsect(V[0],V[1],V[2],U[0],U[1],U[2],abs_tol))
                    {
                    return true;
                    }
                }

            if (a.isSpheroPolyhedron() || b.isSpheroPolyhedron())
                {
                OverlapReal dsqmin(FLT_MAX);

                // Load vertex 0 on a
                unsigned int idx_a = a.data.face_verts[offs_a];
                vec3<OverlapReal> a0 = a.data.verts[idx_a];
                a0 = rotate(q, a0) + dr;

                // vertex 0 on b
                unsigned int idx_b = b.data.face_verts[offs_b];
                vec3<OverlapReal> b0 = b.data.verts[idx_b];

                vec3<OverlapReal> b1, b2;

                if (nverts_b > 1)
                    {
                    // vertex 1 on b
                    idx_b = b.data.face_verts[offs_b + 1];
                    b1 = b.data.verts[idx_b];

                    // vertex 2 on b
                    idx_b = b.data.face_verts[offs_b + 2];
                    b2 = b.data.verts[idx_b];
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
                    a1 = a.data.verts[idx_a];
                    a1 = rotate(q, a1) + dr;

                    // vertex 2 on a
                    idx_a = a.data.face_verts[offs_a + 2];
                    a2 = a.data.verts[idx_a];
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
                    dsqmin = shortest_distance_triangles(a0, a1, a2, b0, b1, b2, abs_tol);
                    }

                if (nverts_a == 1 && nverts_b == 1)
                    {
                    // trivial case
                    dsqmin = dot(a0-b0,a0-b0);
                    }

                OverlapReal R_ab = a.data.sweep_radius + b.data.sweep_radius;

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

// From Real-time Collision Detection (Christer Ericson)
// Given ray pq and triangle abc, returns whether segment intersects
// triangle and if so, also returns the barycentric coordinates (u,v,w)
// of the intersection point
// Note: the triangle is assumed to be oriented counter-clockwise when viewed from the direction of p
DEVICE inline bool IntersectRayTriangle(const vec3<OverlapReal>& p, const vec3<OverlapReal>& q,
     const vec3<OverlapReal>& a, const vec3<OverlapReal>& b, const vec3<OverlapReal>& c,
    OverlapReal &u, OverlapReal &v, OverlapReal &w, OverlapReal &t)
    {
    vec3<OverlapReal> ab = b - a;
    vec3<OverlapReal> ac = c - a;
    vec3<OverlapReal> qp = p - q;

    // Compute triangle normal. Can be precalculated or cached if
    // intersecting multiple segments against the same triangle
    vec3<OverlapReal> n = cross(ab, ac);

    // Compute denominator d. If d <= 0, segment is parallel to or points
    // away from triangle, so exit early
    float d = dot(qp, n);
    if (d <= OverlapReal(0.0)) return false;

    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    vec3<OverlapReal> ap = p - a;
    t = dot(ap, n);
    if (t < OverlapReal(0.0)) return false;
//    if (t > d) return false; // For segment; exclude this code line for a ray test

    // Compute barycentric coordinate components and test if within bounds
    vec3<OverlapReal> e = cross(qp, ap);
    v = dot(ac, e);
    if (v < OverlapReal(0.0) || v > d) return false;
    w = -dot(ab, e);
    if (w < OverlapReal(0.0) || v + w > d) return false;

    // Segment/ray intersects triangle. Perform delayed division and
    // compute the last barycentric coordinate component
    float ood = OverlapReal(1.0) / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = OverlapReal(1.0) - v - w;
    return true;
    }

#ifndef NVCC
//! Traverse the bounding volume test tree recursively
inline bool BVHCollision(const ShapePolyhedron& a, const ShapePolyhedron &b,
     unsigned int cur_node_a, unsigned int cur_node_b,
     const quat<OverlapReal>& q, const vec3<OverlapReal>& dr, unsigned int &err, OverlapReal abs_tol)
    {
    detail::OBB obb_a = a.tree.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr);
    detail::OBB obb_b = b.tree.getOBB(cur_node_b);

    if (!overlap(obb_a, obb_b)) return false;

    if (a.tree.isLeaf(cur_node_a))
        {
        if (b.tree.isLeaf(cur_node_b))
            {
            return test_narrow_phase_overlap(dr, a, b, cur_node_a, cur_node_b, err, abs_tol);
            }
        else
            {
            unsigned int left_b = b.tree.getLeftChild(cur_node_b);
            unsigned int right_b = b.tree.getEscapeIndex(left_b);

            return BVHCollision(a, b, cur_node_a, left_b, q, dr, err, abs_tol)
                || BVHCollision(a, b, cur_node_a, right_b, q, dr, err, abs_tol);
            }
        }
    else
        {
        if (b.tree.isLeaf(cur_node_b))
            {
            unsigned int left_a = a.tree.getLeftChild(cur_node_a);
            unsigned int right_a = a.tree.getEscapeIndex(left_a);

            return BVHCollision(a, b, left_a, cur_node_b, q, dr, err, abs_tol)
                || BVHCollision(a, b, right_a, cur_node_b, q, dr, err, abs_tol);
            }
        else
            {
            unsigned int left_a = a.tree.getLeftChild(cur_node_a);
            unsigned int right_a = a.tree.getEscapeIndex(left_a);
            unsigned int left_b = b.tree.getLeftChild(cur_node_b);
            unsigned int right_b = b.tree.getEscapeIndex(left_b);

            return BVHCollision(a, b, left_a, left_b, q, dr, err, abs_tol)
                || BVHCollision(a, b, left_a, right_b, q, dr, err, abs_tol)
                || BVHCollision(a, b, right_a, left_b, q, dr, err, abs_tol)
                || BVHCollision(a, b, right_a, right_b, q, dr, err, abs_tol);
            }
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
        if (!test_overlap(r_ab, ShapeSpheropolyhedron(a.orientation,a.data.convex_hull_verts),
               ShapeSpheropolyhedron(b.orientation,b.data.convex_hull_verts),err)) return false;
        }
    else
        {
        if (!test_overlap(r_ab, ShapeConvexPolyhedron(a.orientation,a.data.convex_hull_verts),
           ShapeConvexPolyhedron(b.orientation,b.data.convex_hull_verts),err)) return false;
        }

    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    const OverlapReal abs_tol(DaDb*1e-12);
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
    const detail::GPUTree& tree_a = a.tree;
    const detail::GPUTree& tree_b = b.tree;
    #endif

    #ifdef LEAVES_AGAINST_TREE_TRAVERSAL
    #ifdef NVCC
    // Parallel tree traversal
    unsigned int offset = threadIdx.x;
    unsigned int stride = blockDim.x;
    #else
    unsigned int offset = 0;
    unsigned int stride = 1;
    #endif

    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = offset; cur_leaf_a < tree_a.getNumLeaves(); cur_leaf_a += stride)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
            // rotate and translate a's obb into b's body frame
            vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
            obb_a.affineTransform(conj(b.orientation)*a.orientation, dr_rot);

            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b) && test_narrow_phase_overlap(dr_rot, a, b, cur_node_a, query_node, err, abs_tol)) return true;
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
            vec3<OverlapReal> dr_rot(rotate(conj(a.orientation),r_ab));
            obb_b.affineTransform(conj(a.orientation)*b.orientation, dr_rot);

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a) && test_narrow_phase_overlap(dr_rot, b, a, cur_node_b, query_node, err,abs_tol)) return true;
                }
            }
        }
    #else
    vec3<OverlapReal> dr_rot(rotate(conj(b.orientation),-r_ab));
    quat<OverlapReal> q(conj(b.orientation)*a.orientation);

    #ifndef NVCC
    if (BVHCollision(a,b,0,0, q, dr_rot, err, abs_tol)) return true;
    #else
    // stackless traversal on GPU
    unsigned long int stack = 0;
    unsigned int cur_node_a = 0;
    unsigned int cur_node_b = 0;

    detail::OBB obb_a = tree_a.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr_rot);

    detail::OBB obb_b = tree_b.getOBB(cur_node_b);


    while (cur_node_a != tree_a.getNumNodes() && cur_node_b != tree_b.getNumNodes())
        {
        unsigned int query_node_a = cur_node_a;
        unsigned int query_node_b = cur_node_b;

        if (detail::traverseBinaryStack(tree_a, tree_b, cur_node_a, cur_node_b, stack, obb_a, obb_b, q,dr_rot)
            && test_narrow_phase_overlap(dr_rot, a, b, query_node_a, query_node_b, err, abs_tol)) return true;
        }
    #endif
    #endif

    // no intersecting edge, check if one polyhedron is contained in the other

    // if shape(A) == shape(B), only consider intersections
    if (&a.data == &b.data) return false;

    for (unsigned int ord = 0; ord < 2; ++ord)
        {
        // load shape
        const ShapePolyhedron &s1 = (ord == 0) ? b : a;

        // if the shape is a hull only, skip
        if (s1.data.hull_only) continue;

        vec3<OverlapReal> p;

        if (ord == 0)
            {
            p = -dr+rotate(quat<OverlapReal>(a.orientation),a.data.origin);
            }
        else
            {
            p = dr+rotate(quat<OverlapReal>(b.orientation),b.data.origin);
            }

        // Check if s0 is contained in s1 by shooting a ray from its origin
        // in direction of origin separation
        vec3<OverlapReal> n = dr+rotate(quat<OverlapReal>(b.orientation),b.data.origin)-
            rotate(quat<OverlapReal>(a.orientation),a.data.origin);

        // rotate ray in coordinate system of shape s1
        p = rotate(conj(quat<OverlapReal>(s1.orientation)), p);
        n = rotate(conj(quat<OverlapReal>(s1.orientation)), n);

        if (ord != 0)
            {
            n = -n;
            }

        vec3<OverlapReal> q = p + n;

        unsigned int n_overlap = 0;

        // query ray against OBB tree
        unsigned cur_node_s1 = 0;
        while (cur_node_s1 < s1.tree.getNumNodes())
            {
            unsigned int query_node = cur_node_s1;
            if (s1.tree.queryRay(p,n, cur_node_s1, abs_tol))
                {
                unsigned int n_faces = s1.tree.getNumParticles(query_node);

                // loop through faces
                for (unsigned int j = 0; j < n_faces; j ++)
                    {
                    // fetch next face
                    unsigned int jface = s1.tree.getParticle(query_node, j);
                    unsigned int offs_b = s1.data.face_offs[jface];

                    if (s1.data.face_offs[jface + 1] - offs_b < 3) continue;

                    // Load vertex 0
                    vec3<OverlapReal> v_b[3];
                    unsigned int idx_v = s1.data.face_verts[offs_b];
                    v_b[0] = s1.data.verts[idx_v];

                    // vertex 1
                    idx_v = s1.data.face_verts[offs_b + 1];
                    v_b[1] = s1.data.verts[idx_v];

                    // vertex 2
                    idx_v = s1.data.face_verts[offs_b + 2];
                    v_b[2] = s1.data.verts[idx_v];

                    OverlapReal u,v,w,t;

                    // two-sided triangle test
                    if (IntersectRayTriangle(p, q, v_b[0], v_b[1], v_b[2],u,v,w,t)
                     || IntersectRayTriangle(p, q, v_b[2], v_b[1], v_b[0],u,v,w,t))
                        {
                        n_overlap++;
                        }
                    }
                }
            }

        // apply odd parity rule
        if (n_overlap % 2)
            {
            return true;
            }
        }

    return false;
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_POLYHEDRON_H__
