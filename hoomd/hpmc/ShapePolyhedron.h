
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeConvexPolyhedron.h"

#include "hoomd/AABBTree.h"
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
const unsigned int MAX_POLY3D_FACES=32;

//! maximum number of vertices per face
/*! \ingroup hpmc_data_structs */
const unsigned int MAX_POLY3D_FACE_VERTS=5;
const unsigned int MAX_POLY3D_VERTS = 128;

//! maximum number of edges (assuming a non self-intersecting polyhedron)
/*! \ingroup hpmc_data_structures */
const unsigned int MAX_POLY3D_EDGES=MAX_POLY3D_VERTS+MAX_POLY3D_FACES-2;

//! Data structure for general polytopes
/*! \ingroup hpmc_data_structs */

struct poly3d_data : aligned_struct
    {
    poly3d_verts<MAX_POLY3D_VERTS> verts;                             //!< Holds parameters of convex hull
    unsigned int edges[MAX_POLY3D_EDGES*2];         //!< Pairs of vertex IDs for every unique edge
    unsigned int face_offs[MAX_POLY3D_FACES+1];     //!< Offset of every face in the list of vertices per face
    unsigned int face_verts[MAX_POLY3D_FACE_VERTS*MAX_POLY3D_FACES]; //!< Ordered vertex IDs of every face
    unsigned int n_faces;                           //!< Number of faces
    unsigned int n_edges;                           //!< Number of edges
    unsigned int ignore;                            //!< Bitwise ignore flag for stats, overlaps. 1 will ignore, 0 will not ignore
                                                    //   First bit is ignore overlaps, Second bit is ignore statistics
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
    //! Define the parameter type
    typedef struct{
        detail::poly3d_data data;
        hpmc::detail::GPUTree tree;
        }
        param_type;

    //! Initialize a polyhedron
    DEVICE ShapePolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation),
        data(_params.data), tree(_params.tree)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const { return true; }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return data.ignore>>1 & 0x01; }

    //!Ignore flag for overlaps
    DEVICE bool ignoreOverlaps() const { return data.ignore & 0x01; }

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

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, data.verts.diameter/Scalar(2));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_data& data;     //!< Vertices
    const hpmc::detail::GPUTree &tree; //!< Tree for particle features
    };

DEVICE inline OverlapReal det_4x4(vec3<OverlapReal> a, vec3<OverlapReal> b, vec3<OverlapReal> c, vec3<OverlapReal> d)
    {
    return dot(cross(c,d),b-a)+dot(cross(a,b),d-c);
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
    if (!test_overlap(r_ab, ShapeConvexPolyhedron<detail::MAX_POLY3D_VERTS>(a.orientation,a.data.verts),
       ShapeConvexPolyhedron<detail::MAX_POLY3D_VERTS>(b.orientation,b.data.verts),err)) return false;

    vec3<OverlapReal> dr = r_ab;
    /*
     * This overlap test checks if an edge of one polyhedron is overlapping with a face of the other
     */

    /*
     * This overlap test checks if either
     * a) an edge of one polyhedron intersects the face of the other
     * b) the center of mass of one polyhedron is contained in the other
     */

    // An absolute tolerance.
    // Possible improvement: make this adaptive as a function of ratios of occuring length scales
    const OverlapReal abs_tol(1e-7);

    for (unsigned int iedge = 0; iedge < a.data.n_edges + b.data.n_edges; iedge++)
        {
        // Current bounding box
        hpmc::detail::AABB aabb;

        // Begin a new tree traversal
        unsigned int cur_node = 0;

        if (iedge == a.data.n_edges)
            {
            // switch the separation vector
            dr = -dr;
            }

        const ShapePolyhedron &s0 = (iedge < a.data.n_edges) ? a : b;
        unsigned int edge = (iedge < a.data.n_edges) ? iedge : iedge-a.data.n_edges;

        const ShapePolyhedron &s1 = (iedge < a.data.n_edges) ? b : a;

        // Load vertex (h)
        unsigned int idx_a = s0.data.edges[edge*2];
        vec3<OverlapReal> v_a;
        v_a.x = s0.data.verts.x[idx_a];
        v_a.y = s0.data.verts.y[idx_a];
        v_a.z = s0.data.verts.z[idx_a];
        v_a = rotate(quat<OverlapReal>(s0.orientation),v_a) - dr;
        v_a = rotate(conj(quat<OverlapReal>(s1.orientation)), v_a);

        // Load next vertex (t)
        unsigned int idx_next_a = s0.data.edges[edge*2+1];
        vec3<OverlapReal> v_next_a;
        v_next_a.x = s0.data.verts.x[idx_next_a];
        v_next_a.y = s0.data.verts.y[idx_next_a];
        v_next_a.z = s0.data.verts.z[idx_next_a];
        v_next_a = rotate(quat<OverlapReal>(s0.orientation),v_next_a) - dr;
        v_next_a = rotate(conj(quat<OverlapReal>(s1.orientation)), v_next_a);

        vec3<Scalar> lo, hi;
        lo.x = v_a.x; lo.y = v_a.y; lo.z = v_a.z;
        hi.x = v_next_a.x; hi.y = v_next_a.y; hi.z = v_next_a.z;
        if (lo.x > hi.x)
            {
            OverlapReal t = lo.x;
            lo.x = hi.x;
            hi.x = t;
            }
        if (lo.y > hi.y)
            {
            OverlapReal t = lo.y;
            lo.y = hi.y;
            hi.y = t;
            }
        if (lo.z > hi.z)
            {
            OverlapReal t = lo.z;
            lo.z = hi.z;
            hi.z = t;
            }
        aabb = hpmc::detail::AABB(lo,hi);

        bool collinear = false;
        unsigned int next_edge = edge;
        vec3<OverlapReal> v_aux_a;
        do
            {
            next_edge++;
            // Load a third vertex (v)
            next_edge = (next_edge == s0.data.n_edges) ? 0 : next_edge;
            unsigned int idx_aux_a = s0.data.edges[2*next_edge];
            if (idx_aux_a == idx_a || idx_aux_a == idx_next_a) idx_aux_a = s0.data.edges[2*next_edge+1];
            v_aux_a.x = s0.data.verts.x[idx_aux_a];
            v_aux_a.y = s0.data.verts.y[idx_aux_a];
            v_aux_a.z = s0.data.verts.z[idx_aux_a];
            v_aux_a = rotate(quat<OverlapReal>(s0.orientation),v_aux_a) - dr;
            v_aux_a = rotate(conj(quat<OverlapReal>(s1.orientation)), v_aux_a);
            vec3<OverlapReal> c = cross(v_next_a - v_a, v_aux_a - v_a);
            collinear = CHECK_ZERO(dot(c,c),abs_tol);
            } while(collinear);

        int faces[hpmc::detail::NODE_CAPACITY];

        while (cur_node < s1.tree.getNumNodes())
            {
            bool leaf = false;
            // fetch next overlapping leaf node
            while (!leaf && cur_node < s1.tree.getNumNodes())
                {
                leaf = s1.tree.queryNode(aabb, cur_node, faces);
                }

            if (leaf)
                {
                bool overlap = false;

                // loop through faces
                for (unsigned int jface = 0; jface < hpmc::detail::NODE_CAPACITY; jface ++)
                    {
                    unsigned int nverts, offs_b;
                    bool intersect = false;
                    if (faces[jface] != -1)
                        {
                        const hpmc::detail::AABB &leaf_aabb = s1.tree.getLeafAABB(cur_node-1, jface);
                        if (! detail::overlap(leaf_aabb,aabb))
                            {
                            continue;
                            }

                        // fetch next face
                        nverts = s1.data.face_offs[faces[jface] + 1] - s1.data.face_offs[faces[jface]];
                        offs_b = s1.data.face_offs[faces[jface]];

                        // Load vertex 0
                        vec3<OverlapReal> v_next_b;
                        unsigned int idx_v = s1.data.face_verts[offs_b];
                        v_next_b.x = s1.data.verts.x[idx_v];
                        v_next_b.y = s1.data.verts.y[idx_v];
                        v_next_b.z = s1.data.verts.z[idx_v];

                        // vertex 1
                        idx_v = s1.data.face_verts[offs_b + 1];
                        vec3<OverlapReal> v_b;
                        v_b.x = s1.data.verts.x[idx_v];
                        v_b.y = s1.data.verts.y[idx_v];
                        v_b.z = s1.data.verts.z[idx_v];

                        // vertex 2
                        idx_v = s1.data.face_verts[offs_b + 2];
                        vec3<OverlapReal> v_aux_b;
                        v_aux_b.x = s1.data.verts.x[idx_v];
                        v_aux_b.y = s1.data.verts.y[idx_v];
                        v_aux_b.z = s1.data.verts.z[idx_v];

                        OverlapReal det_h = det_4x4(v_next_b, v_b, v_aux_b, v_a);
                        OverlapReal det_t = det_4x4(v_next_b, v_b, v_aux_b, v_next_a);

                        // for edge i to intersect face j, it is a necessary condition that it intersects the supporting plane
                        intersect = CHECK_ZERO(det_h,abs_tol) || CHECK_ZERO(det_t,abs_tol) || detail::signbit(det_h) != detail::signbit(det_t);
                        }

                    if (intersect)
                        {
                        unsigned int n_intersect = 0;

                        for (unsigned int jvert = 0; jvert < nverts; ++jvert)
                            {
                            // Load vertex (p_i)
                            unsigned int idx_v = s1.data.face_verts[offs_b+jvert];
                            vec3<OverlapReal> v_b;
                            v_b.x = s1.data.verts.x[idx_v];
                            v_b.y = s1.data.verts.y[idx_v];
                            v_b.z = s1.data.verts.z[idx_v];

                            // Load next vertex (p_i+1)
                            unsigned int next_vert_b = (jvert + 1 == nverts) ? 0 : jvert + 1;
                            idx_v = s1.data.face_verts[offs_b + next_vert_b];
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
                                unsigned int idx_aux_b = (jvert + 2 >= nverts) ? jvert + 2 - nverts : jvert + 2;
                                idx_v = s1.data.face_verts[offs_b + idx_aux_b];
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
                                    int idx_prev_b = ((int)jvert -1 < 0) ? nverts - 1: jvert -1;
                                    idx_v = s1.data.face_verts[offs_b + idx_prev_b];
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
                        }
                    if (overlap)
                        {
                        // overlap
                        return true;
                        }
                    } // end loop over faces
                } // end if leaf
            } // end loop over leaf nodes

        } // end loop over edges

    // no intersecting edge, check if one polyhedron is contained in the other

    // since the origin must be contained within each shape, a zero separation is an overlap
    dr = vec3<OverlapReal>(r_ab);
    const OverlapReal tol(1e-12);
    if (dot(dr,dr) < tol) return true;

    // if shape(A) == shape(B), only consider intersections
    if (&a.data == &b.data) return false;

    // a small rotation angle for perturbation
    const OverlapReal eps_angle(0.123456);

    // a relative translation amount for perturbation
    const OverlapReal eps_trans(0.456789);

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
