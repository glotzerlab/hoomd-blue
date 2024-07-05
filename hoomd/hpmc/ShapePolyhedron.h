// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once
#include "GPUTree.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSphere.h"
#include "ShapeSpheropolyhedron.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <hoomd/extern/triangle_triangle.h>

#include <cfloat>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

/* This overlap check has been optimized to the best of my ability. However, further optimizations
  may still be possible, in particular regarding the tree traversal and the type of bounding volume
  hierarchy. Generally, I have found OBB's to perform superior to AABB's and spheres, because they
  are tightly fitting. The tandem overlap check is also faster than checking all leaves against the
  tree on the CPU. On the GPU, leave against tree traversal may be faster due to the possibility of
  parallelizing over the leave nodes, but that also leads to longer autotuning times. Even though
  tree traversal is non-recursive, occasionally I see stack errors (= overflows) on Pascal GPUs when
  the shape is highly complicated. Then the stack frame could be increased using
  cudaDeviceSetLimit(). Since GPU performance is mostly deplorable for concave polyhedra, I have not
  put much effort into optimizing for that code path. The parallel overlap check code path has been
  left in here for future experimentation, and can be enabled by uncommenting the below line.
*/
// uncomment for parallel overlap checks
// #define LEAVES_AGAINST_TREE_TRAVERSAL

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/** Polyhedron shape

    Define the parameters of a general polyhedron for HPMC shape overlap checks. Polyhedra are
    defined N vertices and a triangle mesh indexed on those vertices. The shape data precomputes an
    OBB tree of the triangles for use in an efficient overlap check.

     The polyhedrons's diameter is precomputed from the vertex farthest from the origin. Arrays are
    stored in ManagedArray to support arbitrary numbers of verticles.
*/
struct TriangleMesh : ShapeParams
    {
    TriangleMesh() : face_verts(), face_overlap(), n_faces(0), ignore(0) {};

#ifndef __HIPCC__
    /** Initialize with a given number of vertices and vaces
     */
    TriangleMesh(unsigned int n_verts_,
                 unsigned int n_faces_,
                 unsigned int n_face_verts_,
                 bool managed)
        : n_verts(n_verts_), n_faces(n_faces_), hull_only(0), sweep_radius(0), diameter(0.0)
        {
        verts = ManagedArray<vec3<ShortReal>>(n_verts, managed);
        face_offs = ManagedArray<unsigned int>(n_faces + 1, managed);
        face_verts = ManagedArray<unsigned int>(n_face_verts_, managed);
        face_overlap = ManagedArray<unsigned int>(n_faces, managed);
        std::fill(face_overlap.get(), face_overlap.get() + n_faces, 1);
        }

    /// Construct from a Python dictionary
    TriangleMesh(pybind11::dict v, bool managed = false)
        {
        pybind11::list verts_list = v["vertices"];
        pybind11::list face_list = v["faces"];
        pybind11::object overlap = v["overlap"];
        pybind11::tuple origin_tuple = v["origin"];

        if (len(origin_tuple) != 3)
            throw std::runtime_error("origin must have 3 elements");

        ShortReal R = v["sweep_radius"].cast<ShortReal>();
        ignore = v["ignore_statistics"].cast<unsigned int>();
        hull_only = v["hull_only"].cast<unsigned int>();
        n_verts = (unsigned int)pybind11::len(verts_list);
        n_faces = (unsigned int)pybind11::len(face_list);
        origin = vec3<ShortReal>(pybind11::cast<ShortReal>(origin_tuple[0]),
                                 pybind11::cast<ShortReal>(origin_tuple[1]),
                                 pybind11::cast<ShortReal>(origin_tuple[2]));

        unsigned int leaf_capacity = v["capacity"].cast<unsigned int>();

        verts = ManagedArray<vec3<ShortReal>>(n_verts, managed);
        face_offs = ManagedArray<unsigned int>(n_faces + 1, managed);
        face_verts = ManagedArray<unsigned int>(n_faces * 3, managed);
        face_overlap = ManagedArray<unsigned int>(n_faces, managed);

        sweep_radius = R;

        face_offs[0] = 0;
        for (unsigned int i = 1; i <= n_faces; i++)
            {
            face_offs[i] = face_offs[i - 1] + 3;
            }

        if (overlap.is(pybind11::none()))
            {
            for (unsigned int i = 0; i < n_faces; i++)
                {
                face_overlap[i] = 1;
                }
            }
        else
            {
            pybind11::list overlap_list = overlap;
            if (pybind11::len(overlap_list) != n_faces)
                {
                throw std::runtime_error(
                    "Number of member overlap flags must be equal to number faces");
                }
            for (unsigned int i = 0; i < n_faces; i++)
                {
                face_overlap[i] = pybind11::cast<unsigned int>(overlap_list[i]);
                }
            }
        // extract the verts from the python list and compute the radius on the way
        ShortReal radius_sq = ShortReal(0.0);
        for (unsigned int i = 0; i < n_verts; i++)
            {
            pybind11::list vert_list = verts_list[i];
            if (len(vert_list) != 3)
                throw std::runtime_error("Each vertex must have 3 elements");
            vec3<ShortReal> vert;
            vert.x = pybind11::cast<ShortReal>(vert_list[0]);
            vert.y = pybind11::cast<ShortReal>(vert_list[1]);
            vert.z = pybind11::cast<ShortReal>(vert_list[2]);
            verts[i] = vert;
            radius_sq = max(radius_sq, dot(vert, vert));
            }

        // extract the faces
        for (unsigned int i = 0; i < n_faces; i++)
            {
            pybind11::list face_i = face_list[i];
            if (len(face_i) != 3)
                throw std::runtime_error("Each face must have 3 vertices");

            for (unsigned int j = 0; j < 3; j++)
                {
                unsigned int k = pybind11::cast<unsigned int>(face_i[j]);
                if (k >= n_verts)
                    {
                    std::ostringstream oss;
                    oss << "Invalid vertex index " << k << " specified" << std::endl;
                    throw std::runtime_error(oss.str());
                    }

                face_verts[i * 3 + j] = k;
                }
            }

        // construct bounding box tree
        hpmc::detail::OBB* obbs = new hpmc::detail::OBB[n_faces];
        std::vector<std::vector<vec3<ShortReal>>> internal_coordinates;

        for (unsigned int i = 0; i < n_faces; ++i)
            {
            std::vector<vec3<ShortReal>> face_vec;

            unsigned int n_vert = 0;
            for (unsigned int j = face_offs[i]; j < face_offs[i + 1]; ++j)
                {
                face_vec.push_back(verts[face_verts[j]]);
                n_vert++;
                }

            std::vector<ShortReal> vertex_radii(n_vert, sweep_radius);
            obbs[i] = hpmc::detail::compute_obb(face_vec, vertex_radii, false);
            obbs[i].mask = face_overlap[i];
            internal_coordinates.push_back(face_vec);
            }

        OBBTree tree_obb;
        tree_obb.buildTree(obbs, internal_coordinates, sweep_radius, n_faces, leaf_capacity);
        tree = GPUTree(tree_obb, managed);
        delete[] obbs;

        // set the diameter
        diameter = 2 * (sqrt(radius_sq) + sweep_radius);
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        pybind11::list face_list;

        for (unsigned int i = 0; i < n_faces; i++)
            {
            pybind11::list face_vert;
            face_vert.append(face_verts[i * 3]);
            face_vert.append(face_verts[i * 3 + 1]);
            face_vert.append(face_verts[i * 3 + 2]);
            face_list.append(pybind11::tuple(face_vert));
            }

        pybind11::list vert_list;
        for (unsigned int i = 0; i < n_verts; i++)
            {
            pybind11::list vert;
            vert.append(verts[i].x);
            vert.append(verts[i].y);
            vert.append(verts[i].z);
            vert_list.append(pybind11::tuple(vert));
            }

        pybind11::list overlap_list;
        for (unsigned int i = 0; i < face_overlap.size(); i++)
            {
            overlap_list.append(face_overlap[i]);
            }

        pybind11::list origin_list;
        origin_list.append(origin.x);
        origin_list.append(origin.y);
        origin_list.append(origin.z);

        v["vertices"] = vert_list;
        v["faces"] = face_list;
        v["overlap"] = overlap_list;
        v["sweep_radius"] = sweep_radius;
        v["ignore_statistics"] = ignore;
        v["capacity"] = tree.getLeafNodeCapacity();
        v["origin"] = pybind11::tuple(origin_list);
        v["hull_only"] = hull_only;
        return v;
        }

#endif

    /// Tree for fast locality lookups
    GPUTree tree;

    /// Vertex coordinates
    ManagedArray<vec3<ShortReal>> verts;

    /// Offset of every face in the list of vertices per face
    ManagedArray<unsigned int> face_offs;

    /// Ordered vertex IDs of every face
    ManagedArray<unsigned int> face_verts;

    /// Overlap mask per face
    ManagedArray<unsigned int> face_overlap;

    /// Number of vertices
    unsigned int n_verts;

    /// Number of faces
    unsigned int n_faces;

    /// True when move statistics should not be counted
    unsigned int ignore;

    /// Origin point inside the shape
    vec3<ShortReal> origin;

    /// If 1, only the hull of the shape is considered for overlaps
    unsigned int hull_only;

    /// Radius of a sweeping sphere
    ShortReal sweep_radius;

    /// Pre-calculated diameter
    ShortReal diameter;

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
        {
        tree.load_shared(ptr, available_bytes);
        verts.load_shared(ptr, available_bytes);
        face_offs.load_shared(ptr, available_bytes);
        face_verts.load_shared(ptr, available_bytes);
        face_overlap.load_shared(ptr, available_bytes);
        }

    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        tree.allocate_shared(ptr, available_bytes);
        verts.allocate_shared(ptr, available_bytes);
        face_offs.allocate_shared(ptr, available_bytes);
        face_verts.allocate_shared(ptr, available_bytes);
        face_overlap.allocate_shared(ptr, available_bytes);
        }

#ifdef ENABLE_HIP
    void set_memory_hint() const
        {
        tree.set_memory_hint();
        verts.set_memory_hint();
        face_offs.set_memory_hint();
        face_verts.set_memory_hint();
        face_overlap.set_memory_hint();
        }
#endif
    } __attribute__((aligned(32)));
    }; // end namespace detail

/** General polyhedron shape

    Implement the HPMC shape interface for general polyhedra.
*/
struct ShapePolyhedron
    {
    //. Define the parameter type
    typedef detail::TriangleMesh param_type;

    /// Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapePolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), data(_params), tree(_params.tree)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation()
        {
        return data.n_verts > 1;
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return data.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return data.diameter;
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    /// Return true if this is a sphero-shape
    DEVICE ShortReal isSpheroPolyhedron() const
        {
        return data.sweep_radius != ShortReal(0.0);
        }

#ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        unsigned int n_verts = data.n_verts;
        unsigned int n_faces = data.n_faces;

        if (n_verts == 0)
            {
            throw std::runtime_error("Shape definition not supported for 0-vertex polyhedra.");
            }

        std::ostringstream shapedef;
        if (n_verts == 1 && data.verts[0].x == 0.0f && data.verts[0].y == data.verts[0].x
            && data.verts[0].y == data.verts[0].z)
            {
            shapedef << "{\"type\": \"Sphere\", \"diameter\": "
                     << data.sweep_radius * ShortReal(2.0) << "}";
            }
        else
            {
            shapedef << "{\"type\": \"Mesh\", \"vertices\": [";
            for (unsigned int i = 0; i < n_verts - 1; i++)
                {
                shapedef << "[" << data.verts[i].x << ", " << data.verts[i].y << ", "
                         << data.verts[i].z << "], ";
                }
            shapedef << "[" << data.verts[n_verts - 1].x << ", " << data.verts[n_verts - 1].y
                     << ", " << data.verts[n_verts - 1].z << "]], \"indices\": [";
            unsigned int nverts_face, offset;
            for (unsigned int i = 0; i < n_faces; i++)
                {
                // Number of vertices of ith face
                nverts_face = data.face_offs[i + 1] - data.face_offs[i];
                offset = data.face_offs[i];
                shapedef << "[";
                for (unsigned int j = 0; j < nverts_face - 1; j++)
                    {
                    shapedef << data.face_verts[offset + j] << ", ";
                    }
                shapedef << data.face_verts[offset + nverts_face - 1];
                if (i == n_faces - 1)
                    shapedef << "]]}";
                else
                    shapedef << "], ";
                }
            }
        return shapedef.str();
        }
#endif

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return hoomd::detail::AABB(pos, data.diameter / Scalar(2));
        }

    /// Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
        }

    /// Returns true if this shape splits the overlap check over several threads of a warp using
    /// threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
#ifdef LEAVES_AGAINST_TREE_TRAVERSAL
        return true;
#else
        return false;
#endif
        }

    /// Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return false;
        }

    /// Orientation of the shape
    quat<Scalar> orientation;

    /// Vertices
    const detail::TriangleMesh& data;

    /// Tree for particle features
    const detail::GPUTree& tree;
    };

DEVICE inline ShortReal
det_4x4(vec3<ShortReal> a, vec3<ShortReal> b, vec3<ShortReal> c, vec3<ShortReal> d)
    {
    return dot(cross(c, d), b - a) + dot(cross(a, b), d - c);
    }

// Clamp n to lie within the range [min, max]
DEVICE inline ShortReal clamp(ShortReal n, ShortReal min, ShortReal max)
    {
    if (n < min)
        return min;
    if (n > max)
        return max;
    return n;
    }

/** From Real Time Collision Detection (Christer Ericson)
   Computes closest points C1 and C2 of S1(s)=P1+s*(Q1-P1) and
   S2(t)=P2+t*(Q2-P2), returning s and t. Function result is squared
   distance between between S1(s) and S2(t)
*/
DEVICE inline ShortReal closestPtSegmentSegment(const vec3<ShortReal> p1,
                                                const vec3<ShortReal>& q1,
                                                const vec3<ShortReal>& p2,
                                                const vec3<ShortReal>& q2,
                                                ShortReal& s,
                                                ShortReal& t,
                                                vec3<ShortReal>& c1,
                                                vec3<ShortReal>& c2,
                                                ShortReal abs_tol)
    {
    vec3<ShortReal> d1 = q1 - p1; // Direction vector of segment S1
    vec3<ShortReal> d2 = q2 - p2; // Direction vector of segment S2
    vec3<ShortReal> r = p1 - p2;
    ShortReal a = dot(d1, d1); // Squared length of segment S1, always nonnegative
    ShortReal e = dot(d2, d2); // Squared length of segment S2, always nonnegative
    ShortReal f = dot(d2, r);

    // Check if either or both segments degenerate into points
    if (CHECK_ZERO(a, abs_tol) && CHECK_ZERO(e, abs_tol))
        {
        // Both segments degenerate into points
        s = t = ShortReal(0.0);
        c1 = p1;
        c2 = p2;
        return dot(c1 - c2, c1 - c2);
        }

    if (CHECK_ZERO(a, abs_tol))
        {
        // First segment degenerates into a point
        s = ShortReal(0.0);
        t = f / e; // s = 0 => t = (b*s + f) / e = f / e
        t = clamp(t, ShortReal(0.0), ShortReal(1.0));
        }
    else
        {
        ShortReal c = dot(d1, r);

        if (CHECK_ZERO(e, abs_tol))
            {
            // Second segment degenerates into a point
            t = ShortReal(0.0);
            s = clamp(-c / a,
                      ShortReal(0.0),
                      ShortReal(1.0)); // t = 0 => s = (b*t - c) / a = -c / a
            }
        else
            {
            // The general nondegenerate case starts here
            ShortReal b = dot(d1, d2);
            ShortReal denom = a * e - b * b; // Always nonnegative

            // If segments not parallel, compute closest point on L1 to L2 and
            // clamp to segment S1. Else pick arbitrary s (here 0)
            if (denom != ShortReal(0.0))
                {
                s = clamp((b * f - c * e) / denom, ShortReal(0.0), ShortReal(1.0));
                }
            else
                s = ShortReal(0.0);

            // Compute point on L2 closest to S1(s) using
            // t = dot((P1 + D1*s) - P2,D2) / dot(D2,D2) = (b*s + f) / e
            t = (b * s + f) / e;
            // If t in [0,1] done. Else clamp t, recompute s for the new value
            // of t using s = dot((P2 + D2*t) - P1,D1) / dot(D1,D1)= (t*b - c) / a
            // and clamp s to [0, 1]
            if (t < ShortReal(0.0))
                {
                t = ShortReal(0.0);
                s = clamp(-c / a, ShortReal(0.0), ShortReal(1.0));
                }
            else if (t > ShortReal(1.0))
                {
                t = ShortReal(1.0);
                s = clamp((b - c) / a, ShortReal(0.0), ShortReal(1.0));
                }
            }
        }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return dot(c1 - c2, c1 - c2);
    }

/** Test if a point lies on a line segment
    @param v The vertex coordinates
    @param a First point of line segment
    @param b Second point of line segment
 */
DEVICE inline bool test_vertex_line_segment_overlap(const vec3<ShortReal>& v,
                                                    const vec3<ShortReal>& a,
                                                    const vec3<ShortReal>& b,
                                                    ShortReal abs_tol)
    {
    vec3<ShortReal> c = cross(v - a, b - a);
    ShortReal d = dot(v - a, b - a) / dot(b - a, b - a);
    return (CHECK_ZERO(dot(c, c), abs_tol) && d >= ShortReal(0.0) && d <= ShortReal(1.0));
    }
/** Test for intersection of line segments in 3D

    @param p Support vertex of first line segment
    @param q Support vertex of second line segment
    @param a Vector between endpoints of first line segment
    @param b Vector between endpoints of second line segment
    @returns true if line segments intersect or overlap
*/
DEVICE inline bool test_line_segment_overlap(const vec3<ShortReal>& p,
                                             const vec3<ShortReal>& q,
                                             const vec3<ShortReal>& a,
                                             const vec3<ShortReal>& b,
                                             ShortReal abs_tol)
    {
    /*
    if (det_4x4(p,q,p+a,q+b))
        return false; // line segments are skew
    */
    // 2d coordinate frame ex,ey
    vec3<ShortReal> ex = a;
    ShortReal mag_r = fast::sqrt(dot(ex, ex));
    ex /= mag_r;
    vec3<ShortReal> ey = cross(ex, b);
    ey = cross(ey, ex);

    if (dot(ey, ey) != 0)
        ey *= fast::rsqrt(dot(ey, ey));

    vec2<ShortReal> r(mag_r, ShortReal(0.0));
    vec2<ShortReal> s(dot(b, ex), dot(b, ey));
    ShortReal denom = (r.x * s.y - r.y * s.x);
    vec2<ShortReal> del(dot(q - p, ex), dot(q - p, ey));

    if (CHECK_ZERO(denom, abs_tol))
        {
        // collinear or parallel?
        vec3<ShortReal> c = cross(q - p, a);
        if (dot(c, c) != 0)
            return false; // parallel

        ShortReal t = dot(del, r);
        ShortReal u = -dot(del, s);
        if ((t < 0 || t > dot(r, r)) && (u < 0 || u > dot(s, s)))
            return false; // collinear, disjoint

        // collinear, overlapping
        return true;
        }

    ShortReal t = (del.x * s.y - del.y * s.x) / denom;
    ShortReal u = (del.x * r.y - del.y * r.x) / denom;

    if (t >= ShortReal(0.0) && t <= ShortReal(1.0) && u >= ShortReal(0.0) && u <= ShortReal(1.0))
        {
        // intersection
        return true;
        }
    return false;
    }

/** compute shortest distance between two triangles
    @returns square of shortest distance
*/
DEVICE inline ShortReal shortest_distance_triangles(const vec3<ShortReal>& a1,
                                                    const vec3<ShortReal>& b1,
                                                    const vec3<ShortReal>& c1,
                                                    const vec3<ShortReal>& a2,
                                                    const vec3<ShortReal>& b2,
                                                    const vec3<ShortReal>& c2,
                                                    ShortReal abs_tol)
    {
    // nine pairs of edges
    ShortReal dmin_sq(FLT_MAX);
    vec3<ShortReal> p1, p2;
    ShortReal s, t;
    ShortReal dsq;
    dsq = closestPtSegmentSegment(a1, b1, a2, b2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(a1, b1, a2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(a1, b1, b2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(a1, c1, a2, b2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(a1, c1, a2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(a1, c1, b2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(b1, c1, a2, b2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(b1, c1, a2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    dsq = closestPtSegmentSegment(b1, c1, b2, c2, s, t, p1, p2, abs_tol);
    if (dsq < dmin_sq)
        dmin_sq = dsq;

    // six vertex-triangle distances
    vec3<ShortReal> p;
    p = detail::closestPointOnTriangle(a1, a2, b2, c2);
    dsq = dot(p - a1, p - a1);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    p = detail::closestPointOnTriangle(b1, a2, b2, c2);
    dsq = dot(p - b1, p - b1);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    p = detail::closestPointOnTriangle(c1, a2, b2, c2);
    dsq = dot(p - c1, p - c1);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    p = detail::closestPointOnTriangle(a2, a1, b1, c1);
    dsq = dot(p - a2, p - a2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    p = detail::closestPointOnTriangle(b2, a1, b1, c1);
    dsq = dot(p - b2, p - b2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    p = detail::closestPointOnTriangle(c2, a1, b1, c1);
    dsq = dot(p - c2, p - c2);
    if (dsq < dmin_sq)
        dmin_sq = dsq;
    return dmin_sq;
    }

/** Test overlap in narrow phase
    @param dr separation vector between the particles, IN THE REFERENCE FRAME of b
    @param a first shape
    @param b second shape
    @param cur_node_a Node in a's tree to check
    @param cur_node_a Node in b's tree to check
    @param err gets incremented if there are errors (not currently implemented)
    @param abs_tol an absolute tolerance for the triangle triangle check
 */
DEVICE inline bool test_narrow_phase_overlap(vec3<ShortReal> dr,
                                             const ShapePolyhedron& a,
                                             const ShapePolyhedron& b,
                                             unsigned int cur_node_a,
                                             unsigned int cur_node_b,
                                             unsigned int& err,
                                             ShortReal abs_tol)
    {
    // loop through faces of cur_node_a
    unsigned int na = a.tree.getNumParticles(cur_node_a);
    unsigned int nb = b.tree.getNumParticles(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int iface = a.tree.getParticleByNode(cur_node_a, i);
        // Load number of face vertices
        unsigned int nverts_a = a.data.face_offs[iface + 1] - a.data.face_offs[iface];
        unsigned int offs_a = a.data.face_offs[iface];
        unsigned mask_a = a.data.face_overlap[iface];

        float U[3][3];
        quat<ShortReal> q(conj(quat<ShortReal>(b.orientation)) * quat<ShortReal>(a.orientation));
        if (nverts_a > 2)
            {
            for (unsigned int ivert = 0; ivert < 3; ++ivert)
                {
                unsigned int idx_a = a.data.face_verts[offs_a + ivert];
                vec3<float> v = a.data.verts[idx_a];
                v = rotate(quat<float>(q), v) + vec3<float>(dr);
                U[ivert][0] = v.x;
                U[ivert][1] = v.y;
                U[ivert][2] = v.z;
                }
            }

        // loop through faces of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int nverts_b, offs_b;
            unsigned int jface = b.tree.getParticleByNode(cur_node_b, j);
            // fetch next face of particle b
            nverts_b = b.data.face_offs[jface + 1] - b.data.face_offs[jface];
            offs_b = b.data.face_offs[jface];
            unsigned int mask_b = b.data.face_overlap[jface];
            // only check overlaps if required

            if (!(mask_a & mask_b))
                continue;
            if (nverts_a > 2 && nverts_b > 2)
                {
                float V[3][3];
                for (unsigned int ivert = 0; ivert < 3; ++ivert)
                    {
                    unsigned int idx_b = b.data.face_verts[offs_b + ivert];
                    vec3<float> v = b.data.verts[idx_b];
                    V[ivert][0] = v.x;
                    V[ivert][1] = v.y;
                    V[ivert][2] = v.z;
                    }
                // check collision between triangles
                if (NoDivTriTriIsect(V[0], V[1], V[2], U[0], U[1], U[2], abs_tol))
                    {
                    return true;
                    }
                }

            if (bool(a.isSpheroPolyhedron()) || bool(b.isSpheroPolyhedron()))
                {
                ShortReal dsqmin(FLT_MAX);
                // Load vertex 0 on a
                unsigned int idx_a = a.data.face_verts[offs_a];
                vec3<ShortReal> a0 = a.data.verts[idx_a];
                a0 = rotate(q, a0) + dr;
                // vertex 0 on b
                unsigned int idx_b = b.data.face_verts[offs_b];
                vec3<ShortReal> b0 = b.data.verts[idx_b];
                vec3<ShortReal> b1, b2;
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
                    vec3<ShortReal> p;
                    p = detail::closestPointOnTriangle(a0, b0, b1, b2);
                    dsqmin = dot(p - a0, p - a0);
                    }
                vec3<ShortReal> a1, a2;
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
                    vec3<ShortReal> p;
                    p = detail::closestPointOnTriangle(b0, a0, a1, a2);
                    dsqmin = dot(p - b0, p - b0);
                    }
                if (nverts_b > 1 && nverts_a > 1)
                    {
                    dsqmin = shortest_distance_triangles(a0, a1, a2, b0, b1, b2, abs_tol);
                    }
                if (nverts_a == 1 && nverts_b == 1)
                    {
                    // trivial case
                    dsqmin = dot(a0 - b0, a0 - b0);
                    }
                ShortReal R_ab = a.data.sweep_radius + b.data.sweep_radius;
                if (R_ab * R_ab >= dsqmin)
                    {
                    // overlap of spherotriangles
                    return true;
                    }
                }
            } // end loop over faces of b
        } // end loop over over faces of a
    return false;
    }

/** From Real-time Collision Detection (Christer Ericson)
    Given ray pq and triangle abc, returns whether segment intersects
    triangle and if so, also returns the barycentric coordinates (u,v,w)
    of the intersection point
    Note: the triangle is assumed to be oriented counter-clockwise when viewed from the direction of
   p
*/
DEVICE inline bool IntersectRayTriangle(const vec3<ShortReal>& p,
                                        const vec3<ShortReal>& q,
                                        const vec3<ShortReal>& a,
                                        const vec3<ShortReal>& b,
                                        const vec3<ShortReal>& c,
                                        ShortReal& u,
                                        ShortReal& v,
                                        ShortReal& w,
                                        ShortReal& t)
    {
    vec3<ShortReal> ab = b - a;
    vec3<ShortReal> ac = c - a;
    vec3<ShortReal> qp = p - q;
    // Compute triangle normal. Can be precalculated or cached if
    // intersecting multiple segments against the same triangle
    vec3<ShortReal> n = cross(ab, ac);
    // Compute denominator d. If d <= 0, segment is parallel to or points
    // away from triangle, so exit early
    float d = dot(qp, n);
    if (d <= ShortReal(0.0))
        return false;
    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    vec3<ShortReal> ap = p - a;
    t = dot(ap, n);
    if (t < ShortReal(0.0))
        return false;
    // For segment; exclude this code line for a ray test
    // Compute barycentric coordinate components and test if within bounds
    vec3<ShortReal> e = cross(qp, ap);
    v = dot(ac, e);
    if (v < ShortReal(0.0) || v > d)
        return false;
    w = -dot(ab, e);
    if (w < ShortReal(0.0) || v + w > d)
        return false;
    // Segment/ray intersects triangle. Perform delayed division and
    // compute the last barycentric coordinate component
    float ood = ShortReal(1.0) / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = ShortReal(1.0) - v - w;
    return true;
    }

#ifndef __HIPCC__
//! Traverse the bounding volume test tree recursively
inline bool BVHCollision(const ShapePolyhedron& a,
                         const ShapePolyhedron& b,
                         unsigned int cur_node_a,
                         unsigned int cur_node_b,
                         const quat<ShortReal>& q,
                         const vec3<ShortReal>& dr,
                         unsigned int& err,
                         ShortReal abs_tol)
    {
    detail::OBB obb_a = a.tree.getOBB(cur_node_a);
    obb_a.affineTransform(q, dr);
    detail::OBB obb_b = b.tree.getOBB(cur_node_b);
    if (!overlap(obb_a, obb_b))
        return false;
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

/** Polyhedron overlap test
    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a first shape
    @param b second shape
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius Additional sphere radius to sweep the shapes by
    @returns true when *a* and *b* overlap, and false when they are disjoint
*/
template<>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapePolyhedron& a,
                                const ShapePolyhedron& b,
                                unsigned int& err)
    {
    ShortReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    const ShortReal abs_tol(ShortReal(DaDb * 1e-12));
    vec3<ShortReal> dr = r_ab;

/*
 * This overlap test checks if an edge of one polyhedron is overlapping with a face of the other
 */
/*
 * This overlap test checks if either
 * a) an edge of one polyhedron intersects the face of the other
 * b) the center of mass of one polyhedron is contained in the other
 */
#ifdef __HIPCC__
    const detail::GPUTree& tree_a = a.tree;
    const detail::GPUTree& tree_b = b.tree;
#endif
#ifdef LEAVES_AGAINST_TREE_TRAVERSAL
#ifdef __HIPCC__
    // Parallel tree traversal
    unsigned int offset = threadIdx.x;
    unsigned int stride = blockDim.x;
#else
    unsigned int offset = 0;
    unsigned int stride = 1;
#endif
    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = offset; cur_leaf_a < tree_a.getNumLeaves();
             cur_leaf_a += stride)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);
            // rotate and translate a's obb into b's body frame
            vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
            obb_a.affineTransform(conj(b.orientation) * a.orientation, dr_rot);
            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b)
                    && test_narrow_phase_overlap(dr_rot,
                                                 a,
                                                 b,
                                                 cur_node_a,
                                                 query_node,
                                                 err,
                                                 abs_tol))
                    return true;
                }
            }
        }
    else
        {
        for (unsigned int cur_leaf_b = offset; cur_leaf_b < tree_b.getNumLeaves();
             cur_leaf_b += stride)
            {
            unsigned int cur_node_b = tree_b.getLeafNode(cur_leaf_b);
            hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);
            // rotate and translate b's obb into a's body frame
            vec3<ShortReal> dr_rot(rotate(conj(a.orientation), r_ab));
            obb_b.affineTransform(conj(a.orientation) * b.orientation, dr_rot);
            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a)
                    && test_narrow_phase_overlap(dr_rot,
                                                 b,
                                                 a,
                                                 cur_node_b,
                                                 query_node,
                                                 err,
                                                 abs_tol))
                    return true;
                }
            }
        }
#else

    vec3<ShortReal> dr_rot(rotate(conj(b.orientation), -r_ab));
    quat<ShortReal> q(conj(b.orientation) * a.orientation);

#ifndef __HIPCC__
    if (BVHCollision(a, b, 0, 0, q, dr_rot, err, abs_tol))
        return true;
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
        if (detail::traverseBinaryStack(tree_a,
                                        tree_b,
                                        cur_node_a,
                                        cur_node_b,
                                        stack,
                                        obb_a,
                                        obb_b,
                                        q,
                                        dr_rot)
            && test_narrow_phase_overlap(dr_rot, a, b, query_node_a, query_node_b, err, abs_tol))
            return true;
        }

#endif
#endif
    // no intersecting edge, check if one polyhedron is contained in the other
    // if shape(A) == shape(B), only consider intersections
    if (&a.data == &b.data)
        return false;
    for (unsigned int ord = 0; ord < 2; ++ord)
        {
        // load shape
        const ShapePolyhedron& s1 = (ord == 0) ? b : a;
        // if the shape is a hull only, skip
        if (s1.data.hull_only)
            continue;
        vec3<ShortReal> p;
        if (ord == 0)
            {
            p = -dr + rotate(quat<ShortReal>(a.orientation), a.data.origin);
            }
        else
            {
            p = dr + rotate(quat<ShortReal>(b.orientation), b.data.origin);
            }

        // Check if s0 is contained in s1 by shooting a ray from its origin
        // in direction of origin separation
        vec3<ShortReal> n = dr + rotate(quat<ShortReal>(b.orientation), b.data.origin)
                            - rotate(quat<ShortReal>(a.orientation), a.data.origin);
        // rotate ray in coordinate system of shape s1
        p = rotate(conj(quat<ShortReal>(s1.orientation)), p);
        n = rotate(conj(quat<ShortReal>(s1.orientation)), n);
        if (ord != 0)
            {
            n = -n;
            }

        vec3<ShortReal> q = p + n;
        unsigned int n_overlap = 0;
        // query ray against OBB tree
        unsigned cur_node_s1 = 0;
        while (cur_node_s1 < s1.tree.getNumNodes())
            {
            unsigned int query_node = cur_node_s1;
            if (s1.tree.queryRay(p, n, cur_node_s1, abs_tol))
                {
                unsigned int n_faces = s1.tree.getNumParticles(query_node);
                // loop through faces
                for (unsigned int j = 0; j < n_faces; j++)
                    {
                    // fetch next face
                    unsigned int jface = s1.tree.getParticleByNode(query_node, j);
                    unsigned int offs_b = s1.data.face_offs[jface];
                    if (s1.data.face_offs[jface + 1] - offs_b < 3)
                        continue;
                    // Load vertex 0
                    vec3<ShortReal> v_b[3];
                    unsigned int idx_v = s1.data.face_verts[offs_b];
                    v_b[0] = s1.data.verts[idx_v];
                    // vertex 1
                    idx_v = s1.data.face_verts[offs_b + 1];
                    v_b[1] = s1.data.verts[idx_v];
                    // vertex 2
                    idx_v = s1.data.face_verts[offs_b + 2];
                    v_b[2] = s1.data.verts[idx_v];
                    ShortReal u, v, w, t;
                    // two-sided triangle test
                    if (IntersectRayTriangle(p, q, v_b[0], v_b[1], v_b[2], u, v, w, t)
                        || IntersectRayTriangle(p, q, v_b[2], v_b[1], v_b[0], u, v, w, t))
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

#ifndef __HIPCC__
/// Return the shape parameters in the `type_shape` format
template<> inline std::string getShapeSpec(const ShapePolyhedron& s)
    {
    auto& data = s.data;
    unsigned int n_verts = data.n_verts;
    unsigned int n_faces = data.n_faces;
    std::ostringstream shapedef;
    shapedef << "{\"type\": \"Mesh\", \"vertices\": [";
    for (unsigned int i = 0; i < n_verts - 1; i++)
        {
        shapedef << "[" << data.verts[i].x << ", " << data.verts[i].y << ", " << data.verts[i].z
                 << "], ";
        }
    shapedef << "[" << data.verts[n_verts - 1].x << ", " << data.verts[n_verts - 1].y << ", "
             << data.verts[n_verts - 1].z << "]], \"indices\": [";

    unsigned int nverts_face, offset;
    for (unsigned int i = 0; i < n_faces; i++)
        {
        // Number of vertices of ith face
        nverts_face = data.face_offs[i + 1] - data.face_offs[i];
        offset = data.face_offs[i];
        shapedef << "[";
        for (unsigned int j = 0; j < nverts_face - 1; j++)
            {
            shapedef << data.face_verts[offset + j] << ", ";
            }
        shapedef << data.face_verts[offset + nverts_face - 1];
        if (i == n_faces - 1)
            shapedef << "]]}";
        else
            shapedef << "], ";
        }
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd
#undef DEVICE
#undef HOSTDEVICE
