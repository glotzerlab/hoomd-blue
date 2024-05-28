// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "OBB.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSphere.h"
#include "XenoCollide3D.h"
#include "hoomd/AABB.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/** The faceted ellipsoid is defined by the intersection of an ellipsoid of half axes a,b,c with
 * planes given by the face normals n and offsets b, obeying the equation:
 *
 * r.n + b <= 0
 *
 * Intersections of planes within the ellipsoid volume are accounted for.
 * Internally, the overlap check works by computing the support function for an ellipsoid
 * deformed into a unit sphere.
 *
 */
struct FacetedEllipsoidParams : ShapeParams
    {
    /// Empty constructor
    DEVICE FacetedEllipsoidParams()
        : verts(), additional_verts(), n(), offset(), a(1.0), b(1.0), c(1.0), N(0), ignore(1)
        {
        }

#ifndef __HIPCC__
    /// Construct a faceted ellipsoid with n_facet facets
    FacetedEllipsoidParams(unsigned int n_facet, bool managed)
        : a(1.0), b(1.0), c(1.0), N(n_facet), ignore(0)
        {
        n = ManagedArray<vec3<ShortReal>>(n_facet, managed);
        offset = ManagedArray<ShortReal>(n_facet, managed);
        }

    /// Construct from a Python dictionary
    FacetedEllipsoidParams(pybind11::dict v, bool managed = false)
        : FacetedEllipsoidParams((unsigned int)pybind11::len(v["normals"]), managed)
        {
        pybind11::list normals = v["normals"];
        pybind11::list offsets = v["offsets"];
        pybind11::object vertices = v["vertices"];
        a = v["a"].cast<ShortReal>();
        b = v["b"].cast<ShortReal>();
        c = v["c"].cast<ShortReal>();
        pybind11::tuple origin_tuple = v["origin"];
        ignore = v["ignore_statistics"].cast<unsigned int>();

        if (a <= 0.0f || b <= 0.0f || c <= 0.0f)
            {
            throw std::domain_error("All semimajor axes must be nonzero!");
            }

        if (pybind11::len(offsets) != pybind11::len(normals))
            throw std::runtime_error("Number of normals unequal number of offsets");

        // extract the normals from the python list
        for (unsigned int i = 0; i < len(normals); i++)
            {
            pybind11::list normals_i = normals[i];
            if (len(normals_i) != 3)
                throw std::runtime_error("Each normal must have 3 elements: found "
                                         + pybind11::str(normals_i).cast<std::string>() + " in "
                                         + pybind11::str(normals).cast<std::string>());
            n[i] = vec3<ShortReal>(pybind11::cast<ShortReal>(normals_i[0]),
                                   pybind11::cast<ShortReal>(normals_i[1]),
                                   pybind11::cast<ShortReal>(normals_i[2]));
            offset[i] = pybind11::cast<ShortReal>(offsets[i]);
            }

        // extract the vertices from the python list
        pybind11::list vertices_list;
        if (!vertices.is_none())
            {
            vertices_list = pybind11::list(vertices);
            }
        // when vertices is None, pass an empty list to PolyhedronVertices

        pybind11::dict verts_dict;
        verts_dict["vertices"] = vertices_list;
        verts_dict["sweep_radius"] = 0;
        verts_dict["ignore_statistics"] = ignore;
        verts = PolyhedronVertices(verts_dict, managed);

        // scale vertices onto the surface of the ellipsoid
        for (unsigned int i = 0; i < verts.N; ++i)
            {
            verts.x[i] /= a;
            verts.y[i] /= b;
            verts.z[i] /= c;
            }

        // set the origin
        origin = vec3<ShortReal>(pybind11::cast<ShortReal>(origin_tuple[0]),
                                 pybind11::cast<ShortReal>(origin_tuple[1]),
                                 pybind11::cast<ShortReal>(origin_tuple[2]));

        // add the edge-sphere vertices
        initializeVertices(managed);
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::dict v;
        pybind11::list vertices = verts.asDict()["vertices"];
        pybind11::list offsets;
        pybind11::list normals;

        for (unsigned int i = 0; i < pybind11::len(vertices); i++)
            {
            pybind11::list vert_i = vertices[i];
            pybind11::list vert;
            ShortReal x = vert_i[0].cast<ShortReal>();
            ShortReal y = vert_i[1].cast<ShortReal>();
            ShortReal z = vert_i[2].cast<ShortReal>();
            vert.append(x * a);
            vert.append(y * b);
            vert.append(z * c);
            vertices[i] = pybind11::tuple(vert);
            }

        for (unsigned int i = 0; i < offset.size(); i++)
            {
            offsets.append(offset[i]);

            vec3<ShortReal> normal_i = n[i];
            pybind11::list normal_i_list;
            normal_i_list.append(normal_i.x);
            normal_i_list.append(normal_i.y);
            normal_i_list.append(normal_i.z);
            normals.append(pybind11::tuple(normal_i_list));
            }

        pybind11::list origin_list;
        origin_list.append(origin.x);
        origin_list.append(origin.y);
        origin_list.append(origin.z);
        pybind11::tuple origin_tuple = pybind11::tuple(origin_list);

        v["vertices"] = vertices;
        v["normals"] = normals;
        v["offsets"] = offsets;
        v["a"] = a;
        v["b"] = b;
        v["c"] = c;
        v["origin"] = origin_tuple;
        v["ignore_statistics"] = ignore;
        return v;
        }

    /// Generate the intersections points of polyhedron edges with the sphere
    DEVICE void initializeVertices(bool managed = false)
        {
        const Scalar tolerance = 1e-5;
        additional_verts = detail::PolyhedronVertices(2 * N * N, managed);
        additional_verts.diameter = ShortReal(2.0); // for unit sphere
        additional_verts.N = 0;

        // iterate over unique pairs of planes
        for (unsigned int i = 0; i < N; ++i)
            {
            vec3<ShortReal> n_p(n[i]);
            // transform plane normal into the coordinate system of the unit sphere
            n_p.x *= a;
            n_p.y *= b;
            n_p.z *= c;

            ShortReal b(offset[i]);

            for (unsigned int j = i + 1; j < N; ++j)
                {
                vec3<ShortReal> np2(n[j]);
                // transform plane normal into the coordinate system of the unit sphere
                np2.x *= a;
                np2.y *= b;
                np2.z *= c;

                ShortReal b2 = offset[j];
                ShortReal np2_sq = dot(np2, np2);

                // determine intersection line between plane i and plane j

                // point on intersection line closest to center of sphere
                ShortReal dotp = dot(np2, n_p);
                ShortReal denom = dotp * dotp - dot(n_p, n_p) * np2_sq;
                ShortReal lambda0 = ShortReal(2.0) * (b2 * dotp - b * np2_sq) / denom;
                ShortReal lambda1 = ShortReal(2.0) * (-b2 * dot(n_p, n_p) + b * dotp) / denom;

                vec3<ShortReal> r = -(lambda0 * n_p + lambda1 * np2) / ShortReal(2.0);

                // if the line does not intersect the sphere, ignore
                if (dot(r, r) > ShortReal(1.0))
                    continue;

                // the line intersects with the sphere at two points, one of
                // which maximizes the support function
                vec3<ShortReal> c01 = cross(n_p, np2);
                ShortReal s = fast::sqrt((ShortReal(1.0) - dot(r, r)) * dot(c01, c01));

                ShortReal t0 = (-dot(r, c01) - s) / dot(c01, c01);
                ShortReal t1 = (-dot(r, c01) + s) / dot(c01, c01);

                vec3<ShortReal> v1 = r + t0 * c01;
                vec3<ShortReal> v2 = r + t1 * c01;

                // check first point
                bool allowed = true;
                for (unsigned int k = 0; k < N; ++k)
                    {
                    if (k == i || k == j)
                        continue;

                    vec3<ShortReal> np3(n[k]);
                    // transform plane normal into the coordinate system of the unit sphere
                    np3.x *= a;
                    np3.y *= b;
                    np3.z *= c;

                    ShortReal b3(offset[k]);

                    // is this vertex inside the volume bounded by all halfspaces?
                    if (dot(np3, v1) + b3 > ShortReal(0.0))
                        {
                        allowed = false;
                        break;
                        }
                    }

                if (allowed && dot(v1, v1) <= ShortReal(1.0 + tolerance))
                    {
                    additional_verts.x[additional_verts.N] = v1.x;
                    additional_verts.y[additional_verts.N] = v1.y;
                    additional_verts.z[additional_verts.N] = v1.z;
                    additional_verts.N++;
                    }

                // check second point
                allowed = true;
                for (unsigned int k = 0; k < N; ++k)
                    {
                    if (k == i || k == j)
                        continue;

                    vec3<ShortReal> np3(n[k]);
                    // transform plane normal into the coordinate system of the unit sphere
                    np3.x *= a;
                    np3.y *= b;
                    np3.z *= c;

                    ShortReal b3(offset[k]);

                    // is this vertex inside the volume bounded by all halfspaces?
                    if (dot(np3, v2) + b3 > ShortReal(0.0))
                        {
                        allowed = false;
                        break;
                        }
                    }

                if (allowed && dot(v2, v2) <= (ShortReal(1.0 + tolerance)))
                    {
                    additional_verts.x[additional_verts.N] = v2.x;
                    additional_verts.y[additional_verts.N] = v2.y;
                    additional_verts.z[additional_verts.N] = v2.z;
                    additional_verts.N++;
                    }
                }
            }
        }

#endif

    /// Vertices of the polyhedron
    PolyhedronVertices verts;

    /// Vertices of the polyhedron edge-sphere intersection
    PolyhedronVertices additional_verts;

    /// Normal vectors of planes
    ManagedArray<vec3<ShortReal>> n;

    /// Offset of every plane
    ManagedArray<ShortReal> offset;

    /// First half-axis
    ShortReal a;

    /// Second half-axis
    ShortReal b;

    /// Third half-axis
    ShortReal c;

    /// Origin shift
    vec3<ShortReal> origin;

    /// Number of cut planes
    unsigned int N;

    /// True when move statistics should not be counted
    unsigned int ignore;

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
        {
        n.load_shared(ptr, available_bytes);
        offset.load_shared(ptr, available_bytes);
        verts.load_shared(ptr, available_bytes);
        additional_verts.load_shared(ptr, available_bytes);
        }

    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        n.allocate_shared(ptr, available_bytes);
        offset.allocate_shared(ptr, available_bytes);
        verts.allocate_shared(ptr, available_bytes);
        additional_verts.allocate_shared(ptr, available_bytes);
        }

#ifdef ENABLE_HIP
    void set_memory_hint() const
        {
        n.set_memory_hint();
        offset.set_memory_hint();
        verts.set_memory_hint();
        additional_verts.set_memory_hint();
        }
#endif
    } __attribute__((aligned(32)));

/// Support function for ShapeFacetedEllipsoid
class SupportFuncFacetedEllipsoid
    {
    public:
    /** Construct a support function for a faceted ellipsoid

        @param _params Parameters of the faceted ellipsoid
        @param _sweep_radius additional sweep radius
    */
    DEVICE SupportFuncFacetedEllipsoid(const FacetedEllipsoidParams& _params,
                                       const ShortReal& _sweep_radius = ShortReal(0.0))
        : params(_params), sweep_radius(_sweep_radius)
        {
        }

    DEVICE inline bool isInside(const vec3<ShortReal>& v, unsigned int plane) const
        {
        // is this vertex masked by a plane?
        vec3<ShortReal> np = params.n[plane];

        // transform plane normal into the coordinate system of the unit sphere
        // so that dot(np,v) has dimensions of b [length], since v is a unit vector
        np.x *= params.a;
        np.y *= params.b;
        np.z *= params.c;
        ShortReal b = params.offset[plane];

        // is current supporting vertex inside the half-space defined by this plane?
        return (dot(np, v) + b <= ShortReal(0.0));
        }

    /** Compute the support function

        @param n Normal vector input (in the local frame)
        @returns Local coords of the point furthest in the direction of n
    */
    DEVICE vec3<ShortReal> operator()(const vec3<ShortReal>& n_in) const
        {
        const Scalar tolerance = 1e-5;

        // transform support direction into coordinate system of the unit sphere
        vec3<ShortReal> n(n_in);
        n.x *= params.a;
        n.y *= params.b;
        n.z *= params.c;

        ShortReal nsq = dot(n, n);
        vec3<ShortReal> n_sphere = n * fast::rsqrt(nsq);

        vec3<ShortReal> max_vec = n_sphere;
        bool have_vertex = true;

        unsigned int n_planes = params.N;
        for (unsigned int i = 0; i < n_planes; ++i)
            {
            if (!isInside(max_vec, i))
                {
                have_vertex = false;
                break;
                }
            }

        // iterate over intersecting planes
        for (unsigned int i = 0; i < n_planes; i++)
            {
            vec3<ShortReal> n_p = params.n[i];
            // transform plane normal into the coordinate system of the unit sphere
            n_p.x *= params.a;
            n_p.y *= params.b;
            n_p.z *= params.c;

            ShortReal np_sq = dot(n_p, n_p);
            ShortReal b = params.offset[i];

            // compute supporting vertex on intersection boundary (circle)
            // between plane and sphere
            ShortReal alpha = dot(n_sphere, n_p);
            ShortReal arg = (ShortReal(1.0) - alpha * alpha / np_sq);
            vec3<ShortReal> v;
            if (arg >= ShortReal(tolerance))
                {
                ShortReal arg2 = ShortReal(1.0) - b * b / np_sq;
                ShortReal invgamma = fast::sqrt(arg2 / arg);

                // Intersection vertex that maximizes support function
                v = invgamma * (n_sphere - alpha / np_sq * n_p) - n_p * b / np_sq;
                }
            else
                {
                // degenerate case
                v = -b * n_p / np_sq;
                }

            bool valid = true;
            for (unsigned int j = 0; j < n_planes; ++j)
                {
                if (i != j && !isInside(v, j))
                    {
                    valid = false;
                    break;
                    }
                }

            if (valid && (!have_vertex || dot(v, n) > dot(max_vec, n)))
                {
                max_vec = v;
                have_vertex = true;
                }
            }

        // plane-plane-sphere intersection vertices
        if (params.additional_verts.N)
            {
            detail::SupportFuncConvexPolyhedron s(params.additional_verts);
            vec3<ShortReal> v = s(n);

            if (!have_vertex || dot(v, n) > dot(max_vec, n))
                {
                max_vec = v;
                have_vertex = true;
                }
            }

        // plane-plane intersections from user input
        if (params.verts.N)
            {
            detail::SupportFuncConvexPolyhedron s(params.verts);
            vec3<ShortReal> v = s(n);
            if (dot(v, v) <= ShortReal(1.0) && (!have_vertex || dot(v, n) > dot(max_vec, n)))
                {
                max_vec = v;
                have_vertex = true;
                }
            }

        // transform vertex on unit sphere back onto ellipsoid surface
        max_vec.x *= params.a;
        max_vec.y *= params.b;
        max_vec.z *= params.c;

        // origin shift
        max_vec -= params.origin;

        // extend out by sweep radius
        return max_vec + (sweep_radius * fast::rsqrt(dot(n_in, n_in))) * n_in;
        }

    private:
    /// Definition of faceted ellipsoid
    const FacetedEllipsoidParams& params;

    /// The radius of a sphere sweeping the shape
    const ShortReal sweep_radius;
    };

    } // end namespace detail

/** Faceted ellipsoid shape

    Implement the HPMC shape interface for a faceted ellipsoid.
*/
struct ShapeFacetedEllipsoid
    {
    /// Define the parameter type
    typedef detail::FacetedEllipsoidParams param_type;

    //! Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE ShapeFacetedEllipsoid(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), params(_params)
        {
        }

    /// Check if the shape may be rotated
    DEVICE bool hasOrientation()
        {
        return (params.N > 0) || (params.a != params.b) || (params.a != params.c)
               || (params.b != params.c);
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return params.ignore;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        return ShortReal(2) * detail::max(params.a, detail::max(params.b, params.c));
        }

    /// Get the in-sphere radius of the shape
    DEVICE ShortReal getInsphereRadius() const
        {
        return 0.0;
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // use support function of the ellipsoid to determine the furthest extent in each direction
        detail::SupportFuncFacetedEllipsoid sfunc(params);

        vec3<ShortReal> e_x(1, 0, 0);
        vec3<ShortReal> e_y(0, 1, 0);
        vec3<ShortReal> e_z(0, 0, 1);
        quat<ShortReal> q(orientation);
        vec3<ShortReal> s_x_plus
            = rotate(q, sfunc(rotate(conj(q), e_x)) + vec3<ShortReal>(params.origin));
        vec3<ShortReal> s_y_plus
            = rotate(q, sfunc(rotate(conj(q), e_y)) + vec3<ShortReal>(params.origin));
        vec3<ShortReal> s_z_plus
            = rotate(q, sfunc(rotate(conj(q), e_z)) + vec3<ShortReal>(params.origin));
        vec3<ShortReal> s_x_minus
            = rotate(q, sfunc(rotate(conj(q), -e_x)) + vec3<ShortReal>(params.origin));
        vec3<ShortReal> s_y_minus
            = rotate(q, sfunc(rotate(conj(q), -e_y)) + vec3<ShortReal>(params.origin));
        vec3<ShortReal> s_z_minus
            = rotate(q, sfunc(rotate(conj(q), -e_z)) + vec3<ShortReal>(params.origin));

        // translate out from the position by the furthest extent
        vec3<Scalar> upper(pos.x + s_x_plus.x, pos.y + s_y_plus.y, pos.z + s_z_plus.z);
        vec3<Scalar> lower(pos.x + s_x_minus.x, pos.y + s_y_minus.y, pos.z + s_z_minus.z);

        return hoomd::detail::AABB(lower, upper);
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        detail::SupportFuncFacetedEllipsoid sfunc(params);
        vec3<ShortReal> e_x(1, 0, 0);
        vec3<ShortReal> e_y(0, 1, 0);
        vec3<ShortReal> e_z(0, 0, 1);
        vec3<ShortReal> s_x_minus = sfunc(-e_x) + vec3<ShortReal>(params.origin);
        vec3<ShortReal> s_y_minus = sfunc(-e_y) + vec3<ShortReal>(params.origin);
        vec3<ShortReal> s_z_minus = sfunc(-e_z) + vec3<ShortReal>(params.origin);
        vec3<ShortReal> s_x_plus = sfunc(e_x) + vec3<ShortReal>(params.origin);
        vec3<ShortReal> s_y_plus = sfunc(e_y) + vec3<ShortReal>(params.origin);
        vec3<ShortReal> s_z_plus = sfunc(e_z) + vec3<ShortReal>(params.origin);

        detail::OBB obb;
        obb.center = pos;
        obb.rotation = orientation;
        obb.lengths.x = detail::max(s_x_plus.x, -s_x_minus.x);
        obb.lengths.y = detail::max(s_y_plus.y, -s_y_minus.y);
        obb.lengths.z = detail::max(s_z_plus.z, -s_z_minus.z);
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

    /// Faceted sphere parameters
    const param_type& params;
    };

/** Test overlap of faceted ellipsoids

    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a first shape
    @param b second shape
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius Additional sphere radius to sweep the shapes with
    @returns true when *a* and *b* overlap, and false when they are disjoint
*/
template<>
DEVICE inline bool
test_overlap<ShapeFacetedEllipsoid, ShapeFacetedEllipsoid>(const vec3<Scalar>& r_ab,
                                                           const ShapeFacetedEllipsoid& a,
                                                           const ShapeFacetedEllipsoid& b,
                                                           unsigned int& err)
    {
    vec3<ShortReal> dr(r_ab);

    ShortReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return detail::xenocollide_3d(
        detail::SupportFuncFacetedEllipsoid(a.params),
        detail::SupportFuncFacetedEllipsoid(b.params),
        rotate(conj(quat<ShortReal>(a.orientation)),
               dr + rotate(quat<ShortReal>(b.orientation), b.params.origin))
            - a.params.origin,
        conj(quat<ShortReal>(a.orientation)) * quat<ShortReal>(b.orientation),
        DaDb / ShortReal(2.0),
        err);
    }

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
