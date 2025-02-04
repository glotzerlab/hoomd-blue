// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "ShapeSphere.h"
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

#include "SphinxOverlap.h"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/// Maximum number of sphere centers that
const unsigned int MAX_SPHINX_SPHERE_CENTERS = 8;

/** Sphinx particle parameters

    A Sphinx particle is represented by N spheres each with their own diameter.

    ShapeSphinx represents the intersection of spheres. A  positive sphere is one where the volume
    inside the sphere is considered, and a negative sphere is one where the volume outside the
    sphere is considered. This shape is defined using a struct called SphinxParams whose detail
    is named spheres. ShapeSphinx requires an incoming orientation vector, followed by parameters
    for the struct SphinxParams. The parameter defining a sphinx is a structure containing the
    circumsphere diameter of all spheres defined in the shape. This is followed by the number of
    spheres in the shape, their diameters and then a list of the centers of the spheres. It is
    recommended that the list of spheres begin with a positive sphere placed at the origin, and the
    other sphere centers are relative to it. Positive spheres are defined by a positive value in
    their diameter. Negative spheres are defined by a negative value in their diameter.
*/
struct SphinxParams : ShapeParams
    {
    /// Circumsphere Diameter of all spheres defined in intersection
    ShortReal circumsphereDiameter;

    /// Number of spheres
    unsigned int N;

    /// Sphere Diameters
    ShortReal diameter[MAX_SPHINX_SPHERE_CENTERS];

    /// Sphere Centers (in local frame)
    vec3<ShortReal> center[MAX_SPHINX_SPHERE_CENTERS];

    /// True when move statistics should not be counted
    unsigned int ignore;

#ifdef ENABLE_HIP
    void set_memory_hint() const
        {
        // default implementation does nothing
        }
#endif

#ifndef __HIPCC__
    /// Empty constructor
    SphinxParams() : circumsphereDiameter(0.0), N(0), ignore(0)
        {
        for (size_t i = 0; i < MAX_SPHINX_SPHERE_CENTERS; i++)
            {
            diameter[i] = 0;
            }
        }

    /// Construct from a python dictionary
    SphinxParams(pybind11::dict v, bool managed = false)
        {
        pybind11::list centers = v["centers"];
        pybind11::list diameters = v["diameters"];
        ignore = v["ignore_statistics"].cast<unsigned int>();

        N = (unsigned int)pybind11::len(diameters);
        unsigned int N_centers = (unsigned int)pybind11::len(centers);

        if (N_centers > MAX_SPHINX_SPHERE_CENTERS)
            throw std::runtime_error("Too many spheres");

        if (N != N_centers)
            {
            throw std::runtime_error(
                std::string("Number of centers not equal to number of diameters")
                + pybind11::str(centers).cast<std::string>() + " "
                + pybind11::str(diameters).cast<std::string>());
            }

        ShortReal radius = ShortReal(0.0);
        for (unsigned int i = 0; i < N_centers; i++)
            {
            pybind11::list center_i = centers[i];

            vec3<ShortReal> center_vec;
            center_vec.x = center_i[0].cast<ShortReal>();
            center_vec.y = center_i[1].cast<ShortReal>();
            center_vec.z = center_i[2].cast<ShortReal>();

            center[i] = center_vec;

            ShortReal d = diameters[i].cast<ShortReal>();
            diameter[i] = d;

            ShortReal n = sqrt(dot(center_vec, center_vec));
            radius = max(radius, (n + d / ShortReal(2.0)));
            }

        // set the diameter
        circumsphereDiameter = ShortReal(2.0) * radius;
        }

    /// Convert parameters to a python dictionary
    pybind11::dict asDict()
        {
        pybind11::list centers;
        pybind11::dict v;
        pybind11::list diameters;
        for (unsigned int i = 0; i < N; i++)
            {
            vec3<ShortReal> center_i = center[i];
            ShortReal x = center_i.x;
            ShortReal y = center_i.y;
            ShortReal z = center_i.z;
            pybind11::list xyz;
            xyz.append(x);
            xyz.append(y);
            xyz.append(z);
            centers.append(pybind11::tuple(xyz));
            diameters.append(diameter[i]);
            }
        v["diameters"] = diameters;
        v["centers"] = centers;
        v["ignore_statistics"] = ignore;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

    }; // end namespace detail

namespace detail
    {
DEVICE inline ShortReal
initVolume(bool disjoint,
           ShortReal r[MAX_SPHINX_SPHERE_CENTERS],
           int n,
           ShortReal d[MAX_SPHINX_SPHERE_CENTERS * (MAX_SPHINX_SPHERE_CENTERS - 1) / 2]);
    }

/** Sphinx shape

    Implement the HPMC shape interface for sphinx particles.
*/
struct ShapeSphinx
    {
    /// Define the parameter type
    typedef detail::SphinxParams param_type;

    /// Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    /// Construct a shape at a given orientation
    DEVICE inline ShapeSphinx(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), convex(true), spheres(_params)
        {
        volume = 0.0;
        radius = spheres.circumsphereDiameter / ShortReal(2.0);
        n = spheres.N;
        for (unsigned int i = 0; i < n; i++)
            {
            r[i] = spheres.diameter[i] / ShortReal(2.0);
            R[i] = r[i] * r[i];
            s[i] = (r[i] < 0) ? -1 : 1;
            u[i] = spheres.center[i];
            }
        for (unsigned int i = 0; i < n; i++)
            {
            for (unsigned int j = 0; j < i; j++)
                {
                D[(i - 1) * i / 2 + j] = dot(u[i] - u[j], u[i] - u[j]);
                d[(i - 1) * i / 2 + j] = ShortReal(s[i] * s[j]) * sqrt(D[(i - 1) * i / 2 + j]);
                }
            }
        disjoint = ((n > 0) && (s[0] > 0));
        if (disjoint)
            for (unsigned int i = 1; i < n; i++)
                {
                if (s[i] > 0)
                    disjoint = false;
                if (disjoint)
                    for (unsigned int j = 1; j < i; j++)
                        if (!detail::seq2(1, 1, R[i], R[j], D[(i - 1) * i / 2 + j]))
                            disjoint = false;
                }
        volume = detail::initVolume(disjoint, r, n, d);
        }

    /// Check if the shape may be rotated
    DEVICE static bool hasOrientation()
        {
        return true;
        }

    /// Get the circumsphere diameter of the shape
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the diameter of the parent sphere
        return spheres.diameter[0];
        }

    /// Get the in-sphere radius of the shape
    DEVICE Scalar getInsphereRadius() const
        {
        return Scalar(0.0);
        }

    /// Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return hoomd::detail::AABB(pos, getCircumsphereDiameter() / Scalar(2.0));
        }

    /// Return a tight fitting OBB around the shape
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
        }

    /// Check if this shape should be ignored in the move statistics
    DEVICE bool ignoreStatistics() const
        {
        return spheres.ignore;
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

    /// Orientation of the sphinx
    quat<Scalar> orientation;

    /// Number of spheres
    unsigned int n;

    bool convex;

    bool disjoint;

    /// radius of each sphere
    ShortReal r[detail::MAX_SPHINX_SPHERE_CENTERS];

    /// radius^2
    ShortReal R[detail::MAX_SPHINX_SPHERE_CENTERS];

    /// sign of radius of each sphere
    int s[detail::MAX_SPHINX_SPHERE_CENTERS];

    /// original center of each sphere
    vec3<ShortReal> u[detail::MAX_SPHINX_SPHERE_CENTERS];

    // vec3<ShortReal> v[MAX_SPHINX_SPHERE_CENTERS];

    /// distance^2 between every pair of spheres
    ShortReal D[detail::MAX_SPHINX_SPHERE_CENTERS * (detail::MAX_SPHINX_SPHERE_CENTERS - 1) / 2];

    /// distance with sign bet. every pair of spheres
    ShortReal d[detail::MAX_SPHINX_SPHERE_CENTERS * (detail::MAX_SPHINX_SPHERE_CENTERS - 1) / 2];

    ShortReal radius;

    ShortReal volume;

    /// shape parameters
    const detail::SphinxParams& spheres;
    };

/** Sphinx particle overlap test

    @param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    @param a Shape a
    @param b Shape b
    @param err in/out variable incremented when error conditions occur in the overlap test
    @param sweep_radius Additional sphere radius to sweep the shapes with
    @returns true if the two particles overlap
*/
template<>
DEVICE inline bool test_overlap<ShapeSphinx, ShapeSphinx>(const vec3<Scalar>& r_ab,
                                                          const ShapeSphinx& p,
                                                          const ShapeSphinx& q,
                                                          unsigned int& err)
    {
    vec3<ShortReal> pv[detail::MAX_SPHINX_SPHERE_CENTERS]; /// rotated centers of p
    vec3<ShortReal> qv[detail::MAX_SPHINX_SPHERE_CENTERS]; /// rotated centers of q

    quat<ShortReal> qp(p.orientation);
    quat<ShortReal> qq(q.orientation);

    // update the positions of the spheres according to the rotations of the center for p
    for (unsigned int i = 0; i < p.n; i++)
        {
        pv[i] = rotate(qp, p.u[i]);
        }

    // update the positions of the spheres according to the rotations of the center for p
    for (unsigned int i = 0; i < q.n; i++)
        {
        qv[i] = rotate(qq, q.u[i]);
        }

    vec3<ShortReal> x(0.0, 0.0, 0.0);
    vec3<ShortReal> y(r_ab);

    if (p.disjoint && q.disjoint)
        {
        if ((p.n == 1) && (q.n == 1))
            {
            vec3<ShortReal> a = x + pv[0], b = y + qv[0];
            if (detail::sep2(false,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             q.R[0],
                             detail::norm2(a - b)))
                return false;
            }
        if ((p.n > 1) && (q.n == 1))
            {
            vec3<ShortReal> a = x + pv[0], c = y + qv[0];
            for (unsigned int i = 1; i < p.n; i++)
                {
                int k = (i - 1) * i / 2;
                vec3<ShortReal> b = x + pv[i];
                if (detail::sep3(false,
                                 ShortReal(p.s[0]),
                                 ShortReal(p.s[i]),
                                 ShortReal(q.s[0]),
                                 p.R[0],
                                 p.R[i],
                                 q.R[0],
                                 p.D[k],
                                 detail::norm2(a - c),
                                 detail::norm2(b - c)))
                    return false;
                }
            }
        if ((p.n == 1) && (q.n > 1))
            {
            vec3<ShortReal> a = x + pv[0], b = y + qv[0];
            for (unsigned int j = 1; j < q.n; j++)
                {
                int l = (j - 1) * j / 2;
                vec3<ShortReal> c = y + qv[j];
                if (detail::sep3(false,
                                 ShortReal(p.s[0]),
                                 ShortReal(q.s[0]),
                                 ShortReal(q.s[j]),
                                 p.R[0],
                                 q.R[0],
                                 q.R[j],
                                 detail::norm2(a - b),
                                 detail::norm2(a - c),
                                 q.D[l]))
                    return false;
                }
            }
        if ((p.n > 1) && (q.n > 1))
            {
            vec3<ShortReal> a = x + pv[0], c = y + qv[0];
            for (unsigned int i = 1; i < p.n; i++)
                {
                int k = (i - 1) * i / 2;
                for (unsigned int j = 1; j < q.n; j++)
                    {
                    int l = (j - 1) * j / 2;
                    vec3<ShortReal> b = x + pv[i], d = y + qv[j];
                    if (detail::sep4(false,
                                     ShortReal(p.s[0]),
                                     ShortReal(p.s[i]),
                                     ShortReal(q.s[0]),
                                     ShortReal(q.s[j]),
                                     p.R[0],
                                     p.R[i],
                                     q.R[0],
                                     q.R[j],
                                     p.D[k],
                                     detail::norm2(a - c),
                                     detail::norm2(a - d),
                                     detail::norm2(b - c),
                                     detail::norm2(b - d),
                                     q.D[l]))
                        return false;
                    }
                }
            }
        return true;
        }

    if ((p.n == 1) && (q.n == 1))
        {
        vec3<ShortReal> a = x + pv[0], b = y + qv[0];
        return !detail::sep2(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             q.R[0],
                             detail::norm2(a - b));
        }

    if ((p.n == 2) && (q.n == 1))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = y + qv[0];
        return !detail::sep3(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             p.R[1],
                             q.R[0],
                             p.D[0],
                             detail::norm2(a - c),
                             detail::norm2(b - c));
        }
    if ((p.n == 1) && (q.n == 2))
        {
        vec3<ShortReal> a = x + pv[0], b = y + qv[0], c = y + qv[1];
        return !detail::sep3(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             p.R[0],
                             q.R[0],
                             q.R[1],
                             detail::norm2(a - b),
                             detail::norm2(a - c),
                             q.D[0]);
        }

    if ((p.n == 3) && (q.n == 1))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = y + qv[0];
        return !detail::sep4(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             q.R[0],
                             p.D[0],
                             p.D[1],
                             detail::norm2(a - d),
                             p.D[2],
                             detail::norm2(b - d),
                             detail::norm2(c - d));
        }
    if ((p.n == 2) && (q.n == 2))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = y + qv[0], d = y + qv[1];
        return !detail::sep4(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             p.R[0],
                             p.R[1],
                             q.R[0],
                             q.R[1],
                             p.D[0],
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(b - c),
                             detail::norm2(b - d),
                             q.D[0]);
        }
    if ((p.n == 1) && (q.n == 3))
        {
        vec3<ShortReal> a = x + pv[0], b = y + qv[0], c = y + qv[1], d = y + qv[2];
        return !detail::sep4(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             p.R[0],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             detail::norm2(a - b),
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             q.D[0],
                             q.D[1],
                             q.D[2]);
        }

    if ((p.n == 4) && (q.n == 1))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = y + qv[0];
        return !detail::sep5(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             q.R[0],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             detail::norm2(a - e),
                             p.D[2],
                             p.D[4],
                             detail::norm2(b - e),
                             p.D[5],
                             detail::norm2(c - e),
                             detail::norm2(d - e));
        }
    if ((p.n == 3) && (q.n == 2))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = y + qv[0], e = y + qv[1];
        return !detail::sep5(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             q.R[0],
                             q.R[1],
                             p.D[0],
                             p.D[1],
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             p.D[2],
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(c - d),
                             detail::norm2(c - e),
                             q.D[0]);
        }
    if ((p.n == 2) && (q.n == 3))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = y + qv[0], d = y + qv[1], e = y + qv[2];
        return !detail::sep5(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             p.R[0],
                             p.R[1],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             p.D[0],
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(b - c),
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             q.D[0],
                             q.D[1],
                             q.D[2]);
        }
    if ((p.n == 1) && (q.n == 4))
        {
        vec3<ShortReal> a = x + pv[0], b = y + qv[0], c = y + qv[1], d = y + qv[2], e = y + qv[3];
        return !detail::sep5(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             p.R[0],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             detail::norm2(a - b),
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[2],
                             q.D[4],
                             q.D[5]);
        }

    if ((p.n == 5) && (q.n == 1))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = x + pv[4],
                        f = y + qv[0];
        return !detail::sep6(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(p.s[4]),
                             ShortReal(q.s[0]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             p.R[4],
                             q.R[0],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             p.D[6],
                             detail::norm2(a - f),
                             p.D[2],
                             p.D[4],
                             p.D[7],
                             detail::norm2(b - f),
                             p.D[5],
                             p.D[8],
                             detail::norm2(c - f),
                             p.D[9],
                             detail::norm2(d - f),
                             detail::norm2(e - f));
        }
    if ((p.n == 4) && (q.n == 2))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = y + qv[0],
                        f = y + qv[1];
        return !detail::sep6(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             q.R[0],
                             q.R[1],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             p.D[2],
                             p.D[4],
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             p.D[5],
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             detail::norm2(d - e),
                             detail::norm2(d - f),
                             q.D[0]);
        }
    if ((p.n == 3) && (q.n == 3))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = y + qv[0], e = y + qv[1],
                        f = y + qv[2];
        return !detail::sep6(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             p.D[0],
                             p.D[1],
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             p.D[2],
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(c - d),
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             q.D[0],
                             q.D[1],
                             q.D[2]);
        }
    if ((p.n == 2) && (q.n == 4))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = y + qv[0], d = y + qv[1], e = y + qv[2],
                        f = y + qv[3];
        return !detail::sep6(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             p.R[0],
                             p.R[1],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             p.D[0],
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(b - c),
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[2],
                             q.D[4],
                             q.D[5]);
        }
    if ((p.n == 1) && (q.n == 5))
        {
        vec3<ShortReal> a = x + pv[0], b = y + qv[0], c = y + qv[1], d = y + qv[2], e = y + qv[3],
                        f = y + qv[4];
        return !detail::sep6(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             ShortReal(q.s[4]),
                             p.R[0],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             q.R[4],
                             detail::norm2(a - b),
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[6],
                             q.D[2],
                             q.D[4],
                             q.D[7],
                             q.D[5],
                             q.D[8],
                             q.D[9]);
        }

    if ((p.n == 5) && (q.n == 2))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = x + pv[4],
                        f = y + qv[0], g = y + qv[1];
        return !detail::sep7(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(p.s[4]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             p.R[4],
                             q.R[0],
                             q.R[1],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             p.D[6],
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             p.D[2],
                             p.D[4],
                             p.D[7],
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             p.D[5],
                             p.D[8],
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             p.D[9],
                             detail::norm2(d - f),
                             detail::norm2(d - g),
                             detail::norm2(e - f),
                             detail::norm2(e - g),
                             q.D[0]);
        }
    if ((p.n == 4) && (q.n == 3))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = y + qv[0],
                        f = y + qv[1], g = y + qv[2];
        return !detail::sep7(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             p.D[2],
                             p.D[4],
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             p.D[5],
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             detail::norm2(d - e),
                             detail::norm2(d - f),
                             detail::norm2(d - g),
                             q.D[0],
                             q.D[1],
                             q.D[2]);
        }
    if ((p.n == 3) && (q.n == 4))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = y + qv[0], e = y + qv[1],
                        f = y + qv[2], g = y + qv[3];
        return !detail::sep7(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             p.D[0],
                             p.D[1],
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             p.D[2],
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             detail::norm2(c - d),
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[2],
                             q.D[4],
                             q.D[5]);
        }
    if ((p.n == 2) && (q.n == 5))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = y + qv[0], d = y + qv[1], e = y + qv[2],
                        f = y + qv[3], g = y + qv[4];
        return !detail::sep7(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             ShortReal(q.s[4]),
                             p.R[0],
                             p.R[1],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             q.R[4],
                             p.D[0],
                             detail::norm2(a - c),
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             detail::norm2(b - c),
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[6],
                             q.D[2],
                             q.D[4],
                             q.D[7],
                             q.D[5],
                             q.D[8],
                             q.D[9]);
        }

    if ((p.n == 5) && (q.n == 3))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = x + pv[4],
                        f = y + qv[0], g = y + qv[1], h = y + qv[2];
        return !detail::sep8(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(p.s[4]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             p.R[4],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             p.D[6],
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             detail::norm2(a - h),
                             p.D[2],
                             p.D[4],
                             p.D[7],
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             detail::norm2(b - h),
                             p.D[5],
                             p.D[8],
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             detail::norm2(c - h),
                             p.D[9],
                             detail::norm2(d - f),
                             detail::norm2(d - g),
                             detail::norm2(d - h),
                             detail::norm2(e - f),
                             detail::norm2(e - g),
                             detail::norm2(e - h),
                             q.D[0],
                             q.D[1],
                             q.D[2]);
        }
    if ((p.n == 4) && (q.n == 4))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = x + pv[3], e = y + qv[0],
                        f = y + qv[1], g = y + qv[2], h = y + qv[3];
        return !detail::sep8(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(p.s[3]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             p.R[3],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             p.D[0],
                             p.D[1],
                             p.D[3],
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             detail::norm2(a - h),
                             p.D[2],
                             p.D[4],
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             detail::norm2(b - h),
                             p.D[5],
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             detail::norm2(c - h),
                             detail::norm2(d - e),
                             detail::norm2(d - f),
                             detail::norm2(d - g),
                             detail::norm2(d - h),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[2],
                             q.D[4],
                             q.D[5]);
        }
    if ((p.n == 3) && (q.n == 5))
        {
        vec3<ShortReal> a = x + pv[0], b = x + pv[1], c = x + pv[2], d = y + qv[0], e = y + qv[1],
                        f = y + qv[2], g = y + qv[3], h = y + qv[4];
        return !detail::sep8(p.convex && q.convex,
                             ShortReal(p.s[0]),
                             ShortReal(p.s[1]),
                             ShortReal(p.s[2]),
                             ShortReal(q.s[0]),
                             ShortReal(q.s[1]),
                             ShortReal(q.s[2]),
                             ShortReal(q.s[3]),
                             ShortReal(q.s[4]),
                             p.R[0],
                             p.R[1],
                             p.R[2],
                             q.R[0],
                             q.R[1],
                             q.R[2],
                             q.R[3],
                             q.R[4],
                             p.D[0],
                             p.D[1],
                             detail::norm2(a - d),
                             detail::norm2(a - e),
                             detail::norm2(a - f),
                             detail::norm2(a - g),
                             detail::norm2(a - h),
                             p.D[2],
                             detail::norm2(b - d),
                             detail::norm2(b - e),
                             detail::norm2(b - f),
                             detail::norm2(b - g),
                             detail::norm2(b - h),
                             detail::norm2(c - d),
                             detail::norm2(c - e),
                             detail::norm2(c - f),
                             detail::norm2(c - g),
                             detail::norm2(c - h),
                             q.D[0],
                             q.D[1],
                             q.D[3],
                             q.D[6],
                             q.D[2],
                             q.D[4],
                             q.D[7],
                             q.D[5],
                             q.D[8],
                             q.D[9]);
        }
    /*if((p.n == 5) && (q.n == 4))
        {
        vec3<ShortReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g
    = y+qv[1],h = y+qv[2],i = y+qv[3]; return !detail::sep9(p.convex && q.convex,
                      p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],q.s[2],q.s[3],
                      p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],q.R[2],q.R[3],
                      p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),
                      p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),
                      p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),
                      p.D[9],detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),
                      detail::norm2(e-f),detail::norm2(e-g),detail::norm2(e-h),detail::norm2(e-i),
                      q.D[0],q.D[1],q.D[3],
                      q.D[2],q.D[4],
                      q.D[5]);
        }
    if((p.n == 4) && (q.n == 5))
        {
        vec3<ShortReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = y+qv[0],f = y+qv[1],g
    = y+qv[2],h = y+qv[3],i = y+qv[4]; return !detail::sep9(p.convex && q.convex,
                      p.s[0],p.s[1],p.s[2],p.s[3],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                      p.R[0],p.R[1],p.R[2],p.R[3],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                      p.D[0],p.D[1],p.D[3],detail::norm2(a-e),detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),
                      p.D[2],p.D[4],detail::norm2(b-e),detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),
                      p.D[5],detail::norm2(c-e),detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),
                      detail::norm2(d-e),detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),
                      q.D[0],q.D[1],q.D[3],q.D[6],
                      q.D[2],q.D[4],q.D[7],
                      q.D[5],q.D[8],
                      q.D[9]);
        }

    if((p.n == 5) && (q.n == 5))
        {
        vec3<ShortReal> a = x+pv[0],b = x+pv[1],c = x+pv[2],d = x+pv[3],e = x+pv[4],f = y+qv[0],g
    = y+qv[1],h = y+qv[2],i = y+qv[3],j = y+qv[4]; return !detail::sep10(p.convex && q.convex,
                      p.s[0],p.s[1],p.s[2],p.s[3],p.s[4],q.s[0],q.s[1],q.s[2],q.s[3],q.s[4],
                      p.R[0],p.R[1],p.R[2],p.R[3],p.R[4],q.R[0],q.R[1],q.R[2],q.R[3],q.R[4],
                      p.D[0],p.D[1],p.D[3],p.D[6],detail::norm2(a-f),detail::norm2(a-g),detail::norm2(a-h),detail::norm2(a-i),detail::norm2(a-j),
                      p.D[2],p.D[4],p.D[7],detail::norm2(b-f),detail::norm2(b-g),detail::norm2(b-h),detail::norm2(b-i),detail::norm2(b-j),
                      p.D[5],p.D[8],detail::norm2(c-f),detail::norm2(c-g),detail::norm2(c-h),detail::norm2(c-i),detail::norm2(c-j),
                      p.D[9],detail::norm2(d-f),detail::norm2(d-g),detail::norm2(d-h),detail::norm2(d-i),detail::norm2(d-j),
                      detail::norm2(e-f),detail::norm2(e-g),detail::norm2(e-h),detail::norm2(e-i),detail::norm2(e-j),
                      q.D[0],q.D[1],q.D[3],q.D[6],
                      q.D[2],q.D[4],q.D[7],
                      q.D[5],q.D[8],
                      q.D[9]);
        }*/

    return true;
    }

namespace detail
    {
DEVICE inline ShortReal
initVolume(bool disjoint,
           ShortReal r[MAX_SPHINX_SPHERE_CENTERS],
           int n,
           ShortReal d[MAX_SPHINX_SPHERE_CENTERS * (MAX_SPHINX_SPHERE_CENTERS - 1) / 2])
    {
    if (disjoint)
        {
        ShortReal vol = uol1(r[0]);
        for (int i = 1; i < n; i++)
            vol += uol2(r[0], r[i], d[(i - 1) * i / 2]) - uol1(r[0]);
        return vol;
        }

    if (n == 1)
        return uol1(r[0]);

    if (n == 2)
        return uol2(r[0], r[1], d[0]);

    if (n == 3)
        return uol3(r[0], r[1], r[2], d[0], d[1], d[2]);

    if (n == 4)
        return uol4(r[0], r[1], r[2], r[3], d[0], d[1], d[3], d[2], d[4], d[5]);

    /*if(n == 5)
        return uol5(r[0],r[1],r[2],r[3],r[4],
                    d[0],d[1],d[3],d[6],
                    d[2],d[4],d[7],
                    d[5],d[8],
                    d[9]);
    */
    return 0;
    }

    } // namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
