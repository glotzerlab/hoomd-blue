// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _SHAPE_UTILS_H
#define _SHAPE_UTILS_H

#include <fstream>
#include <iostream>
#include <queue>
#include <set>

#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeEllipsoid.h"
#include "ShapePolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeSphere.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSphinx.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"

namespace hoomd
    {

namespace hpmc
    {

namespace detail
    {

class MassPropertiesBase
    {
    public:
    MassPropertiesBase() : m_volume(0.0), m_surface_area(0.0), m_center_of_mass(0.0, 0.0, 0.0)
        {
        m_inertia.resize(6, 0.0);
        }

    virtual ~MassPropertiesBase() {};

    Scalar getVolume()
        {
        return m_volume;
        }

    std::vector<Scalar>& getInertiaTensor()
        {
        return m_inertia;
        }

    const vec3<Scalar>& getCenterOfMass()
        {
        return m_center_of_mass;
        }

    Scalar getDetInertiaTensor()
        {
        vec3<Scalar> a(m_inertia[0], m_inertia[3], m_inertia[5]),
            b(m_inertia[3], m_inertia[1], m_inertia[4]),
            c(m_inertia[5], m_inertia[4], m_inertia[2]);
        // determinant can be negative depending on order of vertices
        return std::abs(dot(a, cross(b, c)));
        }

    virtual void compute() { }

    protected:
    Scalar m_volume;
    Scalar m_surface_area;
    vec3<Scalar> m_center_of_mass;
    std::vector<Scalar> m_inertia; // xx, yy, zz, xy, yz, xz
    }; // end class MassPropertiesBase

template<class Shape> class MassProperties : public MassPropertiesBase
    {
    public:
    MassProperties() : MassPropertiesBase() { }

    MassProperties(const typename Shape::param_type& shape) : MassPropertiesBase()
        {
        this->compute();
        }
    };

template<> class MassProperties<ShapeConvexPolyhedron> : public MassPropertiesBase
    {
    public:
    MassProperties() : MassPropertiesBase() {};

    MassProperties(const typename ShapeConvexPolyhedron::param_type& param) : MassPropertiesBase()
        {
        auto p = getQuickHullVertsAndFaces(param);
        points = p.first;
        faces = p.second;
        this->compute();
        }

    MassProperties(const std::vector<vec3<Scalar>>& p,
                   const std::vector<std::vector<unsigned int>>& f)
        : points(p), faces(f)
        {
        this->compute();
        }

    std::pair<std::vector<vec3<Scalar>>, std::vector<std::vector<unsigned int>>>
    getQuickHullVertsAndFaces(const typename ShapeConvexPolyhedron::param_type& param)
        {
        std::vector<quickhull::Vector3<ShortReal>> verts;
        for (unsigned int i = 0; i < param.N; i++)
            {
            quickhull::Vector3<ShortReal> vert(param.x[i], param.y[i], param.z[i]);
            verts.push_back(vert);
            }
        quickhull::QuickHull<ShortReal> qh;
        auto hull = qh.getConvexHull(&verts[0].x, verts.size(), true, true, 0.0000001f);
        auto verts2 = hull.getVertexBuffer();
        std::vector<vec3<Scalar>> v;
        for (unsigned int i = 0; i < verts2.size(); i++)
            {
            vec3<Scalar> vert(verts2[i].x, verts2[i].y, verts2[i].z);
            v.push_back(vert);
            }
        auto face_inds = hull.getIndexBuffer();
        std::vector<std::vector<unsigned int>> faces;
        for (unsigned int i = 0; i < face_inds.size(); i += 3)
            {
            std::vector<unsigned int> face {static_cast<unsigned int>(face_inds[i]),
                                            static_cast<unsigned int>(face_inds[i + 1]),
                                            static_cast<unsigned int>(face_inds[i + 2])};
            faces.push_back(face);
            }
        return std::make_pair(v, faces);
        }

    void updateParam(const typename ShapeConvexPolyhedron::param_type& param)
        {
        if (param.N != points.size())
            {
            auto p = getQuickHullVertsAndFaces(param);
            points = p.first;
            faces = p.second;
            }
        else
            {
            // assumes that the faces are still good.
            for (unsigned int i = 0; i < param.N; i++)
                {
                points[i] = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
                }
            }
        this->compute();
        }

    /*
        algorithm taken from
        https://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    */
    void compute()
        {
        const Scalar mult[10] = {1.0 / 6.0,
                                 1.0 / 24.0,
                                 1.0 / 24.0,
                                 1.0 / 24.0,
                                 1.0 / 60.0,
                                 1.0 / 60.0,
                                 1.0 / 60.0,
                                 1.0 / 120.0,
                                 1.0 / 120.0,
                                 1.0 / 120.0};
        Scalar intg[10]
            = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // order: 1, x, y, z, xˆ2, yˆ2, zˆ2, xy, yz, zx
        Scalar surface_area = 0.0;
        for (unsigned int t = 0; t < faces.size(); t++)
            {
            // get vertices of triangle
            vec3<Scalar> v0, v1, v2; // vertices
            vec3<Scalar> a1, a2, d;
            v0 = points[faces[t][0]];
            v1 = points[faces[t][1]];
            v2 = points[faces[t][2]];
            // get edges and cross product of edges
            a1 = v1 - v0;
            a2 = v2 - v0;
            d = cross(a1, a2); // = |a1||a2|sin(theta)

            vec3<Scalar> temp0, temp1, temp2, f1, f2, f3, g0, g1, g2;
            temp0 = v0 + v1;
            f1 = temp0 + v2;
            temp1 = v0 * v0;
            temp2 = temp1 + v1 * temp0;
            f2 = temp2 + v2 * f1;
            f3 = v0 * temp1 + v1 * temp2 + v2 * f2;
            g0 = f2 + v0 * (f1 + v0);
            g1 = f2 + v1 * (f1 + v1);
            g2 = f2 + v2 * (f1 + v2);

            intg[0] += d.x * f1.x;
            intg[1] += d.x * f2.x;
            intg[2] += d.y * f2.y;
            intg[3] += d.z * f2.z;
            intg[4] += d.x * f3.x;
            intg[5] += d.y * f3.y;
            intg[6] += d.z * f3.z;
            intg[7] += d.x * (v0.y * g0.x + v1.y * g1.x + v2.y * g2.x);
            intg[8] += d.y * (v0.z * g0.y + v1.z * g1.y + v2.z * g2.y);
            intg[9] += d.z * (v0.x * g0.z + v1.x * g1.z + v2.x * g2.z);

            // add to surface area
            surface_area += 0.5 * sqrt(dot(d, d));

            } // end loop over faces
        for (unsigned int i = 0; i < 10; i++)
            {
            intg[i] *= -1 * mult[i];
            }

        m_volume = intg[0];
        m_surface_area = surface_area;
        m_center_of_mass.x = intg[1];
        m_center_of_mass.y = intg[2];
        m_center_of_mass.z = intg[3];
        m_center_of_mass /= m_volume;

        Scalar cx2 = m_center_of_mass.x * m_center_of_mass.x,
               cy2 = m_center_of_mass.y * m_center_of_mass.y,
               cz2 = m_center_of_mass.z * m_center_of_mass.z;
        Scalar cxy = m_center_of_mass.x * m_center_of_mass.y,
               cyz = m_center_of_mass.y * m_center_of_mass.z,
               cxz = m_center_of_mass.x * m_center_of_mass.z;
        m_inertia[0] = intg[5] + intg[6] - m_volume * (cy2 + cz2);
        m_inertia[1] = intg[4] + intg[6] - m_volume * (cz2 + cx2);
        m_inertia[2] = intg[4] + intg[5] - m_volume * (cx2 + cy2);
        m_inertia[3] = -(intg[7] - m_volume * cxy);
        m_inertia[4] = -(intg[8] - m_volume * cyz);
        m_inertia[5] = -(intg[9] - m_volume * cxz);
        } // end MassProperties<ShapeConvexPolyhedron>::compute()

    protected:
    std::vector<vec3<Scalar>> points;
    std::vector<std::vector<unsigned int>> faces;
    }; // end class MassProperties < ShapeConvexPolyhedron >

template<> class MassProperties<ShapeEllipsoid> : public MassPropertiesBase
    {
    public:
    MassProperties() : MassPropertiesBase() {};

    MassProperties(const typename ShapeEllipsoid::param_type& param)
        : MassPropertiesBase(), m_param(param)
        {
        this->compute();
        }

    void updateParam(const typename ShapeEllipsoid::param_type& param)
        {
        m_param = param;
        this->compute();
        }

    void compute()
        {
        m_volume = Scalar(4.0) / Scalar(3.0) * M_PI * m_param.x * m_param.y * m_param.z;
        Scalar a2 = m_param.x * m_param.x;
        Scalar b2 = m_param.y * m_param.y;
        Scalar c2 = m_param.z * m_param.z;
        m_inertia[0] = m_volume * (b2 + c2) / Scalar(5.0);
        m_inertia[1] = m_volume * (a2 + c2) / Scalar(5.0);
        m_inertia[2] = m_volume * (a2 + b2) / Scalar(5.0);
        m_inertia[3] = Scalar(0);
        m_inertia[4] = Scalar(0);
        m_inertia[5] = Scalar(0);
        }

    private:
    typename ShapeEllipsoid::param_type m_param;
    };

template<> class MassProperties<ShapeSpheropolyhedron> : public MassPropertiesBase
    {
    public:
    MassProperties(const typename ShapeSpheropolyhedron::param_type& param) : MassPropertiesBase()
        {
        // error out if the shape is a true spheropolyhedron
        if (param.sweep_radius != 0 && param.N > 1)
            {
            throw std::runtime_error(
                "This class currently only supports the computation of mass properties \
                for spheres or convex polyhedra, but not for true spheropolyhedra");
            }
        m_param = param;
        this->compute();
        }

    void compute()
        {
        // Assuming it's a sphere, return that moment of inertia tensor.
        // Otherwise, fall back to convex polyhedra specialization.
        if (m_param.sweep_radius > 0)
            {
            Scalar sweep_radius = m_param.sweep_radius;
            this->m_volume
                = Scalar(4) / Scalar(3) * M_PI * sweep_radius * sweep_radius * sweep_radius;
            Scalar moment_inertia = m_volume * 2 * sweep_radius * sweep_radius / 5;
            this->m_inertia[0] = moment_inertia;
            this->m_inertia[1] = moment_inertia;
            this->m_inertia[2] = moment_inertia;
            this->m_inertia[3] = 0;
            this->m_inertia[4] = 0;
            this->m_inertia[5] = 0;
            }
        else
            {
            MassProperties<ShapeConvexPolyhedron> obj(m_param);
            this->m_inertia = obj.getInertiaTensor();
            this->m_volume = obj.getVolume();
            }
        }

    private:
    typename ShapeSpheropolyhedron::param_type m_param;
    };
    } // end namespace detail

void export_MassPropertiesBase(pybind11::module& m);

template<class Shape> void export_MassProperties(pybind11::module& m, std::string name);

    } // end namespace hpmc
    } // namespace hoomd
#endif // end inclusion guard
