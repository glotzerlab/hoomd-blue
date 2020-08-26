#ifndef _SHAPE_UTILS_H
#define _SHAPE_UTILS_H

#include <queue>
#include <set>
#include <iostream>
#include <fstream>

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeSpheropolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeSphinx.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"

namespace hpmc
{


class ShapeUtilError : public std::runtime_error
    {
    public:
        ShapeUtilError(const std::string& msg) : runtime_error(msg) {}
    };

template<class ShapeParam>
inline void printParam(const ShapeParam& param){ std::cout << "not implemented" << std::endl;}

template< >
inline void printParam< ShapeConvexPolyhedron::param_type >(const ShapeConvexPolyhedron::param_type& param)
    {
    for(size_t i = 0; i < param.N; i++)
        {
        std::cout << "vert " << i << ": [" << param.x[i] << ", "
                  << param.y[i] << ", " << param.z[i] << "]" << std::endl;
        }

    }


namespace detail
{
// TODO: template Scalar type.

template<class Shape>
class MassPropertiesBase
    {
    public:
        MassPropertiesBase() : m_volume(0.0),
                               m_surface_area(0.0),
                               m_center_of_mass(0.0, 0.0, 0.0),
                               m_isoperimetric_quotient(0.0)
            {
            for(unsigned int i = 0; i < 6; i++) m_inertia[i] = 0.0;
            }

        virtual ~MassPropertiesBase() {};

        Scalar getVolume() { return m_volume; }

        Scalar getSurfaceArea() { return m_surface_area; }

        Scalar getIsoperimetricQuotient() { return m_isoperimetric_quotient; }

        const vec3<Scalar>& getCenterOfMass() { return m_center_of_mass; }

        Scalar getCenterOfMassElement(unsigned int i)
            {
            if(i == 0 )
                return m_center_of_mass.x;
            else if(i == 1 )
                return m_center_of_mass.y;
            else if (i == 2)
                return m_center_of_mass.z;
            else
                throw std::runtime_error("index out of range");
            }

        Scalar getInertiaTensor(unsigned int i)
            {
            if(i >= 6 )
                throw std::runtime_error("index out of range");
            return m_inertia[i];
            }

        Scalar getDeterminant()
            {
            vec3<Scalar> a(m_inertia[0], m_inertia[3], m_inertia[5]),
                         b(m_inertia[3], m_inertia[1], m_inertia[4]),
                         c(m_inertia[5], m_inertia[4], m_inertia[2]);
            return dot(a, cross(b,c));
            }

        virtual void updateParam(const typename Shape::param_type& param, bool force = true) { }

    protected:
        virtual void compute()
            {
            throw std::runtime_error("MassProperties::compute() is not implemented for this shape.");
            }
        Scalar m_volume;
        Scalar m_surface_area;
        vec3<Scalar> m_center_of_mass;
        Scalar m_isoperimetric_quotient;
        Scalar m_inertia[6]; // xx, yy, zz, xy, yz, xz
    };   // end class MassPropertiesBase

template<class Shape>
class MassProperties : public MassPropertiesBase<Shape>
    {
    public:
        MassProperties() : MassPropertiesBase<Shape>() {}

        MassProperties(const typename Shape::param_type& shape) :  MassPropertiesBase<Shape>()
            {
            this->compute();
            }
    };

inline void normalizeInplace(vec3<Scalar>& v) { v /= sqrt(dot(v,v)); }

inline vec3<Scalar> normalize(const vec3<Scalar>& v) { return v / sqrt(dot(v,v)); }
// face is assumed to be an array of indices of triangular face of a convex body.
// points may contain points inside or outside the body defined by faces.
// faces may include faces that contain vertices that are inside the body.
inline vec3<Scalar> getOutwardNormal(const std::vector< vec3<Scalar> >& points,
                                     const vec3<Scalar>& inside_point,
                                     const std::vector< std::vector<unsigned int> >& faces,
                                     const unsigned int& faceid,
                                     Scalar thresh = 0.0001)
    {
    const std::vector<unsigned int>& face = faces[faceid];
    vec3<Scalar> a = points[face[0]], b = points[face[1]], c = points[face[2]];
    vec3<Scalar> di = (inside_point - a), n;
    n = cross((b - a),(c - a));
    normalizeInplace(n);
    Scalar d = dot(n, di);
    if(fabs(d) < thresh)
        throw(ShapeUtilError("ShapeUtils.h::getOutwardNormal -- inner point is in the plane"));
    return (d > 0) ? -n : n;
    }

inline void sortFace(const std::vector< vec3<Scalar> >& points,
                     const vec3<Scalar>& inside_point,
                     std::vector< std::vector<unsigned int> >& faces,
                     const unsigned int& faceid,
                     Scalar thresh = 0.0001)
    {
    assert(faces[faceid].size() == 3);
    vec3<Scalar> a = points[faces[faceid][0]],
                 b = points[faces[faceid][1]],
                 c = points[faces[faceid][2]],
                 n = cross((b - a),(c - a)),
                 nout = getOutwardNormal(points, inside_point, faces, faceid, thresh);
    if (dot(nout, n) < 0)
        std::reverse(faces[faceid].begin(), faces[faceid].end());
    }

inline void sortFaces(const std::vector< vec3<Scalar> >& points,
                      std::vector< std::vector<unsigned int> >& faces,
                      Scalar thresh = 0.0001)
    {
    vec3<Scalar> inside_point(0.0,0.0,0.0);
    for(size_t i = 0; i < points.size(); i++)
        {
        inside_point += points[i];
        }
    inside_point /= Scalar(points.size());

    for(unsigned int f = 0; f < faces.size(); f++ )
        sortFace(points, inside_point, faces, f, thresh);
    }


template< >
class MassProperties< ShapeConvexPolyhedron > : public MassPropertiesBase< ShapeConvexPolyhedron >
    {
    public:
        MassProperties() : MassPropertiesBase() {};

        MassProperties(const typename ShapeConvexPolyhedron::param_type& param,
                       bool do_compute=true) : MassPropertiesBase()
            {
            std::pair<std::vector<vec3<Scalar>>, std::vector<std::vector<unsigned int>>> p;
            p = getQuickHullVertsAndFaces(param);
            points = p.first;
            faces = p.second;
            if (do_compute)
                {
                compute();
                }
            }

        MassProperties(const std::vector< vec3<Scalar> >& p,
                       const std::vector<std::vector<unsigned int> >& f,
                       bool do_compute = true) : points(p), faces(f)
            {
            if (do_compute)
                {
                compute();
                }
            }

        std::pair<std::vector<vec3<Scalar>>, std::vector<std::vector<unsigned int>>>
        getQuickHullVertsAndFaces(const typename ShapeConvexPolyhedron::param_type& param)
            {
            std::vector<quickhull::Vector3<OverlapReal>> verts;
            for(size_t i = 0; i < param.N; i++)
                {
                quickhull::Vector3<OverlapReal> vert(param.x[i],
                                                     param.y[i],
                                                     param.z[i]);
                verts.push_back(vert);
                }
            quickhull::QuickHull<OverlapReal> qh;
            auto hull = qh.getConvexHull(&verts[0].x, verts.size(), true, true, 0.0000001);
            auto verts2 = hull.getVertexBuffer();
            std::vector<vec3<Scalar>> v;
            for(size_t i = 0; i < verts2.size(); i++)
                {
                vec3<Scalar> vert(verts2[i].x,
                                  verts2[i].y,
                                  verts2[i].z);
                v.push_back(vert);
                }
            auto face_inds = hull.getIndexBuffer();
            std::vector<std::vector<unsigned int>> faces;
            for (size_t i = 0; i < face_inds.size(); i += 3)
                {
                std::vector<unsigned int> face{static_cast<unsigned int>(face_inds[i]),
                                               static_cast<unsigned int>(face_inds[i+1]),
                                               static_cast<unsigned int>(face_inds[i+2])};
                faces.push_back(face);
                }
            return std::make_pair(v, faces);
            }

        pybind11::list getFaceVertices(unsigned int i, unsigned int j)
            {
            vec3<double> pt = points[faces[i][j]];
            pybind11::list l;
            l.append(pt.x);
            l.append(pt.y);
            l.append(pt.z);
            return l;
            }

        unsigned int getNumFaces() { return faces.size(); }

        void updateParam(const typename ShapeConvexPolyhedron::param_type& param, bool force=true)
            {
            if(force || param.N != points.size())
                {
                std::pair<std::vector<vec3<Scalar>>, std::vector<std::vector<unsigned int>>> p;
                p = getQuickHullVertsAndFaces(param);
                points = p.first;
                faces = p.second;
                sortFaces(points, faces);
                }
            else
                {
                // assumes that the faces are still good.
                for(unsigned int i = 0; i < param.N; i++)
                    {
                    points[i] = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
                    }
                }
            compute();
            }

    protected:
        using MassPropertiesBase< ShapeConvexPolyhedron >::m_volume;
        using MassPropertiesBase< ShapeConvexPolyhedron >::m_surface_area;
        using MassPropertiesBase< ShapeConvexPolyhedron >::m_center_of_mass;
        using MassPropertiesBase< ShapeConvexPolyhedron >::m_isoperimetric_quotient;
        using MassPropertiesBase< ShapeConvexPolyhedron >::m_inertia;
        std::vector< vec3<Scalar> > points;
        std::vector<std::vector<unsigned int> > faces;
    /*
        algorithm taken from
        http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    */
        virtual void compute()
            {
            const Scalar mult[10] = {1.0/6.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/60.0,
                                     1.0/60.0, 1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0};
            Scalar intg[10] = {0,0,0,0,0,0,0,0,0,0};  // order: 1, x, y, z, xˆ2, yˆ2, zˆ2, xy, yz, zx
            Scalar surface_area = 0.0;
            for (unsigned int t=0; t<faces.size(); t++)
                {
                //get vertices of triangle
                vec3<Scalar> v0, v1, v2;  // vertices
                vec3<Scalar> a1, a2, d;
                v0 = points[faces[t][0]];
                v1 = points[faces[t][1]];
                v2 = points[faces[t][2]];
                // get edges and cross product of edges
                a1 = v1 - v0;
                a2 = v2 - v0;
                d = cross(a1, a2);  // = |a1||a2|sin(theta)

                vec3<Scalar> temp0, temp1, temp2, f1, f2, f3, g0, g1, g2;
                temp0 = v0 + v1;
                f1 = temp0 + v2;
                temp1 = v0*v0;
                temp2 = temp1 + v1*temp0;
                f2 = temp2 + v2*f1;
                f3 = v0*temp1 + v1*temp2 + v2*f2;
                g0 = f2 + v0*(f1 + v0);
                g1 = f2 + v1*(f1 + v1);
                g2 = f2 + v2*(f1 + v2);

                intg[0] += d.x*f1.x;
                intg[1] += d.x*f2.x; intg[2] += d.y*f2.y; intg[3] += d.z*f2.z;
                intg[4] += d.x*f3.x; intg[5] += d.y*f3.y; intg[6] += d.z*f3.z;
                intg[7] += d.x*(v0.y*g0.x + v1.y*g1.x + v2.y*g2.x);
                intg[8] += d.y*(v0.z*g0.y + v1.z*g1.y + v2.z*g2.y);
                intg[9] += d.z*(v0.x*g0.z + v1.x*g1.z + v2.x*g2.z);

                // add to surface area
                surface_area += 0.5 * sqrt(dot(d,d));

                }  // end loop over faces
            for(unsigned int i = 0; i < 10; i++)
                {
                intg[i] *= -1*mult[i];
                }

            m_volume = intg[0];
            m_surface_area = surface_area;
            m_isoperimetric_quotient = 36 * M_PI * m_volume * m_volume /
                                      (m_surface_area * m_surface_area * m_surface_area);

            m_center_of_mass.x = intg[1];
            m_center_of_mass.y = intg[2];
            m_center_of_mass.z = intg[3];
            m_center_of_mass /= m_volume;

            Scalar cx2 = m_center_of_mass.x*m_center_of_mass.x,
                   cy2 = m_center_of_mass.y*m_center_of_mass.y,
                   cz2 = m_center_of_mass.z*m_center_of_mass.z;
            Scalar cxy = m_center_of_mass.x*m_center_of_mass.y,
                   cyz = m_center_of_mass.y*m_center_of_mass.z,
                   cxz = m_center_of_mass.x*m_center_of_mass.z;
            m_inertia[0] = intg[5] + intg[6] - m_volume*(cy2 + cz2);
            m_inertia[1] = intg[4] + intg[6] - m_volume*(cz2 + cx2);
            m_inertia[2] = intg[4] + intg[5] - m_volume*(cx2 + cy2);
            m_inertia[3] = -(intg[7] - m_volume*cxy);
            m_inertia[4] = -(intg[8] - m_volume*cyz);
            m_inertia[5] = -(intg[9] - m_volume*cxz);
            }  // end MassProperties<ShapeConvexPolyhedron>::compute()
    };  // end class MassProperties < ShapeConvexPolyhedron >

template<>
class MassProperties<ShapeEllipsoid> : public MassPropertiesBase<ShapeEllipsoid>
    {

    public:
        MassProperties() : MassPropertiesBase() {};

        MassProperties(const typename ShapeEllipsoid::param_type& param, bool do_compute = true)
                       : MassPropertiesBase(), m_param(param)
            {
            if (do_compute) compute();
            }

        virtual void updateParam(const typename ShapeEllipsoid::param_type& param,
                                 bool force = true)
            {
            m_param = param;
            compute();
            }

        Scalar getDeterminant()
            {
            return m_inertia[0]*m_inertia[1]*m_inertia[2];
            }

    static constexpr Scalar m_vol_factor = Scalar(4.0)/Scalar(3.0)*M_PI;

    protected:
        using MassPropertiesBase<ShapeEllipsoid>::m_volume;
        using MassPropertiesBase<ShapeEllipsoid>::m_inertia;

        virtual void compute()
            {
            m_volume = m_vol_factor*m_param.x*m_param.y*m_param.z;
            Scalar a2 = m_param.x*m_param.x;
            Scalar b2 = m_param.y*m_param.y;
            Scalar c2 = m_param.z*m_param.z;
            m_inertia[0] = (b2+c2)/Scalar(5.0);
            m_inertia[1] = (a2+c2)/Scalar(5.0);
            m_inertia[2] = (a2+b2)/Scalar(5.0);
            m_inertia[3] = Scalar(0);
            m_inertia[4] = Scalar(0);
            m_inertia[5] = Scalar(0);
            }

    private:
        typename ShapeEllipsoid::param_type m_param;
    };

//TODO: Enable true spheropolyhedron calculation
template < >
class MassProperties< ShapeSpheropolyhedron > : public MassProperties < ShapeConvexPolyhedron >
    {
    using MassProperties< ShapeConvexPolyhedron >::m_inertia;
    public:
        MassProperties(const typename ShapeSpheropolyhedron::param_type& param)
            // Prevent computation on construction of the parent
            // so we can first check if it's possible.
            : MassProperties< ShapeConvexPolyhedron >(param, false)
            {
            if (param.sweep_radius != 0 and param.N > 0)
                {
                throw std::runtime_error("The ShapeSpheropolyhedra class currently only supports the calculation of mass properties for spheres or convex polyhedra");
                }

            // Explicit typecast required here
            m_sweep_radius = param.sweep_radius;
            compute();
            }

    protected:
        virtual void compute()
            {
            if (m_sweep_radius > 0)
                {
                // Ensure it's not a true spheropolyhedron
                if (this->points.size() > 0)
                    {
                    throw std::runtime_error("The ShapeSpheropolyhedra class currently only supports the calculation of mass properties for spheres or convex polyhedra");
                    }
                // Assuming it's a sphere, return that moment of inertia tensor
                float moment_inertia = 2*m_sweep_radius*m_sweep_radius/5;
                m_inertia[0] = moment_inertia;
                m_inertia[1] = moment_inertia;
                m_inertia[2] = moment_inertia;
                m_inertia[3] = 0;
                m_inertia[4] = 0;
                m_inertia[5] = 0;
                }
            else
                {
                return MassProperties< ShapeConvexPolyhedron >::compute();
                }
            }

    private:
        OverlapReal m_sweep_radius;

    };
} // end namespace detail


template<class Shape>
void export_MassPropertiesBase(pybind11::module& m, std::string name);

template<class Shape>
void export_MassProperties(pybind11::module& m, std::string name);

} // end namespace hpmc
#endif // end inclusion guard
