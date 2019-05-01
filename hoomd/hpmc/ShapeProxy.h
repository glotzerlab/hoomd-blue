// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __SHAPE_PROXY_H__
#define __SHAPE_PROXY_H__

#include "IntegratorHPMCMono.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedEllipsoid.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>

#include "hoomd/extern/quickhull/QuickHull.hpp"
#endif

namespace hpmc{
namespace detail{

// make these global constants in one of the shape headers.
#define IGNORE_STATS 0x0001

template<class param_type>
inline pybind11::list poly2d_verts_to_python(param_type& param)
    {
    pybind11::list verts;
    for(size_t i = 0; i < param.N; i++)
        {
        pybind11::list v;
        v.append(pybind11::cast<Scalar>(param.x[i]));
        v.append(pybind11::cast<Scalar>(param.y[i]));
        verts.append(v);
        }
    return verts;
    }

template<class param_type>
inline pybind11::list poly3d_verts_to_python(param_type& param)
    {
    pybind11::list verts;
    for(size_t i = 0; i < param.N; i++)
        {
        pybind11::list v;
        v.append(pybind11::cast<Scalar>(param.x[i]));
        v.append(pybind11::cast<Scalar>(param.y[i]));
        v.append(pybind11::cast<Scalar>(param.z[i]));
        verts.append(v);
        }
    return verts;
    }

template<class ScalarType>
pybind11::list vec3_to_python(const vec3<ScalarType>& vec)
    {
    pybind11::list v;
    v.append(pybind11::cast<Scalar>(vec.x));
    v.append(pybind11::cast<Scalar>(vec.y));
    v.append(pybind11::cast<Scalar>(vec.z));
    return v;
    }

template<class ScalarType>
pybind11::list quat_to_python(const quat<ScalarType>& qu)
    {
    pybind11::list v;
    v.append(pybind11::cast<Scalar>(qu.s));
    v.append(pybind11::cast<Scalar>(qu.v.x));
    v.append(pybind11::cast<Scalar>(qu.v.y));
    v.append(pybind11::cast<Scalar>(qu.v.z));
    return v;
    }

//! helper function to make ignore flag, not exported to pytho
unsigned int make_ignore_flag(bool stats, bool ovrlps)
    {
    unsigned int ret=0;
    if(stats)
      {
      ret=2;
      }

    if(ovrlps)
      {
      ret++;
      }

    return ret;
    }

//! Helper function to build ell_params from python
ell_params make_ell_params(OverlapReal x, OverlapReal y, OverlapReal z, bool ignore_stats)
    {
    ell_params result;
    result.ignore = ignore_stats;
    result.x=x;
    result.y=y;
    result.z=z;
    return result;
    }
//
//! Helper function to build sph_params from python
sph_params make_sph_params(OverlapReal radius, bool ignore_stats, bool orientable)
    {
    sph_params result;
    result.ignore = ignore_stats;
    result.radius=radius;
    result.isOriented = orientable;
    return result;
    }

//! Helper function to build poly2d_verts from python
poly2d_verts make_poly2d_verts(pybind11::list verts, OverlapReal sweep_radius, bool ignore_stats)
    {
    if (len(verts) > MAX_POLY2D_VERTS)
        throw std::runtime_error("Too many polygon vertices");

    poly2d_verts result;
    result.N = len(verts);
    result.ignore = ignore_stats;
    result.sweep_radius = sweep_radius;

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(verts); i++)
        {
        pybind11::list verts_i = pybind11::cast<pybind11::list>(verts[i]);
        vec2<OverlapReal> vert = vec2<OverlapReal>(pybind11::cast<OverlapReal>(verts_i[0]), pybind11::cast<OverlapReal>(verts_i[1]));
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        radius_sq = max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = len(verts); i < MAX_POLY2D_VERTS; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        }

    // set the diameter
    result.diameter = 2*(sqrt(radius_sq)+sweep_radius);

    return result;
    }

//! Helper function to build poly3d_data from python
inline ShapePolyhedron::param_type make_poly3d_data(pybind11::list verts,pybind11::list face_verts,
                             pybind11::list face_offs,
                             pybind11::list overlap,
                             OverlapReal R, bool ignore_stats,
                             unsigned int leaf_capacity,
                             pybind11::list origin,
                             unsigned int hull_only,
                             std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    ShapePolyhedron::param_type result;

    // compute convex hull of vertices
    typedef quickhull::Vector3<OverlapReal> vec;

    std::vector<vec> qh_pts;
    for (unsigned int i = 0; i < len(verts); i++)
        {
        pybind11::list v = pybind11::cast<pybind11::list>(verts[i]);
        vec vert;
        vert.x = pybind11::cast<OverlapReal>(v[0]);
        vert.y = pybind11::cast<OverlapReal>(v[1]);
        vert.z = pybind11::cast<OverlapReal>(v[2]);
        qh_pts.push_back(vert);
        }

    quickhull::QuickHull<OverlapReal> qh;
    auto hull = qh.getConvexHull(qh_pts, true, false);
    auto vertexBuffer = hull.getVertexBuffer();

    result = detail::poly3d_data(len(verts), len(face_offs)-1, len(face_verts), vertexBuffer.size(), exec_conf->isCUDAEnabled());
    result.ignore = ignore_stats;
    result.sweep_radius = result.convex_hull_verts.sweep_radius = R;
    result.n_verts = len(verts);
    result.n_faces = len(face_offs)-1;
    result.origin = vec3<OverlapReal>(pybind11::cast<OverlapReal>(origin[0]), pybind11::cast<OverlapReal>(origin[1]), pybind11::cast<OverlapReal>(origin[2]));
    result.hull_only = hull_only;

    if (len(overlap) != result.n_faces)
        {
        throw std::runtime_error("Number of member overlap flags must be equal to number faces");
        }

    unsigned int k = 0;
    for (auto it = vertexBuffer.begin(); it != vertexBuffer.end(); ++it)
        {
        result.convex_hull_verts.x[k] = it->x;
        result.convex_hull_verts.y[k] = it->y;
        result.convex_hull_verts.z[k] = it->z;
        k++;
        }

    for (unsigned int i = 0; i < len(face_offs); i++)
        {
        unsigned int offs = pybind11::cast<unsigned int>(face_offs[i]);
        result.face_offs[i] = offs;
        }

    for (unsigned int i = 0; i < result.n_faces; i++)
        {
        result.face_overlap[i] = pybind11::cast<unsigned int>(overlap[i]);
        }

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(verts); i++)
        {
        pybind11::list v = pybind11::cast<pybind11::list>(verts[i]);
        vec3<OverlapReal> vert;
        vert.x = pybind11::cast<OverlapReal>(v[0]);
        vert.y = pybind11::cast<OverlapReal>(v[1]);
        vert.z = pybind11::cast<OverlapReal>(v[2]);
        result.verts[i] = vert;
        radius_sq = max(radius_sq, dot(vert, vert));
        }

    for (unsigned int i = 0; i < len(face_verts); i++)
        {
        unsigned int j = pybind11::cast<unsigned int>(face_verts[i]);
        if (j >= result.n_verts)
            {
            std::ostringstream oss;
            oss << "Invalid vertex index " << j << " specified" << std::endl;
            throw std::runtime_error(oss.str());
            }
        result.face_verts[i] = j;
        }

    hpmc::detail::OBB *obbs = new hpmc::detail::OBB[len(face_offs)];
    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

    // construct bounding box tree
    for (unsigned int i = 0; i < len(face_offs)-1; ++i)
        {
        std::vector<vec3<OverlapReal> > face_vec;

        unsigned int n_vert = 0;
        for (unsigned int j = result.face_offs[i]; j < result.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v = result.verts[result.face_verts[j]];
            face_vec.push_back(v);
            n_vert++;
            }

        std::vector<OverlapReal> vertex_radii(n_vert, result.sweep_radius);
        obbs[i] = hpmc::detail::compute_obb(face_vec, vertex_radii, false);
        obbs[i].mask = result.face_overlap[i];
        internal_coordinates.push_back(face_vec);
        }

    OBBTree tree;
    tree.buildTree(obbs, internal_coordinates, result.sweep_radius, len(face_offs)-1, leaf_capacity);
    result.tree = GPUTree(tree, exec_conf->isCUDAEnabled());
    delete [] obbs;

    // set the diameter
    result.convex_hull_verts.diameter = 2*(sqrt(radius_sq)+result.sweep_radius);

    return result;
    }

//! Helper function to build poly3d_verts from python
poly3d_verts make_poly3d_verts(pybind11::list verts, OverlapReal sweep_radius, bool ignore_stats,
                                        std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    poly3d_verts result(len(verts), exec_conf->isCUDAEnabled());
    result.N = len(verts);
    result.sweep_radius = sweep_radius;
    result.ignore = ignore_stats;

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(verts); i++)
        {
        pybind11::list verts_i = pybind11::cast<pybind11::list>(verts[i]);
        vec3<OverlapReal> vert = vec3<OverlapReal>(pybind11::cast<OverlapReal>(verts_i[0]), pybind11::cast<OverlapReal>(verts_i[1]), pybind11::cast<OverlapReal>(verts_i[2]));
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        result.z[i] = vert.z;
        radius_sq = max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = len(verts); i < result.N; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        result.z[i] = 0;
        }

    // set the diameter
    result.diameter = 2*(sqrt(radius_sq) + sweep_radius);

    return result;
    }

//! Helper function to build faceted_ellipsoid_params from python
faceted_ellipsoid_params make_faceted_ellipsoid(pybind11::list normals, pybind11::list offsets,
    pybind11::list vertices, Scalar a, Scalar b, Scalar c, pybind11::tuple origin, bool ignore_stats,
    std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    if (len(offsets) != len(normals))
        throw std::runtime_error("Number of normals unequal number of offsets");

    faceted_ellipsoid_params result(len(normals), exec_conf->isCUDAEnabled());
    result.ignore = ignore_stats;

    // extract the normals from the python list
    for (unsigned int i = 0; i < len(normals); i++)
        {
        pybind11::list normals_i = pybind11::cast<pybind11::list>(normals[i]);
        result.n[i] = vec3<OverlapReal>(pybind11::cast<OverlapReal>(normals_i[0]), pybind11::cast<OverlapReal>(normals_i[1]), pybind11::cast<OverlapReal>(normals_i[2]));
        result.offset[i] = pybind11::cast<OverlapReal>(offsets[i]);
        }

    // extract the vertices from the python list
    result.verts=make_poly3d_verts(vertices, 0.0, false, exec_conf);

    // scale vertices onto unit sphere
    for (unsigned int i = 0; i < result.verts.N; ++i)
        {
        result.verts.x[i] /= a;
        result.verts.y[i] /= b;
        result.verts.z[i] /= c;
        }

    // set the half-axes
    result.a = a; result.b = b; result.c = c;

    // set the origin
    result.origin = vec3<OverlapReal>(pybind11::cast<OverlapReal>(origin[0]), pybind11::cast<OverlapReal>(origin[1]), pybind11::cast<OverlapReal>(origin[2]));

    // add the edge-sphere vertices
    ShapeFacetedEllipsoid::initializeVertices(result, exec_conf->isCUDAEnabled());

    return result;
    }

//! Helper function to build sphinx3d_verts from python
sphinx3d_params make_sphinx3d_params(pybind11::list diameters, pybind11::list centers, bool ignore_stats)
    {
    if (len(centers) > MAX_SPHERE_CENTERS)
        throw std::runtime_error("Too many spheres");

    sphinx3d_params result;
    result.N = len(diameters);
    if (len(diameters) != len(centers))
        {
        throw std::runtime_error("Number of centers not equal to number of diameters");
        }

    result.ignore = ignore_stats;

    // extract the centers from the python list and compute the radius on the way
    OverlapReal radius = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(centers); i++)
        {
        OverlapReal d = pybind11::cast<OverlapReal>(diameters[i]);
        pybind11::list centers_i = pybind11::cast<pybind11::list>(centers[i]);
        result.center[i] = vec3<OverlapReal>(pybind11::cast<OverlapReal>(centers_i[0]), pybind11::cast<OverlapReal>(centers_i[1]), pybind11::cast<OverlapReal>(centers_i[2]));
        result.diameter[i] = d;
        OverlapReal n = sqrt(dot(result.center[i],result.center[i]));
        radius = max(radius, (n+d/OverlapReal(2.0)));
        }

    // set the diameter
    result.circumsphereDiameter = 2.0*radius;

    return result;
    }

//! Templated helper function to build shape union params from constituent shape params
template<class Shape>
typename ShapeUnion<Shape>::param_type make_union_params(pybind11::list _members,
                                        pybind11::list positions,
                                        pybind11::list orientations,
                                        pybind11::list overlap,
                                        bool ignore_stats,
                                        unsigned int leaf_capacity,
                                        std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    typename ShapeUnion<Shape>::param_type result(len(_members), exec_conf->isCUDAEnabled());

    if (len(positions) != result.N)
        {
        throw std::runtime_error("Number of member positions not equal to number of members");
        }
    if (len(orientations) != result.N)
        {
        throw std::runtime_error("Number of member orientations not equal to number of members");
        }

    if (len(overlap) != result.N)
        {
        throw std::runtime_error("Number of member overlap flags not equal to number of members");
        }

    result.ignore = ignore_stats;

    hpmc::detail::OBB *obbs = new hpmc::detail::OBB[result.N];

    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

    // extract member parameters, positions, and orientations and compute the radius along the way
    OverlapReal diameter = OverlapReal(0.0);
    for (unsigned int i = 0; i < result.N; i++)
        {
        typename Shape::param_type param = pybind11::cast<typename Shape::param_type>(_members[i]);
        pybind11::list positions_i = pybind11::cast<pybind11::list>(positions[i]);
        vec3<OverlapReal> pos = vec3<OverlapReal>(pybind11::cast<OverlapReal>(positions_i[0]), pybind11::cast<OverlapReal>(positions_i[1]), pybind11::cast<OverlapReal>(positions_i[2]));
        pybind11::list orientations_i = pybind11::cast<pybind11::list>(orientations[i]);
        OverlapReal s = pybind11::cast<OverlapReal>(orientations_i[0]);
        OverlapReal x = pybind11::cast<OverlapReal>(orientations_i[1]);
        OverlapReal y = pybind11::cast<OverlapReal>(orientations_i[2]);
        OverlapReal z = pybind11::cast<OverlapReal>(orientations_i[3]);
        quat<OverlapReal> orientation(s, vec3<OverlapReal>(x,y,z));
        result.mparams[i] = param;
        result.mpos[i] = pos;
        result.morientation[i] = orientation;
        result.moverlap[i] = pybind11::cast<unsigned int>(overlap[i]);

        Shape dummy(quat<Scalar>(), param);
        Scalar d = sqrt(dot(pos,pos));
        diameter = max(diameter, OverlapReal(2*d + dummy.getCircumsphereDiameter()));

        obbs[i] = detail::OBB(pos,dummy.getCircumsphereDiameter()/2.0);
        obbs[i].mask = result.moverlap[i];
        }

    // set the diameter
    result.diameter = diameter;

    // build tree and store GPU accessible version in parameter structure
    typedef typename ShapeUnion<Shape>::param_type::gpu_tree_type gpu_tree_type;
    OBBTree tree;
    tree.buildTree(obbs, result.N, leaf_capacity, true);
    delete [] obbs;
    result.tree = gpu_tree_type(tree,exec_conf->isCUDAEnabled());

    return result;
    }

template< typename Shape >
struct get_param_data_type { typedef typename Shape::param_type type; };

template< >
struct get_param_data_type< ShapePolyhedron > { typedef poly3d_data type; }; // hard to dig into the structure but this could be made more general by modifying the ShapePolyhedron::param_type

template< typename Shape >
struct access
    {
    template< class ParamType >
    typename get_param_data_type<Shape>::type& operator()(ParamType& param) { return param; }
    template< class ParamType >
    const typename get_param_data_type<Shape>::type& operator()(const ParamType& param) const  { return param; }
    };

template< >
struct access < ShapePolyhedron >
    {
    template< class ParamType >
    typename get_param_data_type<ShapePolyhedron>::type& operator()(ParamType& param) { return param; }
    template< class ParamType >
    const typename get_param_data_type<ShapePolyhedron>::type& operator()(const ParamType& param) const  { return param; }
    };

template < typename Shape , typename AccessType = access<Shape> >
class shape_param_proxy // base class to avoid adding the ignore flag logic to every other class and holds the integrator pointer + typeid
{
protected:
    typedef typename Shape::param_type param_type;
public:
    shape_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : m_mc(mc), m_typeid(typendx), m_access(acc) {}
    //!Ignore flag for acceptance statistics
    bool getIgnoreStatistics() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return (m_access(params[m_typeid]).ignore & IGNORE_STATS);
        }

    void setIgnoreStatistics(bool stat)
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        if(stat)    m_access(params[m_typeid]).ignore |= IGNORE_STATS;
        else        m_access(params[m_typeid]).ignore &= ~IGNORE_STATS;
        }

protected:
    std::shared_ptr< IntegratorHPMCMono<Shape> > m_mc;
    unsigned int m_typeid;
    AccessType m_access;
};

template<class Shape, class AccessType = access<Shape> >
class sphere_param_proxy : public shape_param_proxy<Shape, AccessType>
{
using shape_param_proxy<Shape, AccessType>::m_mc;
using shape_param_proxy<Shape, AccessType>::m_typeid;
using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename Shape::param_type  param_type;
public:
    typedef sph_params access_type;
    sphere_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    OverlapReal getDiameter()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return OverlapReal(2.0)*m_access(params[m_typeid]).radius;
        }

    bool getOrientable()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).isOriented;
        }
};

template<class Shape, class AccessType = access<Shape> >
class ell_param_proxy : public shape_param_proxy<Shape, AccessType>
{
using shape_param_proxy<Shape, AccessType>::m_mc;
using shape_param_proxy<Shape, AccessType>::m_typeid;
using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename Shape::param_type  param_type;
public:
    typedef ell_params  access_type;
    ell_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx, acc) {}

    OverlapReal getX()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).x;
        }

    OverlapReal getY()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).y;
        }

    OverlapReal getZ()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).z;
        }
};

template< typename Shape, class AccessType = access<Shape> >
class poly2d_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<Shape, AccessType>::m_mc;
    using shape_param_proxy<Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy<Shape, AccessType>::param_type param_type;
public:
    typedef poly2d_verts access_type;
    poly2d_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    pybind11::list getVerts() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return poly2d_verts_to_python(m_access(params[m_typeid]));
        }

    OverlapReal getSweepRadius() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).sweep_radius;
        }
};

template< typename Shape, class AccessType = access<Shape> >
class poly3d_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<Shape, AccessType>::m_mc;
    using shape_param_proxy<Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy<Shape, AccessType>::param_type param_type;
public:
    typedef poly3d_verts access_type;
    poly3d_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc) {}

    pybind11::list getVerts() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return poly3d_verts_to_python(m_access(params[m_typeid]));
        }

    OverlapReal getSweepRadius() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).sweep_radius;
        }

};

template< typename Shape, class AccessType = access<Shape> >
class polyhedron_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<ShapePolyhedron>::m_mc;
    using shape_param_proxy<ShapePolyhedron>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef shape_param_proxy<ShapePolyhedron>::param_type param_type;
public:
    typedef poly3d_data access_type;
    polyhedron_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    pybind11::list getVerts()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);

        pybind11::list verts;
        for(size_t i = 0; i < param.n_verts; i++)
            {
            pybind11::list v;
            v.append(pybind11::cast<Scalar>(param.verts[i].x));
            v.append(pybind11::cast<Scalar>(param.verts[i].y));
            v.append(pybind11::cast<Scalar>(param.verts[i].z));
            verts.append(v);
            }
        return verts;
        }

    pybind11::list getFaces()
        {
        pybind11::list faces;
        // populate faces.
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        for(size_t i = 0; i < param.n_faces; i++)
            {
            pybind11::list face;
            for(unsigned int f = param.face_offs[i]; f < param.face_offs[i+1]; f++)
                {
                face.append(pybind11::int_(param.face_verts[f]));
                }
            faces.append(face);
            }
        return faces;
        }

    pybind11::list getOverlap()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list overlap;
        for(size_t i = 0; i < param.n_faces; i++)
            overlap.append(pybind11::cast<unsigned int>(param.face_overlap[i]));
        return overlap;
        }

    pybind11::tuple getOrigin()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        vec3<OverlapReal> origin(m_access(params[m_typeid]).origin);
        return pybind11::make_tuple(origin.x, origin.y, origin.z);
        }

    OverlapReal getSweepRadius() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).sweep_radius;
        }

    unsigned int getCapacity() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).tree.getLeafNodeCapacity();
        }

    bool getHullOnly() const
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return m_access(params[m_typeid]).hull_only;
        }

};

template< typename Shape, class AccessType = access<Shape> >
class faceted_ellipsoid_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<Shape, AccessType>::m_mc;
    using shape_param_proxy<Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy<Shape, AccessType>::param_type param_type;
public:
    typedef ShapeFacetedEllipsoid::param_type access_type;
    faceted_ellipsoid_param_proxy(std::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy<Shape, AccessType>(mc,typendx,acc)
        {}

    pybind11::list getVerts()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        return poly3d_verts_to_python(m_access(params[m_typeid]).verts);
        }

    pybind11::list getNormals()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list normals;
        for(size_t i = 0; i < param.N; i++ ) normals.append(vec3_to_python(param.n[i]));
        return normals;
        }

    pybind11::list getOrigin()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return vec3_to_python(param.origin);
        }

    OverlapReal getDiameter()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return OverlapReal(2)*detail::max(param.a, detail::max(param.b, param.c));
        }

    OverlapReal getHalfAxisA()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return param.a; // first half-axis
        }

    OverlapReal getHalfAxisB()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return param.b; // second half-axis
        }

    OverlapReal getHalfAxisC()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return param.c; // third half-axis
        }

    pybind11::list getOffsets()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list offsets;
        for(size_t i = 0; i < param.N; i++) offsets.append(pybind11::cast<Scalar>(param.offset[i]));
        return offsets;
        }
};

template< typename Shape, class AccessType = access<Shape> >
class sphinx3d_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<Shape, AccessType>::m_mc;
    using shape_param_proxy<Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy<Shape, AccessType>::param_type param_type;
public:
    typedef ShapeSphinx::param_type access_type;
    sphinx3d_param_proxy(std::shared_ptr< IntegratorHPMCMono<ShapeSphinx> > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy<Shape, AccessType>(mc,typendx,acc)
        {}

    pybind11::list getCenters()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list centers;
        for(size_t i = 0; i < param.N; i++) centers.append(vec3_to_python(param.center[i]));
        return centers;
        }

    pybind11::list getDiameters()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list diams;
        for(size_t i = 0; i < param.N; i++) diams.append(pybind11::cast<Scalar>(param.diameter[i]));
        return diams;
        }

    OverlapReal getCircumsphereDiameter()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return param.circumsphereDiameter;
        }
};

template< class ShapeUnionType>
struct get_member_type{};

template<class BaseShape>
struct get_member_type< ShapeUnion<BaseShape> >
    {
    typedef typename BaseShape::param_type type;
    typedef BaseShape base_shape;
    };

template< typename Shape, typename ShapeUnionType, typename AccessType>
struct get_member_proxy{};

template<typename Shape, typename AccessType >
struct get_member_proxy<Shape, ShapeUnion<ShapeSphere>, AccessType >{ typedef sphere_param_proxy<Shape, AccessType> proxy_type; };

template<typename Shape, typename AccessType >
struct get_member_proxy<Shape, ShapeUnion<ShapeSpheropolyhedron>, AccessType >{ typedef poly3d_param_proxy<Shape, AccessType> proxy_type; };

template<typename Shape, typename AccessType >
struct get_member_proxy<Shape, ShapeUnion<ShapeFacetedEllipsoid>, AccessType >{ typedef faceted_ellipsoid_param_proxy<Shape, AccessType> proxy_type; };

template< class ShapeUnionType >
struct access_shape_union_members
{
    typedef typename get_member_type<ShapeUnionType>::type member_type;
    unsigned int offset;
    access_shape_union_members(unsigned int ndx = 0) { offset = ndx; }
    member_type& operator()(typename ShapeUnionType::param_type& param ) {return param.mparams[offset]; }
    const member_type& operator()(const typename ShapeUnionType::param_type& param ) const {return param.mparams[offset]; }
};

template< typename Shape, typename ShapeUnionType, typename AccessType = access<Shape> >
class shape_union_param_proxy : public shape_param_proxy< Shape, AccessType>
{
    using shape_param_proxy< Shape, AccessType>::m_mc;
    using shape_param_proxy< Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy< Shape, AccessType>::param_type param_type;
    typedef typename get_member_type<ShapeUnionType>::type member_type;
    typedef typename get_member_type<ShapeUnionType>::base_shape base_shape;
    typedef typename get_member_proxy<Shape, ShapeUnionType, access_shape_union_members<ShapeUnionType> >::proxy_type proxy_type;
public:
    typedef typename ShapeUnionType::param_type access_type;
    shape_union_param_proxy(std::shared_ptr< IntegratorHPMCMono< Shape > > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy< Shape, AccessType>(mc,typendx,acc)
        {}
    pybind11::list getPositions()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list pos;
        for(size_t i = 0; i < param.N; i++) pos.append(vec3_to_python(param.mpos[i]));
        return pos;
        }

    pybind11::list getOrientations()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list orient;
        for(size_t i = 0; i < param.N; i++)
            orient.append(quat_to_python(param.morientation[i]));
        return orient;
        }

    std::vector< std::shared_ptr< proxy_type > > getMembers()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        std::vector< std::shared_ptr< proxy_type > > members;
        for(size_t i = 0; i < param.N; i++)
            {
            access_shape_union_members<ShapeUnionType> acc(i);
            std::shared_ptr< proxy_type > p(new proxy_type(m_mc, m_typeid, acc));
            members.push_back(p);
            }
        return members;
        }

    pybind11::list getOverlap()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        pybind11::list overlap;
        for(size_t i = 0; i < param.N; i++)
            overlap.append(pybind11::cast<unsigned int>(param.moverlap[i]));
        return overlap;
        }


    OverlapReal getDiameter()
        {
        std::vector<param_type, managed_allocator<param_type> > & params = m_mc->getParams();
        access_type& param = m_access(params[m_typeid]);
        return param.diameter;
        }
};

} // end namespace detail

template<class Shape, class AccessType>
void export_shape_param_proxy(pybind11::module& m, const std::string& name)
    {
    // export the base class.
    using detail::shape_param_proxy;
    pybind11::class_<shape_param_proxy<Shape, AccessType>, std::shared_ptr< shape_param_proxy<Shape, AccessType> > >(m, name.c_str())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<Shape> >, unsigned int>())
    .def_property("ignore_statistics", &shape_param_proxy<Shape, AccessType>::getIgnoreStatistics, &shape_param_proxy<Shape, AccessType>::setIgnoreStatistics)
    ;
    }

template<class ShapeType, class AccessType>
void export_sphere_proxy(pybind11::module& m, const std::string& class_name)
    {
    using detail::shape_param_proxy;
    using detail::sphere_param_proxy;
    typedef shape_param_proxy<ShapeType, AccessType>    proxy_base;
    typedef sphere_param_proxy<ShapeType, AccessType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, AccessType>(m, base_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("diameter", &proxy_class::getDiameter)
    .def_property_readonly("orientable", &proxy_class::getOrientable)
    ;
    }

void export_ell_proxy(pybind11::module& m)
    {
    using detail::shape_param_proxy;
    using detail::ell_param_proxy;
    typedef ShapeEllipsoid                  ShapeType;
    typedef shape_param_proxy<ShapeType>    proxy_base;
    typedef ell_param_proxy<ShapeType>      proxy_class;
    std::string class_name="ell_param_proxy";
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(m, base_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("a", &proxy_class::getX)
    .def_property_readonly("b", &proxy_class::getY)
    .def_property_readonly("c", &proxy_class::getZ)
    ;
    }

template<class ShapeType>
void export_poly2d_proxy(pybind11::module& m, std::string class_name, bool sweep_radius_valid)
    {
    using detail::shape_param_proxy;
    using detail::poly2d_param_proxy;
    typedef shape_param_proxy<ShapeType>    proxy_base;
    typedef poly2d_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";
    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(m, base_name);
    if (sweep_radius_valid)
        {
        pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
        .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
        .def_property_readonly("vertices", &proxy_class::getVerts)
        .def_property_readonly("sweep_radius", &proxy_class::getSweepRadius)
        ;
        }
    else
        {
        pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
        .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
        .def_property_readonly("vertices", &proxy_class::getVerts)
        ;
        }
    }

template<class ShapeType, class AccessType>
void export_poly3d_proxy(pybind11::module& m, std::string class_name, bool sweep_radius_valid)
    {
    using detail::shape_param_proxy;
    using detail::poly3d_param_proxy;
    typedef shape_param_proxy<ShapeType, AccessType>    proxy_base;
    typedef poly3d_param_proxy<ShapeType, AccessType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, AccessType >(m, base_name);
    if (sweep_radius_valid)
        {
        pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
        .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
        .def_property_readonly("vertices", &proxy_class::getVerts)
        .def_property_readonly("sweep_radius", &proxy_class::getSweepRadius)
        ;
        }
    else
        {
        pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
        .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
        .def_property_readonly("vertices", &proxy_class::getVerts)
        ;
        }
    }

void export_polyhedron_proxy(pybind11::module& m, std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::polyhedron_param_proxy;
    typedef ShapePolyhedron                     ShapeType;
    typedef shape_param_proxy<ShapeType>        proxy_base;
    typedef polyhedron_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(m, base_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("vertices", &proxy_class::getVerts)
    .def_property_readonly("faces", &proxy_class::getFaces)
    .def_property_readonly("overlap", &proxy_class::getOverlap)
    .def_property_readonly("origin", &proxy_class::getOrigin)
    .def_property_readonly("sweep_radius", &proxy_class::getSweepRadius)
    .def_property_readonly("capacity", &proxy_class::getCapacity)
    .def_property_readonly("hull_only", &proxy_class::getHullOnly)
    ;
    }

template<class ShapeType, class AccessType>
void export_faceted_ellipsoid_proxy(pybind11::module& m, std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::faceted_ellipsoid_param_proxy;
    typedef shape_param_proxy<ShapeType, AccessType>    proxy_base;
    typedef faceted_ellipsoid_param_proxy<ShapeType, AccessType>   proxy_class;

    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, AccessType >(m, base_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("vertices", &proxy_class::getVerts)
    .def_property_readonly("normals", &proxy_class::getNormals)
    .def_property_readonly("origin", &proxy_class::getOrigin)
    .def_property_readonly("diameter", &proxy_class::getDiameter)
    .def_property_readonly("a", &proxy_class::getHalfAxisA)
    .def_property_readonly("b", &proxy_class::getHalfAxisB)
    .def_property_readonly("c", &proxy_class::getHalfAxisC)
    .def_property_readonly("offsets", &proxy_class::getOffsets)
    ;

    }

void export_sphinx_proxy(pybind11::module& m, std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::sphinx3d_param_proxy;
    typedef ShapeSphinx                         ShapeType;
    typedef shape_param_proxy<ShapeType>        proxy_base;
    typedef sphinx3d_param_proxy<ShapeType>     proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(m, base_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("centers", &proxy_class::getCenters)
    .def_property_readonly("diameters", &proxy_class::getDiameters)
    .def_property_readonly("diameter", &proxy_class::getCircumsphereDiameter)
    ;

    }

template<class Shape, class ExportFunction >
void export_shape_union_proxy(pybind11::module& m, std::string class_name, ExportFunction& export_member_proxy)
    {
    using detail::shape_param_proxy;
    using detail::shape_union_param_proxy;
    typedef ShapeUnion<Shape>                     ShapeType;
    typedef shape_param_proxy<ShapeType>                    proxy_base;
    typedef shape_union_param_proxy<ShapeType, ShapeType>   proxy_class;

    std::string base_name=class_name+"_base";
    std::string member_name=class_name+"_member_proxy";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(m, base_name);
    export_member_proxy(m, member_name);
    pybind11::class_<proxy_class, std::shared_ptr< proxy_class > >(m, class_name.c_str(), pybind11::base< proxy_base >())
    .def(pybind11::init<std::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>())
    .def_property_readonly("centers", &proxy_class::getPositions)
    .def_property_readonly("orientations", &proxy_class::getOrientations)
    .def_property_readonly("diameter", &proxy_class::getDiameter)
    .def_property_readonly("members", &proxy_class::getMembers)
    .def_property_readonly("overlap", &proxy_class::getOverlap)
    ;

    }


void export_shape_params(pybind11::module& m)
    {
    export_sphere_proxy<ShapeSphere, detail::access<ShapeSphere> >(m, "sphere_param_proxy");
    export_ell_proxy(m);
    export_poly2d_proxy<ShapeConvexPolygon>(m, "convex_polygon_param_proxy", false);
    export_poly2d_proxy<ShapeSpheropolygon>(m, "convex_spheropolygon_param_proxy", true);
    export_poly2d_proxy<ShapeSimplePolygon>(m, "simple_polygon_param_proxy", false);

    export_poly3d_proxy< ShapeConvexPolyhedron, detail::access<ShapeConvexPolyhedron> >(m, "convex_polyhedron_param_proxy", false);

    export_poly3d_proxy< ShapeSpheropolyhedron, detail::access<ShapeSpheropolyhedron> >(m, "convex_spheropolyhedron_param_proxy", true);

    export_polyhedron_proxy(m, "polyhedron_param_proxy");
    export_faceted_ellipsoid_proxy<ShapeFacetedEllipsoid, detail::access<ShapeFacetedEllipsoid> >(m, "faceted_ellipsoid_param_proxy");
    export_sphinx_proxy(m, "sphinx3d_param_proxy");

    auto export_fnct_sphero = std::bind(export_poly3d_proxy<ShapeUnion<ShapeSpheropolyhedron>, detail::access_shape_union_members< ShapeUnion<ShapeSpheropolyhedron> > >, std::placeholders::_1, std::placeholders::_2, true);
    export_shape_union_proxy<ShapeSpheropolyhedron>(m, "convex_polyhedron_union_param_proxy", export_fnct_sphero);

    auto export_fnct_faceted = std::bind(export_faceted_ellipsoid_proxy<ShapeUnion<ShapeFacetedEllipsoid>, detail::access_shape_union_members< ShapeUnion<ShapeFacetedEllipsoid> > >, std::placeholders::_1, std::placeholders::_2);
    export_shape_union_proxy<ShapeFacetedEllipsoid>(m, "faceted_ellipsoid_union_param_proxy", export_fnct_faceted);

    export_shape_union_proxy<ShapeSphere>(m, "sphere_union_param_proxy", export_sphere_proxy<ShapeUnion<ShapeSphere>, detail::access_shape_union_members< ShapeUnion<ShapeSphere> > > );
    }

} // end namespace hpmc


#endif // end __SHAPE_PROXY_H__
