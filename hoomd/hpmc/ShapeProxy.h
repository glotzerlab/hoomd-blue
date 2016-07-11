// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __SHAPE_PROXY_H__
#define __SHAPE_PROXY_H__

#include <boost/python.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility.hpp>

#include "IntegratorHPMCMono.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"


namespace hpmc{
namespace detail{

// make these global constants in one of the shape headers.
#define IGNORE_OVRLP 0x0001
#define IGNORE_STATS 0x0002

template<class param_type>
inline boost::python::list poly2d_verts_to_python(param_type& param)
    {
    boost::python::list verts;
    for(size_t i = 0; i < param.N; i++)
        {
        boost::python::list v;
        v.append(param.x[i]);
        v.append(param.y[i]);
        verts.append(v);
        }
    return verts;
    }

template<class param_type>
inline boost::python::list poly3d_verts_to_python(param_type& param)
    {
    boost::python::list verts;
    for(size_t i = 0; i < param.N; i++)
        {
        boost::python::list v;
        v.append(param.x[i]);
        v.append(param.y[i]);
        v.append(param.z[i]);
        verts.append(v);
        }
    return verts;
    }

template<class ScalarType>
boost::python::list vec3_to_python(const vec3<ScalarType>& vec)
    {
    boost::python::list v;
    v.append(vec.x);
    v.append(vec.y);
    v.append(vec.z);
    return v;
    }

template<class ScalarType>
boost::python::list quat_to_python(const quat<ScalarType>& qu)
    {
    boost::python::list v;
    v.append(qu.s);
    v.append(qu.v.x);
    v.append(qu.v.y);
    v.append(qu.v.z);
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
sph_params make_sph_params(OverlapReal radius, bool ignore_stats)
    {
    sph_params result;
    result.ignore = ignore_stats;
    result.radius=radius;
    return result;
    }

//! Helper function to build poly2d_verts from python
poly2d_verts make_poly2d_verts(boost::python::list verts, OverlapReal sweep_radius, bool ignore_stats)
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
        vec2<OverlapReal> vert = vec2<OverlapReal>(extract<OverlapReal>(verts[i][0]), extract<OverlapReal>(verts[i][1]));
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
inline ShapePolyhedron::param_type make_poly3d_data(boost::python::list verts,boost::python::list face_verts,
                             boost::python::list face_offs, OverlapReal R, bool ignore_stats)
    {
    if (len(verts) > MAX_POLY3D_VERTS)
        throw std::runtime_error("Too many polyhedron vertices");

    if (len(face_verts) > MAX_POLY3D_FACE_VERTS*MAX_POLY3D_FACES)
        throw std::runtime_error("Too many polyhedron face vertices");

    if (len(face_offs) > MAX_POLY3D_FACES + 1)
        throw std::runtime_error("Too many polyhedron faces");

    // rounding radius

    ShapePolyhedron::param_type result;
    result.data.ignore = ignore_stats;
    result.data.verts.N = len(verts);
    result.data.verts.sweep_radius = R;
    result.data.n_faces = len(face_offs)-1;

    for (unsigned int i = 0; i < len(face_offs); i++)
        {
        unsigned int offs = extract<unsigned int>(face_offs[i]);
        result.data.face_offs[i] = offs;
        }

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(verts); i++)
        {
        boost::python::list v = extract<boost::python::list>(verts[i]);
        vec3<OverlapReal> vert;
        vert.x = extract<OverlapReal>(v[0]);
        vert.y = extract<OverlapReal>(v[1]);
        vert.z = extract<OverlapReal>(v[2]);
        result.data.verts.x[i] = vert.x;
        result.data.verts.y[i] = vert.y;
        result.data.verts.z[i] = vert.z;
        radius_sq = max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = len(verts); i < MAX_POLY3D_VERTS; i++)
        {
        result.data.verts.x[i] = 0;
        result.data.verts.y[i] = 0;
        result.data.verts.z[i] = 0;
        }

    for (unsigned int i = 0; i < len(face_verts); i++)
        {
        unsigned int j = extract<unsigned int>(face_verts[i]);
        if (j >= result.data.verts.N)
            {
            std::ostringstream oss;
            oss << "Invalid vertex index " << j << " specified" << std::endl;
            throw std::runtime_error(oss.str());
            }
        result.data.face_verts[i] = j;
        }

    hpmc::detail::OBB *obbs;
    int retval = posix_memalign((void**)&obbs, 32, sizeof(hpmc::detail::OBB)*len(face_offs));
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned OBB memory.");
        }

    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

    // construct bounding box tree
    for (unsigned int i = 0; i < len(face_offs)-1; ++i)
        {
        std::vector<vec3<OverlapReal> > face_vec;

        unsigned int nverts = result.data.face_offs[i+1] - result.data.face_offs[i];
        if (nverts > 3 && R != OverlapReal(0.0))
            {
            throw std::runtime_error("With finite rounding radii, only faces with <= 3 vertices are supported.\n");
            }

        for (unsigned int j = result.data.face_offs[i]; j < result.data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v;
            v.x = result.data.verts.x[result.data.face_verts[j]];
            v.y = result.data.verts.y[result.data.face_verts[j]];
            v.z = result.data.verts.z[result.data.face_verts[j]];

            face_vec.push_back(v);
            }
        obbs[i] = hpmc::detail::compute_obb(face_vec, result.data.verts.sweep_radius);
        internal_coordinates.push_back(face_vec);
        }

    ShapePolyhedron::gpu_tree_type::obb_tree_type tree;
    tree.buildTree(obbs, internal_coordinates, result.data.verts.sweep_radius, len(face_offs)-1);
    result.tree = ShapePolyhedron::gpu_tree_type(tree);
    free(obbs);

    // set the diameter
    result.data.verts.diameter = 2*(sqrt(radius_sq)+result.data.verts.sweep_radius);

    return result;
    }

//! Helper function to build poly3d_verts from python
template<unsigned int max_verts>
poly3d_verts<max_verts> make_poly3d_verts(boost::python::list verts, OverlapReal sweep_radius, bool ignore_stats)
    {
    if (len(verts) > max_verts)
        throw std::runtime_error("Too many polygon vertices");

    poly3d_verts<max_verts> result;
    result.N = len(verts);
    result.sweep_radius = sweep_radius;
    result.ignore = ignore_stats;

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < len(verts); i++)
        {
        vec3<OverlapReal> vert = vec3<OverlapReal>(extract<OverlapReal>(verts[i][0]), extract<OverlapReal>(verts[i][1]), extract<OverlapReal>(verts[i][2]));
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        result.z[i] = vert.z;
        radius_sq = max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = len(verts); i < max_verts; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        result.z[i] = 0;
        }

    // set the diameter
    result.diameter = 2*(sqrt(radius_sq) + sweep_radius);

    return result;
    }

//! Helper function to build faceted_sphere_params from python
faceted_sphere_params make_faceted_sphere(boost::python::list normals, boost::python::list offsets,
    boost::python::list vertices, Scalar diameter, boost::python::tuple origin, bool ignore_stats)
    {
    if (len(normals) > MAX_SPHERE_FACETS)
        throw std::runtime_error("Too many face normals");

    if (len(vertices) > MAX_FPOLY3D_VERTS)
        throw std::runtime_error("Too many vertices");

    if (len(offsets) != len(normals))
        throw std::runtime_error("Number of normals unequal number of offsets");

    faceted_sphere_params result;
    result.ignore = ignore_stats;
    result.N = len(normals);

    // extract the normals from the python list
    for (unsigned int i = 0; i < len(normals); i++)
        {
        result.n[i] = vec3<OverlapReal>(extract<OverlapReal>(normals[i][0]), extract<OverlapReal>(normals[i][1]), extract<OverlapReal>(normals[i][2]));
        result.offset[i] = extract<OverlapReal>(offsets[i]);
        }
    for (unsigned int i = len(normals); i < MAX_SPHERE_FACETS; i++)
        {
        result.n[i] = vec3<OverlapReal>(0,0,0);
        result.offset[i] = 0.0;
        }

    // extract the vertices from the python list
    result.verts=make_poly3d_verts<MAX_FPOLY3D_VERTS>(vertices, 0.0, false);

    // set the diameter
    result.diameter = diameter;

    result.insphere_radius = diameter/Scalar(2.0);

    // set the origin
    result.origin = vec3<OverlapReal>(extract<OverlapReal>(origin[0]), extract<OverlapReal>(origin[1]), extract<OverlapReal>(origin[2]));

    // compute insphere radius
    for (unsigned int i = 0; i < result.N; ++i)
        {
        Scalar rsq = result.offset[i]*result.offset[i]/dot(result.n[i],result.n[i]);
        // is the origin inside the shape?
        if (result.offset < 0)
            {
            if (rsq < result.insphere_radius*result.insphere_radius)
                {
                result.insphere_radius = fast::sqrt(rsq);
                }
            }
        else
            {
            result.insphere_radius = OverlapReal(0.0);
            }
        }

    // add the edge-sphere vertices
    ShapeFacetedSphere::initializeVertices(result);

    return result;
    }

//! Helper function to build sphinx3d_verts from python
sphinx3d_params make_sphinx3d_params(boost::python::list diameters, boost::python::list centers, bool ignore_stats)
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
        OverlapReal d = extract<OverlapReal>(diameters[i]);
        result.center[i] = vec3<OverlapReal>(extract<OverlapReal>(centers[i][0]), extract<OverlapReal>(centers[i][1]), extract<OverlapReal>(centers[i][2]));
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
union_params<Shape> make_union_params(boost::python::list _members,
                                                boost::python::list positions,
                                                boost::python::list orientations,
                                                bool ignore_stats)
    {
    union_params<Shape> result;

    result.N = len(_members);
    if (result.N > hpmc::detail::MAX_MEMBERS)
        {
        throw std::runtime_error("Too many constituent particles");
        }
    if (len(positions) != result.N)
        {
        throw std::runtime_error("Number of member positions not equal to number of members");
        }
    if (len(orientations) != result.N)
        {
        throw std::runtime_error("Number of member orientations not equal to number of members");
        }

    result.ignore = ignore_stats;

    hpmc::detail::OBB *obbs;
    int retval = posix_memalign((void**)&obbs, 32, sizeof(hpmc::detail::OBB)*result.N);
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned OBB memory.");
        }

    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;

    // extract member parameters, posistions, and orientations and compute the radius along the way
    OverlapReal diameter = OverlapReal(0.0);
    for (unsigned int i = 0; i < result.N; i++)
        {
        typename Shape::param_type param = extract<typename Shape::param_type>(_members[i]);
        vec3<Scalar> pos = vec3<Scalar>(extract<Scalar>(positions[i][0]), extract<Scalar>(positions[i][1]), extract<Scalar>(positions[i][2]));
        Scalar s = extract<Scalar>(orientations[i][0]);
        Scalar x = extract<Scalar>(orientations[i][1]);
        Scalar y = extract<Scalar>(orientations[i][2]);
        Scalar z = extract<Scalar>(orientations[i][3]);
        quat<Scalar> orientation(s, vec3<Scalar>(x,y,z));
        result.mparams[i] = param;
        result.mpos[i] = pos;
        result.morientation[i] = orientation;

        Shape dummy(quat<Scalar>(), param);
        Scalar d = sqrt(dot(pos,pos));
        diameter = max(diameter, OverlapReal(2*d + dummy.getCircumsphereDiameter()));

        obbs[i] = detail::OBB(dummy.getAABB(pos));
        }

    // set the diameter
    result.diameter = diameter;

    // build tree and store GPU accessible version in parameter structure
    detail::union_gpu_tree_type::obb_tree_type tree;
    tree.buildTree(obbs, result.N);
    free(obbs);
    result.tree = detail::union_gpu_tree_type(tree);

    return result;
    }

template< typename ShapeParamType >
struct get_max_verts { /* nothing here */ }; // will probably get an error if you use it with the wrong type.

template< template<unsigned int> class ShapeParamType, unsigned int _max_verts >
struct get_max_verts< ShapeParamType<_max_verts> > { static const unsigned int max_verts=_max_verts; };

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
    typename get_param_data_type<ShapePolyhedron>::type& operator()(ParamType& param) { return param.data; }
    template< class ParamType >
    const typename get_param_data_type<ShapePolyhedron>::type& operator()(const ParamType& param) const  { return param.data; }
    };

template < typename Shape , typename AccessType = access<Shape> >
class shape_param_proxy // base class to avoid adding the ignore flag logic to every other class and holds the integrator pointer + typeid
{
protected:
    typedef typename Shape::param_type param_type;
public:
    shape_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : m_mc(mc), m_typeid(typendx), m_access(acc) {}
    //!Ignore flag for acceptance statistics
    bool getIgnoreStatistics() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return (m_access(h_params.data[m_typeid]).ignore & IGNORE_STATS);
        }

    //!Ignore flag for overlaps
    bool getIgnoreOverlaps() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return (m_access(h_params.data[m_typeid]).ignore & IGNORE_OVRLP);
        }

    void setIgnoreStatistics(bool stat)
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        if(stat)    m_access(h_params.data[m_typeid]).ignore |= IGNORE_STATS;
        else        m_access(h_params.data[m_typeid]).ignore &= ~IGNORE_STATS;
        }

    void setIgnoreOverlaps(bool ovrlps)
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        if(ovrlps)  m_access(h_params.data[m_typeid]).ignore |= IGNORE_OVRLP;
        else        m_access(h_params.data[m_typeid]).ignore &= ~IGNORE_OVRLP;
        }

protected:
    boost::shared_ptr< IntegratorHPMCMono<Shape> > m_mc;
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
    sphere_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    OverlapReal getDiameter()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return OverlapReal(2.0)*m_access(h_params.data[m_typeid]).radius;
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
    ell_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx, acc) {}

    OverlapReal getX()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).x;
        }

    OverlapReal getY()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).y;
        }

    OverlapReal getZ()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).z;
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
    poly2d_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    boost::python::list getVerts() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return poly2d_verts_to_python(m_access(h_params.data[m_typeid]));
        }

    OverlapReal getSweepRadius() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).sweep_radius;
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
    static const unsigned int max_verts = get_max_verts<param_type>::max_verts;
public:
    typedef poly3d_verts<max_verts> access_type;
    poly3d_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc) {}

    boost::python::list getVerts() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return poly3d_verts_to_python(m_access(h_params.data[m_typeid]));
        }

    OverlapReal getSweepRadius() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).sweep_radius;
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
    polyhedron_param_proxy(boost::shared_ptr< IntegratorHPMCMono<Shape> > mc, unsigned int typendx, const AccessType& acc = AccessType()) : shape_param_proxy<Shape, AccessType>(mc,typendx,acc){}

    boost::python::list getVerts()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        return poly3d_verts_to_python(m_access(h_params.data[m_typeid]).verts);
        }

    boost::python::list getFaces()
        {
        boost::python::list faces;
        // populate faces.
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        access_type& param = m_access(h_params.data[m_typeid]);
        for(size_t i = 0; i < param.n_faces; i++)
            {
            boost::python::list face;
            for(unsigned int f = param.face_offs[i]; f < param.face_offs[i+1]; f++)
                {
                face.append(param.face_verts[f]);
                }
            faces.append(face);
            }
        return faces;
        }

    OverlapReal getSweepRadius() const
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return m_access(h_params.data[m_typeid]).verts.sweep_radius;
        }
};

template< typename Shape, class AccessType = access<Shape> >
class faceted_sphere_param_proxy : public shape_param_proxy<Shape, AccessType>
{
    using shape_param_proxy<Shape, AccessType>::m_mc;
    using shape_param_proxy<Shape, AccessType>::m_typeid;
    using shape_param_proxy<Shape, AccessType>::m_access;
protected:
    typedef typename shape_param_proxy<Shape, AccessType>::param_type param_type;
public:
    typedef ShapeFacetedSphere::param_type access_type;
    faceted_sphere_param_proxy(boost::shared_ptr< IntegratorHPMCMono<ShapeFacetedSphere> > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy<Shape, AccessType>(mc,typendx,acc)
        {}

    boost::python::list getVerts()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        return poly3d_verts_to_python(m_access(h_params.data[m_typeid]).verts);
        }

    boost::python::list getNormals()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list normals;
        for(size_t i = 0; i < param.N; i++ ) normals.append(vec3_to_python(param.n[i]));
        return normals;
        }

    boost::python::list getOrigin()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        return vec3_to_python(param.origin);
        }

    OverlapReal getDiameter()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        return param.diameter;
        }

    boost::python::list getOffsets()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list offsets;
        for(size_t i = 0; i < param.N; i++) offsets.append(param.offset[i]);
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
    sphinx3d_param_proxy(boost::shared_ptr< IntegratorHPMCMono<ShapeSphinx> > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy<Shape, AccessType>(mc,typendx,acc)
        {}

    boost::python::list getCenters()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list centers;
        for(size_t i = 0; i < param.N; i++) centers.append(vec3_to_python(param.center[i]));
        return centers;
        }

    boost::python::list getDiameters()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list diams;
        for(size_t i = 0; i < param.N; i++) diams.append(param.diameter[i]);
        return diams;
        }

    OverlapReal getCircumsphereDiameter()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
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
    shape_union_param_proxy(boost::shared_ptr< IntegratorHPMCMono< Shape > > mc, unsigned int typendx, const AccessType& acc = AccessType())
        : shape_param_proxy< Shape, AccessType>(mc,typendx,acc)
        {}
    boost::python::list getPositions()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list pos;
        for(size_t i = 0; i < param.N; i++) pos.append(vec3_to_python(param.mpos[i]));
        return pos;
        }

    boost::python::list getOrientations()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list orient;
        for(size_t i = 0; i < param.N; i++)
            orient.append(quat_to_python(param.morientation[i]));
        return orient;
        }

    boost::python::list getMembers()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        boost::python::list members;
        for(size_t i = 0; i < param.N; i++)
            {
            access_shape_union_members<ShapeUnionType> acc(i);
            boost::shared_ptr< proxy_type > p(new proxy_type(m_mc, m_typeid, acc));
            members.append(p);
            }
        return members;
        }


    OverlapReal getDiameter()
        {
        ArrayHandle<param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
        access_type& param = m_access(h_params.data[m_typeid]);
        return param.diameter;
        }
};

} // end namespace detail

template<class Shape, class AccessType>
void export_shape_param_proxy(const std::string& name)
    {
    // export the base class.
    using detail::shape_param_proxy;
    boost::python::class_<shape_param_proxy<Shape, AccessType>, boost::shared_ptr< shape_param_proxy<Shape, AccessType> > >
        (   name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<Shape> >, unsigned int>()
        )
    .add_property("ignore_statistics", &shape_param_proxy<Shape>::getIgnoreStatistics, &shape_param_proxy<Shape>::setIgnoreStatistics)
    ;
    }

template<class ShapeType, class AccessType>
void export_sphere_proxy(const std::string& class_name)
    {
    using detail::shape_param_proxy;
    using detail::sphere_param_proxy;
    typedef shape_param_proxy<ShapeType, AccessType>    proxy_base;
    typedef sphere_param_proxy<ShapeType, AccessType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, AccessType>(base_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("diameter", &proxy_class::getDiameter)
    ;
    }

void export_ell_proxy()
    {
    using detail::shape_param_proxy;
    using detail::ell_param_proxy;
    typedef ShapeEllipsoid                  ShapeType;
    typedef shape_param_proxy<ShapeType>    proxy_base;
    typedef ell_param_proxy<ShapeType>      proxy_class;
    std::string class_name="ell_param_proxy";
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("a", &proxy_class::getX)
    .add_property("b", &proxy_class::getY)
    .add_property("c", &proxy_class::getZ)
    ;
    }

template<class ShapeType>
void export_poly2d_proxy(std::string class_name, bool sweep_radius_valid)
    {
    using detail::shape_param_proxy;
    using detail::poly2d_param_proxy;
    typedef shape_param_proxy<ShapeType>    proxy_base;
    typedef poly2d_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";
    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    if (sweep_radius_valid)
        {
        boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
            (   class_name.c_str(),
                boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
            )
        .add_property("vertices", &proxy_class::getVerts)
        .add_property("sweep_radius", &proxy_class::getSweepRadius)
        ;
        }
    else
        {
        boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
            (   class_name.c_str(),
                boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
            )
        .add_property("vertices", &proxy_class::getVerts)
        ;
        }
    }

template<class ShapeType>
void export_poly3d_proxy(std::string class_name, bool sweep_radius_valid)
    {
    using detail::shape_param_proxy;
    using detail::poly3d_param_proxy;
    typedef shape_param_proxy<ShapeType>    proxy_base;
    typedef poly3d_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    if (sweep_radius_valid)
        {
        boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
            (   class_name.c_str(),
                boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
            )
        .add_property("vertices", &proxy_class::getVerts)
        .add_property("sweep_radius", &proxy_class::getSweepRadius)
        ;
        }
    else
        {
        boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
            (   class_name.c_str(),
                boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
            )
        .add_property("vertices", &proxy_class::getVerts)
        ;
        }
    }

void export_polyhedron_proxy(std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::polyhedron_param_proxy;
    typedef ShapePolyhedron                     ShapeType;
    typedef shape_param_proxy<ShapeType>        proxy_base;
    typedef polyhedron_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("vertices", &proxy_class::getVerts)
    .add_property("faces", &proxy_class::getFaces)
    .add_property("sweep_radius", &proxy_class::getSweepRadius)
    ;
    }

void export_faceted_sphere_proxy(std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::faceted_sphere_param_proxy;
    typedef ShapeFacetedSphere                  ShapeType;
    typedef shape_param_proxy<ShapeType>        proxy_base;
    typedef faceted_sphere_param_proxy<ShapeType>   proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("vertices", &proxy_class::getVerts)
    .add_property("normals", &proxy_class::getNormals)
    .add_property("origin", &proxy_class::getOrigin)
    .add_property("diameter", &proxy_class::getDiameter)
    .add_property("offsets", &proxy_class::getOffsets)
    ;

    }

void export_sphinx_proxy(std::string class_name)
    {
    using detail::shape_param_proxy;
    using detail::sphinx3d_param_proxy;
    typedef ShapeSphinx                         ShapeType;
    typedef shape_param_proxy<ShapeType>        proxy_base;
    typedef sphinx3d_param_proxy<ShapeType>     proxy_class;
    std::string base_name=class_name+"_base";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("centers", &proxy_class::getCenters)
    .add_property("diameters", &proxy_class::getDiameters)
    .add_property("diameter", &proxy_class::getCircumsphereDiameter)
    ;

    }

template<class Shape, class ExportFunction >
void export_shape_union_proxy(std::string class_name, ExportFunction& export_member_proxy)
    {
    using detail::shape_param_proxy;
    using detail::shape_union_param_proxy;
    typedef ShapeUnion<Shape>                               ShapeType;
    typedef shape_param_proxy<ShapeType>                    proxy_base;
    typedef shape_union_param_proxy<ShapeType, ShapeType>   proxy_class;

    std::string base_name=class_name+"_base";
    std::string member_name=class_name+"_member_proxy";

    export_shape_param_proxy<ShapeType, detail::access<ShapeType> >(base_name);
    export_member_proxy(member_name);
    boost::python::class_<proxy_class, boost::shared_ptr< proxy_class >, boost::python::bases< proxy_base > >
        (   class_name.c_str(),
            boost::python::init<boost::shared_ptr< IntegratorHPMCMono<ShapeType> >, unsigned int>()
        )
    .add_property("centers", &proxy_class::getPositions)
    .add_property("orientations", &proxy_class::getOrientations)
    .add_property("diameter", &proxy_class::getDiameter)
    .add_property("members", &proxy_class::getMembers)
    ;

    }



void export_shape_params()
    {
    export_sphere_proxy<ShapeSphere, detail::access<ShapeSphere> >("sphere_param_proxy");
    export_ell_proxy();
    export_poly2d_proxy<ShapeConvexPolygon>("convex_polygon_param_proxy", false);
    export_poly2d_proxy<ShapeSpheropolygon>("convex_spheropolygon_param_proxy", true);
    export_poly2d_proxy<ShapeSimplePolygon>("simple_polygon_param_proxy", false);

    export_poly3d_proxy< ShapeConvexPolyhedron<8> >("convex_polyhedron_param_proxy8", false);
    export_poly3d_proxy< ShapeConvexPolyhedron<16> >("convex_polyhedron_param_proxy16", false);
    export_poly3d_proxy< ShapeConvexPolyhedron<32> >("convex_polyhedron_param_proxy32", false);
    export_poly3d_proxy< ShapeConvexPolyhedron<64> >("convex_polyhedron_param_proxy64", false);
    export_poly3d_proxy< ShapeConvexPolyhedron<128> >("convex_polyhedron_param_proxy128", false);

    export_poly3d_proxy< ShapeSpheropolyhedron<8> >("convex_spheropolyhedron_param_proxy8", true);
    export_poly3d_proxy< ShapeSpheropolyhedron<16> >("convex_spheropolyhedron_param_proxy16", true);
    export_poly3d_proxy< ShapeSpheropolyhedron<32> >("convex_spheropolyhedron_param_proxy32", true);
    export_poly3d_proxy< ShapeSpheropolyhedron<64> >("convex_spheropolyhedron_param_proxy64", true);
    export_poly3d_proxy< ShapeSpheropolyhedron<128> >("convex_spheropolyhedron_param_proxy128", true);

    export_polyhedron_proxy("polyhedron_param_proxy");
    export_faceted_sphere_proxy("faceted_sphere_param_proxy");
    export_sphinx_proxy("sphinx3d_param_proxy");
    export_shape_union_proxy<ShapeSphere>("sphere_union_param_proxy", export_sphere_proxy<ShapeUnion<ShapeSphere>, detail::access_shape_union_members< ShapeUnion<ShapeSphere> > >);
    }

} // end namespace hpmc


#endif // end __SHAPE_PROXY_H__
