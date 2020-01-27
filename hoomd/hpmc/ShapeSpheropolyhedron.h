// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeConvexPolyhedron.h"
#include "XenoCollide3D.h"

#ifndef __SHAPE_SPHEROPOLYHEDRON_H__
#define __SHAPE_SPHEROPOLYHEDRON_H__

/*! \file ShapeSpheropolyhedron.h
    \brief Defines the Spheropolyhedron shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

#ifndef __HIPCC__
#include <vector>
#endif

namespace hpmc
{

//! Convex (Sphero)Polyhedron shape template
/*! ShapeSpheropolyhedron represents a convex polygon swept out by a sphere with special cases. A shape with zero
    vertices is a sphere centered at the particle location. This is degenerate with the one-vertex case and marginal
    more performant. As a consequence of the algorithm, two vertices with a sweep radius represents a prolate
    spherocylinder but not according to any standard convention and a simulation of spherocylinders using
    ShapeSpheropolyhedron will not perform as efficiently as a more specialized algorithm.

    The parameter defining a polyhedron is a structure containing a list of N vertices, centered on 0,0. In fact, it is
    **required** that the origin is inside the shape, and it is best if the origin is the center of mass.

    ShapeSpheropolygon interprets two additional fields in the verts struct that ShapeConvexPolyhedron lacks.
    The first is sweep_radius which defines the radius of the sphere to sweep around the polygon. The 2nd
    is ignore. When two shapes are checked for overlap, if both of them have ignore set to true (non-zero) then
    there is assumed to be no collision. This is intended for use with the penetrable hard-sphere model for depletants,
    but could be useful in other cases.

    \ingroup shape
*/
struct ShapeSpheropolyhedron
    {
    //! Define the parameter type
    typedef detail::poly3d_verts param_type;

    //! Initialize a polyhedron
    DEVICE ShapeSpheropolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const {
        if (verts.N > 1)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return verts.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-sphere radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        unsigned int nverts = verts.N;
        if (nverts == 1)
            {
            shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << verts.diameter << "}";
            }
        else if (nverts == 2)
            {
            throw std::runtime_error("Shape definition not supported for 2-vertex spheropolyhedra");
            }
        else
            {
            shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": " << verts.sweep_radius << ", \"vertices\": [";
            for (unsigned int i = 0; i < nverts-1; i++)
                {
                shapedef << "[" << verts.x[i] << ", " << verts.y[i] << ", " << verts.z[i] << "], ";
                }
            shapedef << "[" << verts.x[nverts-1] << ", " << verts.y[nverts-1] << ", " << verts.z[nverts-1] << "]]}";
            }
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // generate a tight fitting AABB
        // detail::SupportFuncSpheropolyhedron sfunc(verts);

        // // use support function of the to determine the furthest extent in each direction
        // quat<OverlapReal> o(orientation);
        // vec3<OverlapReal> e_x(1,0,0);
        // vec3<OverlapReal> e_y(0,1,0);
        // vec3<OverlapReal> e_z(0,0,1);
        // vec3<OverlapReal> s_x = rotate(o, sfunc(rotate(conj(o),e_x)));
        // vec3<OverlapReal> s_y = rotate(o, sfunc(rotate(conj(o),e_y)));
        // vec3<OverlapReal> s_z = rotate(o, sfunc(rotate(conj(o),e_z)));
        // vec3<OverlapReal> s_neg_x = rotate(o, sfunc(rotate(conj(o),-e_x)));
        // vec3<OverlapReal> s_neg_y = rotate(o, sfunc(rotate(conj(o),-e_y)));
        // vec3<OverlapReal> s_neg_z = rotate(o, sfunc(rotate(conj(o),-e_z)));

        // // translate out from the position by the furthest extents
        // vec3<Scalar> upper(pos.x + s_x.x, pos.y + s_y.y, pos.z + s_z.z);
        // vec3<Scalar> lower(pos.x + s_neg_x.x, pos.y + s_neg_y.y, pos.z + s_neg_z.z);

        // return detail::AABB(lower, upper);
        // ^^^^^^ The above method is slow, just use the bounding sphere
        return detail::AABB(pos, verts.diameter/Scalar(2));
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        detail::OBB obb = verts.obb;
        obb.affineTransform(orientation, pos);
        return obb;
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    //! Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return true;
        }

    quat<Scalar> orientation;    //!< Orientation of the polyhedron

    const detail::poly3d_verts& verts;     //!< Vertices
    };

//! Convex polyhedron overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \param sweep_radius_a Radius of a sphere to sweep the first shape by
    \param sweep_radius_b Radius of a sphere to sweep the second shape by
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                 const ShapeSpheropolyhedron& a,
                                 const ShapeSpheropolyhedron& b,
                                 unsigned int& err,
                                 Scalar sweep_radius_a,
                                 Scalar sweep_radius_b)
    {
    vec3<OverlapReal> dr = r_ab;

    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    return xenocollide_3d(detail::SupportFuncConvexPolyhedron(a.verts,a.verts.sweep_radius+sweep_radius_a),
                          detail::SupportFuncConvexPolyhedron(b.verts,b.verts.sweep_radius+sweep_radius_b),
                          rotate(conj(quat<OverlapReal>(a.orientation)),dr),
                          conj(quat<OverlapReal>(a.orientation)) * quat<OverlapReal>(b.orientation),
                          DaDb/2.0,
                          err);

    /*
    return gjke_3d(detail::SupportFuncSpheropolyhedron(a.verts),
                   detail::SupportFuncSpheropolyhedron(b.verts),
                          dr,
                          a.orientation,
                          b.orientation,
                          DaDb/2.0,
                          err);
    */
    }

//! Test for overlap of a third particle with the intersection of two shapes
/*! \param a First shape to test
    \param b Second shape to test
    \param c Third shape to test
    \param ab_t Position of second shape relative to first
    \param ac_t Position of third shape relative to first
    \param err Output variable that is incremented upon non-convergence
    \param sweep_radius_a Radius of a sphere to sweep the first shape by
    \param sweep_radius_b Radius of a sphere to sweep the second shape by
*/
template<>
DEVICE inline bool test_overlap_intersection(const ShapeSpheropolyhedron& a,
    const ShapeSpheropolyhedron& b,
    const ShapeSpheropolyhedron& c,
    const vec3<Scalar>& ab_t, const vec3<Scalar>& ac_t, unsigned int &err,
    Scalar sweep_radius_a, Scalar sweep_radius_b, Scalar sweep_radius_c)
    {
    return detail::map_three(a,b,c,
        detail::SupportFuncConvexPolyhedron(a.verts,a.verts.sweep_radius+sweep_radius_a),
        detail::SupportFuncConvexPolyhedron(b.verts,b.verts.sweep_radius+sweep_radius_b),
        detail::SupportFuncConvexPolyhedron(c.verts,c.verts.sweep_radius+sweep_radius_c),
        detail::ProjectionFuncConvexPolyhedron(a.verts,a.verts.sweep_radius+sweep_radius_a),
        detail::ProjectionFuncConvexPolyhedron(b.verts,b.verts.sweep_radius+sweep_radius_b),
        detail::ProjectionFuncConvexPolyhedron(c.verts,c.verts.sweep_radius+sweep_radius_c),
        vec3<OverlapReal>(ab_t),
        vec3<OverlapReal>(ac_t),
        err);
    }

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_SPHEROPOLYHEDRON_H__
