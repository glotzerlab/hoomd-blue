// Copyright (c) 2009-2018 The Regents of the University of Michigan
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
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <iostream>
#endif

namespace hpmc
{

namespace detail
{

//! Support function for ShapeSpheropolyhedron
/*! SupportFuncSpheropolyhedron is a functor that computes the support function for ShapeSpheropolyhedron. For a given
    input vector in local coordinates, it finds the vertex most in that direction and then extends it out further
    by the sweep radius.

    There are some current features under consideration for special handling of 0 and 1-vertex inputs. See
    ShapeSphereopolyhedron for documentation on these cases.

    \ingroup minkowski
*/
class SupportFuncSpheropolyhedron
    {
    public:
        //! Construct a support function for a convex spheropolyhedron
        /*! \param _verts Polyhedron vertices and additional parameters
        */
        DEVICE SupportFuncSpheropolyhedron(const poly3d_verts& _verts)
            : verts(_verts)
            {
            }

        //! Compute the support function
        /*! \param n Normal vector input (in the local frame)
            \returns Local coords of the point furthest in the direction of n
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& n) const
            {
            // get the support function of the underlying convex polyhedron
            vec3<OverlapReal> max_poly3d = SupportFuncConvexPolyhedron(verts)(n);
            // add to that the support mapping of the sphere
            vec3<OverlapReal> max_sphere = (verts.sweep_radius * fast::rsqrt(dot(n,n))) * n;

            return max_poly3d + max_sphere;
            }

    private:
        const poly3d_verts& verts;        //!< Vertices of the polyhedron
    };

//! Projection function for ShapeSpheropolyhedron
/*! ProjectionFuncConvexPolyhedron is a functor that computes the projection function for ShapePolyhedron. For a given
    input point in local coordinates, it finds the point on the sphere-swept shape closest to that point.

    \ingroup minkowski
*/
class ProjectionFuncSpheropolyhedron
    {
    public:
        //! Construct a projection function for a convex spheropolyhedron
        /*! \param _verts Polyhedron vertices and additional parameters
        */
        DEVICE ProjectionFuncSpheropolyhedron(const poly3d_verts& _verts)
            : verts(_verts)
            {
            }

        //! Compute the projection
        /*! \param p Point to compute the projection for
            \returns Local coords of the point in the shape closest to p
        */
        DEVICE vec3<OverlapReal> operator() (const vec3<OverlapReal>& p) const
            {
            // get the projection function of the underlying convex polyhedron
            vec3<OverlapReal> proj_poly3d = ProjectionFuncConvexPolyhedron(verts)(p);

            vec3<OverlapReal> del = p - proj_poly3d;
            OverlapReal dsq = dot(del,del);
            if (dsq > verts.sweep_radius*verts.sweep_radius)
                {
                // add the sphere radius in direction of closest approach, or the closest point inside the sphere
                OverlapReal d = fast::sqrt(dsq);
                return proj_poly3d + verts.sweep_radius/d*del;
                }
            else
                // point is inside base shape
                return p;
            }

    private:
        const poly3d_verts& verts;        //!< Vertices of the polyhedron
    };


}; // end namespace detail

//! Convex (Sphero)Polyhedron shape template
/*! ShapeSpheropolyhedron represents a convex polygon swept out by a sphere with special cases. A shape with zero
    vertices is a sphere centered at the particle location. This is degenerate with the one-vertex case and marginal
    more performant. As a consequence of the algorithm, two vertices with a sweep radius represents a prolate
    spherocylinder but not according to any standard convention and a simulation of spherocylinders using
    ShapeSpheropolyhedron will not perform as efficiently as a more specialized algorithm.

    The parameter defining a polyhedron is a structure containing a list of N vertices, centered on 0,0. In fact, it is
    **required** that the origin is inside the shape, and it is best if the origin is the center of mass.

    ShapeSpheropolygon interprets two additional fields in the verts struct that ShapeConvexPolyhedron lacks.
    The first is sweep_radius which defines the radius of the sphere to sweep around the polyong. The 2nd
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
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
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

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeSpheropolyhedron& a,
    const ShapeSpheropolyhedron &b)
    {
    vec3<OverlapReal> dr(r_ab);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

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

    if (sweep_radius_a == Scalar(0.0) && sweep_radius_b == Scalar(0.0))
        {
        OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

        return xenocollide_3d(detail::SupportFuncSpheropolyhedron(a.verts),
                              detail::SupportFuncSpheropolyhedron(b.verts),
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
    else
        {
        return detail::map_two(a,b,
            detail::SupportFuncSpheropolyhedron(a.verts),
            detail::SupportFuncSpheropolyhedron(b.verts),
            detail::ProjectionFuncSpheropolyhedron(a.verts),
            detail::ProjectionFuncSpheropolyhedron(b.verts),
            dr,
            err,
            sweep_radius_a,
            sweep_radius_b);
        }
    }

//! Test for a common point in the intersection of three spheropolyhedra
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
DEVICE inline bool test_overlap_three(const ShapeSpheropolyhedron& a,
    const ShapeSpheropolyhedron& b,
    const ShapeSpheropolyhedron& c,
    const vec3<Scalar>& ab_t, const vec3<Scalar>& ac_t, unsigned int &err,
    Scalar sweep_radius_a, Scalar sweep_radius_b, Scalar sweep_radius_c)
    {
    return detail::map_three(a,b,c,
        detail::SupportFuncSpheropolyhedron(a.verts),
        detail::SupportFuncSpheropolyhedron(b.verts),
        detail::SupportFuncSpheropolyhedron(c.verts),
        detail::ProjectionFuncSpheropolyhedron(a.verts),
        detail::ProjectionFuncSpheropolyhedron(b.verts),
        detail::ProjectionFuncSpheropolyhedron(c.verts),
        vec3<OverlapReal>(ab_t),
        vec3<OverlapReal>(ac_t),
        err,
        sweep_radius_a,
        sweep_radius_b,
        sweep_radius_c);
    }

}; // end namespace hpmc

#endif //__SHAPE_SPHEROPOLYHEDRON_H__
