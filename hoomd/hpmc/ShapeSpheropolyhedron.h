// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeConvexPolyhedron.h"
#include "ShapeSphere.h" //< For the base template of test_overlap
#include "XenoCollide3D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

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

namespace hoomd
    {
namespace hpmc
    {
//! Convex (Sphero)Polyhedron shape template
/*! ShapeSpheropolyhedron represents a convex polygon swept out by a sphere with special cases. A
   shape with zero vertices is a sphere centered at the particle location. This is degenerate with
   the one-vertex case and marginal more performant. As a consequence of the algorithm, two vertices
   with a sweep radius represents a prolate spherocylinder but not according to any standard
   convention and a simulation of spherocylinders using ShapeSpheropolyhedron will not perform as
   efficiently as a more specialized algorithm.

    The parameter defining a polyhedron is a structure containing a list of N vertices, centered on
   0,0. In fact, it is
    **required** that the origin is inside the shape, and it is best if the origin is the center of
   mass.

    ShapeSpheropolygon interprets two additional fields in the verts struct that
   ShapeConvexPolyhedron lacks. The first is sweep_radius which defines the radius of the sphere to
   sweep around the polygon. The 2nd is ignore. When two shapes are checked for overlap, if both of
   them have ignore set to true (non-zero) then there is assumed to be no collision. This is
   intended for use with the penetrable hard-sphere model for depletants, but could be useful in
   other cases.

    \ingroup shape
*/
struct ShapeSpheropolyhedron
    {
    //! Define the parameter type
    typedef detail::PolyhedronVertices param_type;

    /// Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    //! Initialize a polyhedron
    DEVICE ShapeSpheropolyhedron(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const
        {
        if (verts.N > 1)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    //! Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const
        {
        return verts.ignore;
        }

    //! Get the circumsphere diameter
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-sphere radius
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        // Generate the AABB of a bounding sphere, computing tight fitting AABBs is slow.
        return hoomd::detail::AABB(pos, verts.diameter / Scalar(2));
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        detail::OBB obb = verts.obb;
        obb.affineTransform(orientation, pos);
        return obb;
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using
    //! threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
        return false;
        }

    //! Returns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return true;
        }

    quat<Scalar> orientation; //!< Orientation of the polyhedron

    const detail::PolyhedronVertices& verts; //!< Vertices
    };

//! Convex polyhedron overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap(const vec3<Scalar>& r_ab,
                                const ShapeSpheropolyhedron& a,
                                const ShapeSpheropolyhedron& b,
                                unsigned int& err)
    {
    vec3<ShortReal> dr = r_ab;

    ShortReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();

    return xenocollide_3d(detail::SupportFuncConvexPolyhedron(a.verts, a.verts.sweep_radius),
                          detail::SupportFuncConvexPolyhedron(b.verts, b.verts.sweep_radius),
                          rotate(conj(quat<ShortReal>(a.orientation)), dr),
                          conj(quat<ShortReal>(a.orientation)) * quat<ShortReal>(b.orientation),
                          DaDb / ShortReal(2.0),
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

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeSpheropolyhedron& spoly)
    {
    std::ostringstream shapedef;
    auto& verts = spoly.verts;
    unsigned int nverts = verts.N;
    if (nverts == 0)
        {
        throw std::runtime_error("Shape definition not supported for 0-vertex spheropolyhedra.");
        }
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
        shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": " << verts.sweep_radius
                 << ", \"vertices\": [";
        for (unsigned int i = 0; i < nverts - 1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << ", " << verts.z[i] << "], ";
            }
        shapedef << "[" << verts.x[nverts - 1] << ", " << verts.y[nverts - 1] << ", "
                 << verts.z[nverts - 1] << "]]}";
        }
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE
#endif //__SHAPE_SPHEROPOLYHEDRON_H__
