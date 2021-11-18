// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_FIELD_WALL_H_
#define _EXTERNAL_FIELD_WALL_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/
#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"

#include "ExternalField.h"
#include "IntegratorHPMCMono.h"

#include <limits>
#include <tuple>

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
struct SphereWall
    {
    SphereWall()
        : rsq(0), inside(false), origin(0, 0, 0), verts(new detail::PolyhedronVertices(1, false))
        {
        }
    SphereWall(Scalar r, vec3<Scalar> orig, bool ins = true)
        : rsq(r * r), inside(ins), origin(orig), verts(new detail::PolyhedronVertices(1, false))
        {
        verts->N = 0; // case for sphere (can be 0 or 1)
        verts->diameter = OverlapReal(r + r);
        verts->sweep_radius = OverlapReal(r);
        verts->ignore = 0;
        }
    SphereWall(const SphereWall& src)
        : rsq(src.rsq), inside(src.inside), origin(src.origin),
          verts(new detail::PolyhedronVertices(*src.verts))
        {
        }
    // scale all distances associated with the sphere wall by some factor alpha
    void scale(const Scalar alpha)
        {
        rsq *= alpha * alpha;
        origin *= alpha;
        verts->diameter = OverlapReal(verts->diameter * alpha);
        verts->sweep_radius = OverlapReal(verts->diameter * alpha);
        }

    Scalar rsq;
    bool inside;
    vec3<Scalar> origin;
    std::shared_ptr<detail::PolyhedronVertices> verts;
    };

struct CylinderWall
    {
    CylinderWall()
        : rsq(0), inside(false), origin(0, 0, 0), orientation(origin),
          verts(new detail::PolyhedronVertices)
        {
        }
    CylinderWall(Scalar r, vec3<Scalar> orig, vec3<Scalar> orient, bool ins = true)
        : rsq(0), inside(false), origin(0, 0, 0), orientation(origin),
          verts(new detail::PolyhedronVertices(2, false))
        {
        rsq = r * r;
        inside = ins;
        origin = orig;
        orientation = orient;

        Scalar len = sqrt(dot(orientation, orientation)); // normalize the orientation vector
        orientation /= len;

        // set the position of the vertices and the diameter later
        verts->N = 2;
        verts->sweep_radius = OverlapReal(r);
        verts->ignore = 0;
        }
    CylinderWall(const CylinderWall& src)
        : rsq(src.rsq), inside(src.inside), origin(src.origin), orientation(src.orientation),
          verts(new detail::PolyhedronVertices(*src.verts))
        {
        }
    // scale all distances associated with the sphere wall by some factor alpha
    void scale(const Scalar alpha)
        {
        rsq *= alpha * alpha;
        origin *= alpha;
        verts->sweep_radius = OverlapReal(verts->sweep_radius * alpha);
        }

    Scalar rsq;
    bool inside;
    vec3<Scalar> origin;      // center of cylinder.
    vec3<Scalar> orientation; // (normal) vector pointing in direction of long axis of cylinder
                              // (sign of vector has no meaning)
    std::shared_ptr<detail::PolyhedronVertices> verts;
    };

struct PlaneWall
    {
    PlaneWall(vec3<Scalar> nvec, vec3<Scalar> pt, bool ins = true)
        : normal(nvec), origin(pt), inside(ins)
        {
        Scalar len = sqrt(dot(normal, normal));
        normal /= len;
        d = -dot(normal, origin);
        }

    // scale all distances associated with the sphere wall by some factor alpha
    void scale(const Scalar alpha)
        {
        origin *= alpha;
        d *= alpha;
        }

    vec3<Scalar> normal; // unit normal n = (a, b, c)
    vec3<Scalar> origin; // we could remove this.
    bool inside;         // not used
    Scalar d;            // ax + by + cz + d =  0
    };

template<class WallShape, class ParticleShape>
DEVICE inline bool test_confined(const WallShape& wall,
                                 const ParticleShape& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    return false;
    }

// Spherical Walls and Spheres
template<>
DEVICE inline bool test_confined<SphereWall, ShapeSphere>(const SphereWall& wall,
                                                          const ShapeSphere& shape,
                                                          const vec3<Scalar>& position,
                                                          const vec3<Scalar>& box_origin,
                                                          const BoxDim& box)
    {
    Scalar3 t = vec_to_scalar3(position - box_origin);
    t.x = t.x - wall.origin.x;
    t.y = t.y - wall.origin.y;
    t.z = t.z - wall.origin.z;
    vec3<Scalar> shifted_pos(box.minImage(t));

    Scalar rxyz_sq = shifted_pos.x * shifted_pos.x + shifted_pos.y * shifted_pos.y
                     + shifted_pos.z * shifted_pos.z; // distance from the container origin.
    Scalar max_dist = sqrt(rxyz_sq) + (shape.getCircumsphereDiameter() / OverlapReal(2.0));
    if (!wall.inside)
        {
        // if we must be outside the wall, subtract particle radius from min_dist
        max_dist = sqrt(rxyz_sq) - (shape.getCircumsphereDiameter() / Scalar(2.0));
        // if the particle radius is larger than the distance between the particle
        // and the container, however, then ALWAYS check verts. this is equivalent
        // to two particle circumspheres overlapping.
        if (max_dist < 0)
            {
            max_dist = 0;
            }
        }

    return wall.inside ? (wall.rsq > max_dist * max_dist) : (wall.rsq < max_dist * max_dist);
    }

// Spherical Walls and Convex Polyhedra
DEVICE inline bool test_confined(const SphereWall& wall,
                                 const ShapeConvexPolyhedron& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    bool accept = true;
    Scalar3 t = vec_to_scalar3(position - box_origin);
    t.x = t.x - wall.origin.x;
    t.y = t.y - wall.origin.y;
    t.z = t.z - wall.origin.z;
    vec3<Scalar> shifted_pos(box.minImage(t));

    Scalar rxyz_sq = shifted_pos.x * shifted_pos.x + shifted_pos.y * shifted_pos.y
                     + shifted_pos.z * shifted_pos.z;
    Scalar max_dist = (sqrt(rxyz_sq) + shape.getCircumsphereDiameter() / Scalar(2.0));
    if (!wall.inside)
        {
        // if we must be outside the wall, subtract particle radius from min_dist
        max_dist = sqrt(rxyz_sq) - (shape.getCircumsphereDiameter() / Scalar(2.0));
        // if the particle radius is larger than the distance between the particle
        // and the container, however, then ALWAYS check verts. this is equivalent
        // to two particle circumspheres overlapping.
        if (max_dist < 0)
            {
            max_dist = 0;
            }
        }

    bool check_verts
        = wall.inside ? (wall.rsq <= max_dist * max_dist)
                      : (wall.rsq >= max_dist * max_dist); // condition to check vertices, dependent
                                                           // on inside or outside container

    if (check_verts)
        {
        if (wall.inside)
            {
            for (unsigned int v = 0; v < (unsigned int)shape.verts.N && accept; v++)
                {
                vec3<Scalar> pos(shape.verts.x[v], shape.verts.y[v], shape.verts.z[v]);
                vec3<Scalar> rotated_pos = rotate(shape.orientation, pos);
                rotated_pos += shifted_pos;
                rxyz_sq = rotated_pos.x * rotated_pos.x + rotated_pos.y * rotated_pos.y
                          + rotated_pos.z * rotated_pos.z;
                accept = wall.inside ? (wall.rsq > rxyz_sq) : (wall.rsq < rxyz_sq);
                }
            }
        else
            {
            // build a sphero-polyhedron and for the wall and the convex polyhedron
            quat<Scalar> q; // default is (1, 0, 0, 0)
            unsigned int err = 0;
            ShapeSpheropolyhedron wall_shape(q, *wall.verts);
            ShapeSpheropolyhedron part_shape(shape.orientation, shape.verts);

            accept = !test_overlap(shifted_pos, wall_shape, part_shape, err);
            }
        }
    return accept;
    }

// Spherical Walls and Convex Spheropolyhedra
DEVICE inline bool test_confined(const SphereWall& wall,
                                 const ShapeSpheropolyhedron& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    bool accept = true;
    Scalar3 t = vec_to_scalar3(position - box_origin);
    t.x = t.x - wall.origin.x;
    t.y = t.y - wall.origin.y;
    t.z = t.z - wall.origin.z;
    vec3<Scalar> shifted_pos(box.minImage(t));

    Scalar rxyz_sq = shifted_pos.x * shifted_pos.x + shifted_pos.y * shifted_pos.y
                     + shifted_pos.z * shifted_pos.z;
    Scalar max_dist;
    if (!wall.inside)
        {
        // if we must be outside the wall, subtract particle radius from min_dist
        max_dist = sqrt(rxyz_sq) - (shape.getCircumsphereDiameter() / Scalar(2.0));
        // if the particle radius is larger than the distance between the particle
        // and the container, however, then ALWAYS check verts. this is equivalent
        // to two particle circumspheres overlapping.
        if (max_dist < 0)
            {
            max_dist = 0;
            }
        }
    else
        {
        max_dist = sqrt(rxyz_sq) + shape.getCircumsphereDiameter() / Scalar(2.0);
        }

    bool check_verts
        = wall.inside ? (wall.rsq <= max_dist * max_dist)
                      : (wall.rsq >= max_dist * max_dist); // condition to check vertices, dependent
                                                           // on inside or outside container

    if (check_verts)
        {
        if (shape.verts.N)
            {
            if (wall.inside)
                {
                for (unsigned int v = 0; v < shape.verts.N && accept; v++)
                    {
                    vec3<Scalar> pos(shape.verts.x[v], shape.verts.y[v], shape.verts.z[v]);
                    vec3<Scalar> rotated_pos = rotate(shape.orientation, pos);
                    rotated_pos += shifted_pos;
                    rxyz_sq = rotated_pos.x * rotated_pos.x + rotated_pos.y * rotated_pos.y
                              + rotated_pos.z * rotated_pos.z;
                    Scalar tot_rxyz = sqrt(rxyz_sq) + shape.verts.sweep_radius;
                    Scalar tot_rxyz_sq = tot_rxyz * tot_rxyz;
                    accept = wall.rsq > tot_rxyz_sq;
                    }
                }
            else
                {
                // build a sphero-polyhedron and for the wall and the convex polyhedron
                quat<Scalar> q; // default is (1, 0, 0, 0)
                unsigned int err = 0;
                ShapeSpheropolyhedron wall_shape(q, *wall.verts);
                ShapeSpheropolyhedron part_shape(shape.orientation, shape.verts);

                accept = !test_overlap(shifted_pos, wall_shape, part_shape, err);
                }
            }
        // Edge case; pure sphere. In this case, check_verts implies that the
        // sphere will be outside.
        else
            {
            accept = false;
            }
        }
    return accept;
    }

// Cylindrical Walls and Spheres
template<>
DEVICE inline bool test_confined<CylinderWall, ShapeSphere>(const CylinderWall& wall,
                                                            const ShapeSphere& shape,
                                                            const vec3<Scalar>& position,
                                                            const vec3<Scalar>& box_origin,
                                                            const BoxDim& box)
    {
    Scalar3 t = vec_to_scalar3(position - box_origin);
    t.x = t.x - wall.origin.x;
    t.y = t.y - wall.origin.y;
    t.z = t.z - wall.origin.z;
    vec3<Scalar> shifted_pos(box.minImage(t));

    vec3<Scalar> dist_vec
        = cross(shifted_pos,
                wall.orientation); // find the component of the shifted position that is
                                   // perpendicular to the normalized orientation vector
    Scalar max_dist = sqrt(dot(dist_vec, dist_vec));
    if (wall.inside)
        {
        max_dist += (shape.getCircumsphereDiameter() / Scalar(2.0));
        } // add the circumradius of the particle if the particle must be inside the wall
    else
        {
        // subtract the circumradius of the particle if it must be outside the wall
        max_dist -= (shape.getCircumsphereDiameter() / Scalar(2.0));
        // if the particle radius is larger than the distance between the particle
        // and the container, however, then ALWAYS check verts. this is equivalent
        // to two particle circumspheres overlapping.
        if (max_dist < 0)
            {
            max_dist = 0;
            }
        }

    return wall.inside ? (wall.rsq > max_dist * max_dist) : (wall.rsq < max_dist * max_dist);
    }

// Cylindrical Walls and Convex Polyhedra
DEVICE inline bool test_confined(const CylinderWall& wall,
                                 const ShapeConvexPolyhedron& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    bool accept = true;
    Scalar3 t = vec_to_scalar3(position - box_origin);
    t.x = t.x - wall.origin.x;
    t.y = t.y - wall.origin.y;
    t.z = t.z - wall.origin.z;
    vec3<Scalar> shifted_pos(box.minImage(t));

    vec3<Scalar> dist_vec
        = cross(shifted_pos,
                wall.orientation); // find the component of the shifted position that is
                                   // perpendicular to the normalized orientation vector
    Scalar max_dist = sqrt(dot(dist_vec, dist_vec));
    if (wall.inside)
        {
        max_dist += (shape.getCircumsphereDiameter() / Scalar(2.0));
        } // add the circumradius of the particle if the particle must be inside the wall
    else
        {
        // subtract the circumradius of the particle if it must be outside the wall
        max_dist -= (shape.getCircumsphereDiameter() / Scalar(2.0));
        // if the particle radius is larger than the distance between the particle
        // and the container, however, then ALWAYS check verts. this is equivalent
        // to two particle circumspheres overlapping.
        if (max_dist < 0)
            {
            max_dist = 0;
            }
        }

    bool check_verts = wall.inside ? (wall.rsq <= max_dist * max_dist)
                                   : (wall.rsq >= max_dist * max_dist); // condition to check vertic
    if (check_verts)
        {
        if (wall.inside)
            {
            for (unsigned int v = 0; v < shape.verts.N && accept; v++)
                {
                vec3<Scalar> pos(shape.verts.x[v], shape.verts.y[v], shape.verts.z[v]);
                vec3<Scalar> rotated_pos = rotate(shape.orientation, pos);
                rotated_pos += shifted_pos;

                dist_vec = cross(rotated_pos, wall.orientation);
                max_dist = sqrt(dot(dist_vec, dist_vec));
                accept = wall.inside ? (wall.rsq > max_dist * max_dist)
                                     : (wall.rsq < max_dist * max_dist);
                }
            }
        else
            {
            // build a spheropolyhedron for the wall and the convex polyhedron
            // set the vertices and diameter for the wall.

            vec3<Scalar> r_ab, proj;
            proj = dot(shifted_pos, wall.orientation) * wall.orientation;
            r_ab = shifted_pos - proj;
            unsigned int err = 0;
            assert(shape.verts.sweep_radius == 0);
            ShapeSpheropolyhedron wall_shape(quat<Scalar>(), *wall.verts);
            ShapeSpheropolyhedron part_shape(shape.orientation, shape.verts);
            accept = !test_overlap(r_ab, wall_shape, part_shape, err);
            }
        }
    return accept;
    }

// Plane Walls and Spheres
template<>
DEVICE inline bool test_confined<PlaneWall, ShapeSphere>(const PlaneWall& wall,
                                                         const ShapeSphere& shape,
                                                         const vec3<Scalar>& position,
                                                         const vec3<Scalar>& box_origin,
                                                         const BoxDim& box)
    {
    Scalar3 t = vec_to_scalar3(position - box_origin);
    vec3<Scalar> shifted_pos(box.minImage(t));
    Scalar max_dist
        = dot(wall.normal, shifted_pos) + wall.d; // proj onto unit normal. (signed distance)
    return (max_dist < 0) ? false : 0 < (max_dist - shape.getCircumsphereDiameter() / Scalar(2.0));
    }

// Plane Walls and Convex Polyhedra
DEVICE inline bool test_confined(const PlaneWall& wall,
                                 const ShapeConvexPolyhedron& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    bool accept = true;
    Scalar3 t = vec_to_scalar3(position - box_origin);
    vec3<Scalar> shifted_pos(box.minImage(t));
    Scalar max_dist
        = dot(wall.normal, shifted_pos) + wall.d; // proj onto unit normal. (signed distance)
    accept = Scalar(0.0) < max_dist;              // center is on the correct side of the plane.
    if (accept
        && (max_dist <= shape.getCircumsphereDiameter()
                            / Scalar(2.0))) // should this be <= for consistency? should never
                                            // matter... it was previously just <
        {
        for (unsigned int v = 0; v < shape.verts.N && accept; v++)
            {
            vec3<Scalar> pos(shape.verts.x[v], shape.verts.y[v], shape.verts.z[v]);
            vec3<Scalar> rotated_pos = rotate(shape.orientation, pos);
            rotated_pos += shifted_pos;
            max_dist = dot(wall.normal, rotated_pos)
                       + wall.d;             // proj onto unit normal. (signed distance)
            accept = Scalar(0.0) < max_dist; // vert is on the correct side of the plane.
            }
        }
    return accept;
    }

// Plane Walls and Convex Spheropolyhedra
DEVICE inline bool test_confined(const PlaneWall& wall,
                                 const ShapeSpheropolyhedron& shape,
                                 const vec3<Scalar>& position,
                                 const vec3<Scalar>& box_origin,
                                 const BoxDim& box)
    {
    bool accept = true;
    Scalar3 t = vec_to_scalar3(position - box_origin);
    vec3<Scalar> shifted_pos(box.minImage(t));
    Scalar max_dist
        = dot(wall.normal, shifted_pos) + wall.d; // proj onto unit normal. (signed distance)
    accept = Scalar(0.0) < max_dist;              // center is on the correct side of the plane.
    if (accept && (max_dist <= shape.getCircumsphereDiameter() / Scalar(2.0)))
        {
        if (shape.verts.N)
            {
            for (unsigned int v = 0; v < shape.verts.N && accept; v++)
                {
                vec3<Scalar> pos(shape.verts.x[v], shape.verts.y[v], shape.verts.z[v]);
                vec3<Scalar> rotated_pos = rotate(shape.orientation, pos);
                rotated_pos += shifted_pos;
                max_dist = dot(wall.normal, rotated_pos)
                           + wall.d; // proj onto unit normal. (signed distance)
                accept = shape.verts.sweep_radius
                         < max_dist; // vert is on the correct side of the plane.
                }
            }
        // Pure sphere
        else
            {
            accept = shape.verts.sweep_radius < max_dist;
            }
        }
    return accept;
    }

template<class Shape> class ExternalFieldWall : public ExternalFieldMono<Shape>
    {
    using Compute::m_pdata;

    public:
    ExternalFieldWall(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ExternalFieldMono<Shape>(sysdef), m_mc(mc)
        {
        m_box = m_pdata->getGlobalBox();
        //! scale the container walls every time the box changes
        m_pdata->getBoxChangeSignal()
            .template connect<ExternalFieldWall<Shape>, &ExternalFieldWall<Shape>::scaleWalls>(
                this);
        }
    ~ExternalFieldWall()
        {
        m_pdata->getBoxChangeSignal()
            .template disconnect<ExternalFieldWall<Shape>, &ExternalFieldWall<Shape>::scaleWalls>(
                this);
        }

    double energydiff(const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        const BoxDim& box = this->m_pdata->getGlobalBox();
        vec3<Scalar> origin(m_pdata->getOrigin());

        for (size_t i = 0; i < m_Spheres.size(); i++)
            {
            if (!test_confined(m_Spheres[i], shape_new, position_new, origin, box))
                {
                return INFINITY;
                }
            }

        for (size_t i = 0; i < m_Cylinders.size(); i++)
            {
            set_cylinder_wall_verts(m_Cylinders[i], shape_new);
            if (!test_confined(m_Cylinders[i], shape_new, position_new, origin, box))
                {
                return INFINITY;
                }
            }

        for (size_t i = 0; i < m_Planes.size(); i++)
            {
            if (!test_confined(m_Planes[i], shape_new, position_new, origin, box))
                {
                return INFINITY;
                }
            }

        return double(0.0);
        }

    Scalar calculateBoltzmannWeight(uint64_t timestep)
        {
        unsigned int numOverlaps = countOverlaps(timestep, false);
        if (numOverlaps > 0)
            {
            return Scalar(0.0);
            }
        else
            {
            return Scalar(1.0);
            }
        }

    double calculateDeltaE(const Scalar4* const position_old,
                           const Scalar4* const orientation_old,
                           const BoxDim* const box_old)
        {
        unsigned int numOverlaps = countOverlaps(0, false);
        if (numOverlaps > 0)
            {
            return INFINITY;
            }
        else
            {
            return double(0.0);
            }
        }

    // assumes cubic box
    void scaleWalls()
        {
        BoxDim newBox = m_pdata->getGlobalBox();
        Scalar newVol = newBox.getVolume();
        Scalar oldVol = m_box.getVolume();
        double alpha = pow((newVol / oldVol), Scalar(1.0 / 3.0));
        m_Volume *= (newVol / oldVol);

        for (size_t i = 0; i < m_Spheres.size(); i++)
            {
            m_Spheres[i].scale(alpha);
            }

        for (size_t i = 0; i < m_Cylinders.size(); i++)
            {
            m_Cylinders[i].scale(alpha);
            }

        for (size_t i = 0; i < m_Planes.size(); i++)
            {
            m_Planes[i].scale(alpha);
            }

        m_box = newBox;
        }

    std::tuple<Scalar, vec3<Scalar>, bool> GetSphereWallParameters(size_t index)
        {
        SphereWall w = GetSphereWall(index);
        return std::make_tuple(w.rsq, w.origin, w.inside);
        }

    pybind11::tuple GetSphereWallParametersPy(size_t index)
        {
        Scalar rsq;
        vec3<Scalar> origin;
        bool inside;
        pybind11::list pyorigin;
        std::tuple<Scalar, vec3<Scalar>, bool> t = GetSphereWallParameters(index);
        std::tie(rsq, origin, inside) = t;
        pyorigin.append(pybind11::cast(origin.x));
        pyorigin.append(pybind11::cast(origin.y));
        pyorigin.append(pybind11::cast(origin.z));
        return pybind11::make_tuple(rsq, pyorigin, inside);
        }

    std::tuple<Scalar, vec3<Scalar>, vec3<Scalar>, bool> GetCylinderWallParameters(size_t index)
        {
        CylinderWall w = GetCylinderWall(index);
        return std::make_tuple(w.rsq, w.origin, w.orientation, w.inside);
        }

    pybind11::tuple GetCylinderWallParametersPy(size_t index)
        {
        Scalar rsq;
        vec3<Scalar> origin;
        vec3<Scalar> orientation;
        bool inside;
        pybind11::list pyorigin;
        pybind11::list pyorientation;
        std::tuple<Scalar, vec3<Scalar>, vec3<Scalar>, bool> t = GetCylinderWallParameters(index);
        std::tie(rsq, origin, orientation, inside) = t;
        pyorigin.append(pybind11::cast(origin.x));
        pyorigin.append(pybind11::cast(origin.y));
        pyorigin.append(pybind11::cast(origin.z));
        pyorientation.append(pybind11::cast(orientation.x));
        pyorientation.append(pybind11::cast(orientation.y));
        pyorientation.append(pybind11::cast(orientation.z));
        return pybind11::make_tuple(rsq, pyorigin, pyorientation, inside);
        }

    std::tuple<vec3<Scalar>, vec3<Scalar>> GetPlaneWallParameters(size_t index)
        {
        PlaneWall w = GetPlaneWall(index);
        return std::make_tuple(w.normal, w.origin);
        }

    pybind11::tuple GetPlaneWallParametersPy(size_t index)
        {
        vec3<Scalar> normal;
        vec3<Scalar> origin;
        pybind11::list pynormal;
        pybind11::list pyorigin;
        std::tuple<vec3<Scalar>, vec3<Scalar>> t = GetPlaneWallParameters(index);
        std::tie(normal, origin) = t;
        pynormal.append(pybind11::cast(normal.x));
        pynormal.append(pybind11::cast(normal.y));
        pynormal.append(pybind11::cast(normal.z));
        pyorigin.append(pybind11::cast(origin.x));
        pyorigin.append(pybind11::cast(origin.y));
        pyorigin.append(pybind11::cast(origin.z));
        return pybind11::make_tuple(pynormal, pyorigin);
        }

    const std::vector<SphereWall>& GetSphereWalls()
        {
        return m_Spheres;
        }

    const SphereWall& GetSphereWall(size_t index)
        {
        if (index >= m_Spheres.size())
            throw std::runtime_error("Out of bounds of sphere walls.");
        return m_Spheres[index];
        }

    const std::vector<CylinderWall>& GetCylinderWalls()
        {
        return m_Cylinders;
        }

    const CylinderWall& GetCylinderWall(size_t index)
        {
        if (index >= m_Cylinders.size())
            throw std::runtime_error("Out of bounds of cylinder walls.");
        return m_Cylinders[index];
        }

    const std::vector<PlaneWall>& GetPlaneWalls()
        {
        return m_Planes;
        }

    const PlaneWall& GetPlaneWall(size_t index)
        {
        if (index >= m_Planes.size())
            throw std::runtime_error("Out of bounds of plane walls.");
        return m_Planes[index];
        }

    void SetSphereWallParameter(size_t index, const SphereWall& wall)
        {
        if (index >= m_Spheres.size())
            throw std::runtime_error("Out of bounds of sphere walls.");
        m_Spheres[index] = wall;
        }

    void SetCylinderWallParameter(size_t index, const CylinderWall& wall)
        {
        if (index >= m_Cylinders.size())
            throw std::runtime_error("Out of bounds of cylinder walls.");
        m_Cylinders[index] = wall;
        }

    void SetPlaneWallParameter(size_t index, const PlaneWall& wall)
        {
        if (index >= m_Planes.size())
            throw std::runtime_error("Out of bounds of plane walls.");
        m_Planes[index] = wall;
        }

    void SetSphereWalls(const std::vector<SphereWall>& Spheres)
        {
        m_Spheres = Spheres;
        }

    void SetCylinderWalls(const std::vector<CylinderWall>& Cylinders)
        {
        m_Cylinders = Cylinders;
        }

    void SetPlaneWalls(const std::vector<PlaneWall>& Planes)
        {
        m_Planes = Planes;
        }

    void AddSphereWall(const SphereWall& wall)
        {
        m_Spheres.push_back(wall);
        }

    void AddCylinderWall(const CylinderWall& wall)
        {
        m_Cylinders.push_back(wall);
        }

    void AddPlaneWall(const PlaneWall& wall)
        {
        m_Planes.push_back(wall);
        }

    // is this messy ...
    void RemoveSphereWall(size_t index)
        {
        m_Spheres.erase(m_Spheres.begin() + index);
        }

    void RemoveCylinderWall(size_t index)
        {
        m_Cylinders.erase(m_Cylinders.begin() + index);
        }

    void RemovePlaneWall(size_t index)
        {
        m_Planes.erase(m_Planes.begin() + index);
        }

    bool wall_overlap(const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        double energy = energydiff(index, position_old, shape_old, position_new, shape_new);
        return (energy == INFINITY);
        }

    unsigned int countOverlaps(uint64_t timestep, bool early_exit = false)
        {
        unsigned int numOverlaps = 0;
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        const std::vector<typename Shape::param_type,
                          hoomd::detail::managed_allocator<typename Shape::param_type>>& params
            = m_mc->getParams();

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            Scalar4 orientation_i = h_orientation.data[i];
            vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
            int typ_i = __scalar_as_int(postype_i.w);
            Shape shape_i(quat<Scalar>(orientation_i), params[typ_i]);

            if (wall_overlap(i, pos_i, shape_i, pos_i, shape_i))
                {
                numOverlaps++;
                }

            if (early_exit && numOverlaps > 0)
                {
                numOverlaps = 1;
                break;
                }
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &numOverlaps,
                          1,
                          MPI_UNSIGNED,
                          MPI_SUM,
                          this->m_exec_conf->getMPICommunicator());
            if (early_exit && numOverlaps > 1)
                numOverlaps = 1;
            }
#endif

        return numOverlaps;
        }

    void setVolume(Scalar volume)
        {
        m_Volume = volume;
        }

    Scalar getVolume()
        {
        return m_Volume;
        }

    size_t getNumSphereWalls()
        {
        return m_Spheres.size();
        }
    size_t getNumCylinderWalls()
        {
        return m_Cylinders.size();
        }
    size_t getNumPlaneWalls()
        {
        return m_Planes.size();
        }

    bool hasVolume()
        {
        return true;
        }

    Scalar GetCurrBoxLx()
        {
        return m_box.getL().x;
        }
    Scalar GetCurrBoxLy()
        {
        return m_box.getL().y;
        }
    Scalar GetCurrBoxLz()
        {
        return m_box.getL().z;
        }
    Scalar GetCurrBoxTiltFactorXY()
        {
        return m_box.getTiltFactorXY();
        }
    Scalar GetCurrBoxTiltFactorXZ()
        {
        return m_box.getTiltFactorXZ();
        }
    Scalar GetCurrBoxTiltFactorYZ()
        {
        return m_box.getTiltFactorYZ();
        }

    void SetCurrBox(Scalar Lx, Scalar Ly, Scalar Lz, Scalar xy, Scalar xz, Scalar yz)
        {
        m_box.setL(make_scalar3(Lx, Ly, Lz));
        m_box.setTiltFactors(xy, xz, yz);
        }

    protected:
    void set_cylinder_wall_verts(CylinderWall& wall, const Shape& shape)
        {
        vec3<Scalar> v0;
        v0 = Scalar(shape.getCircumsphereDiameter()) * wall.orientation;
        wall.verts->x[0] = OverlapReal(-v0.x);
        wall.verts->y[0] = OverlapReal(-v0.y);
        wall.verts->z[0] = OverlapReal(-v0.z);

        wall.verts->x[1] = OverlapReal(v0.x);
        wall.verts->y[1] = OverlapReal(v0.y);
        wall.verts->z[1] = OverlapReal(v0.z);
        wall.verts->diameter
            = OverlapReal(2.0 * (shape.getCircumsphereDiameter() + wall.verts->sweep_radius));
        }
    std::string getSphWallParamName(size_t i)
        {
        std::stringstream ss;
        std::string snum;
        ss << i;
        ss >> snum;
        return "hpmc_wall_sph_rsq-" + snum;
        }
    std::string getCylWallParamName(size_t i)
        {
        std::stringstream ss;
        std::string snum;
        ss << i;
        ss >> snum;
        return "hpmc_wall_cyl_rsq-" + snum;
        }

    protected:
    std::vector<SphereWall> m_Spheres;
    std::vector<CylinderWall> m_Cylinders;
    std::vector<PlaneWall> m_Planes;
    Scalar m_Volume;

    private:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< integrator
    BoxDim m_box;                                    //!< the current box
    };

namespace detail
    {
template<class Shape> void export_ExternalFieldWall(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ExternalFieldWall<Shape>,
                     ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalFieldWall<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def("SetSphereWallParameter", &ExternalFieldWall<Shape>::SetSphereWallParameter)
        .def("SetCylinderWallParameter", &ExternalFieldWall<Shape>::SetCylinderWallParameter)
        .def("SetPlaneWallParameter", &ExternalFieldWall<Shape>::SetPlaneWallParameter)
        .def("AddSphereWall", &ExternalFieldWall<Shape>::AddSphereWall)
        .def("AddCylinderWall", &ExternalFieldWall<Shape>::AddCylinderWall)
        .def("AddPlaneWall", &ExternalFieldWall<Shape>::AddPlaneWall)
        .def("RemoveSphereWall", &ExternalFieldWall<Shape>::RemoveSphereWall)
        .def("RemoveCylinderWall", &ExternalFieldWall<Shape>::RemoveCylinderWall)
        .def("RemovePlaneWall", &ExternalFieldWall<Shape>::RemovePlaneWall)
        .def("countOverlaps", &ExternalFieldWall<Shape>::countOverlaps)
        .def("setVolume", &ExternalFieldWall<Shape>::setVolume)
        .def("getVolume", &ExternalFieldWall<Shape>::getVolume)
        .def("getNumSphereWalls", &ExternalFieldWall<Shape>::getNumSphereWalls)
        .def("getNumCylinderWalls", &ExternalFieldWall<Shape>::getNumCylinderWalls)
        .def("getNumPlaneWalls", &ExternalFieldWall<Shape>::getNumPlaneWalls)
        .def("GetSphereWallParametersPy", &ExternalFieldWall<Shape>::GetSphereWallParametersPy)
        .def("GetCylinderWallParametersPy", &ExternalFieldWall<Shape>::GetCylinderWallParametersPy)
        .def("GetPlaneWallParametersPy", &ExternalFieldWall<Shape>::GetPlaneWallParametersPy)
        .def("GetCurrBoxLx", &ExternalFieldWall<Shape>::GetCurrBoxLx)
        .def("GetCurrBoxLy", &ExternalFieldWall<Shape>::GetCurrBoxLy)
        .def("GetCurrBoxLz", &ExternalFieldWall<Shape>::GetCurrBoxLz)
        .def("GetCurrBoxTiltFactorXY", &ExternalFieldWall<Shape>::GetCurrBoxTiltFactorXY)
        .def("GetCurrBoxTiltFactorXZ", &ExternalFieldWall<Shape>::GetCurrBoxTiltFactorXZ)
        .def("GetCurrBoxTiltFactorYZ", &ExternalFieldWall<Shape>::GetCurrBoxTiltFactorYZ)
        .def("SetCurrBox", &ExternalFieldWall<Shape>::SetCurrBox);
    }

    } // end namespace detail
    } // namespace hpmc
    } // end namespace hoomd
#undef DEVICE
#endif // inclusion guard
