// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
#include <pybind11/stl_bind.h>
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
        : rsq(0), inside(true), origin(0, 0, 0), verts(new detail::PolyhedronVertices(1, false))
        {
        }
    SphereWall(Scalar r, vec3<Scalar> origin_, bool inside_ = true)
        : rsq(r * r), inside(inside_), origin(origin_),
          verts(new detail::PolyhedronVertices(1, false))
        {
        verts->N = 0; // case for sphere (can be 0 or 1)
        verts->diameter = ShortReal(r + r);
        verts->sweep_radius = ShortReal(r);
        verts->ignore = 0;
        }
#ifndef __HIPCC__
    SphereWall(Scalar r, pybind11::tuple origin, bool inside = true)
        : SphereWall(r,
                     vec3<Scalar>(origin[0].cast<Scalar>(),
                                  origin[1].cast<Scalar>(),
                                  origin[2].cast<Scalar>()),
                     inside)
        {
        }

    pybind11::tuple getOrigin()
        {
        return pybind11::make_tuple(origin.x, origin.y, origin.z);
        }
#endif
    SphereWall(const SphereWall& src)
        : rsq(src.rsq), inside(src.inside), origin(src.origin),
          verts(new detail::PolyhedronVertices(*src.verts))
        {
        }

    Scalar getRadius()
        {
        return sqrt(rsq);
        }

    bool getInside()
        {
        return inside;
        }

    bool getOpen()
        {
        return open;
        }

    Scalar rsq;
    bool inside;
    vec3<Scalar> origin;
    std::shared_ptr<detail::PolyhedronVertices> verts;
    bool open;
    };

struct CylinderWall
    {
    CylinderWall()
        : rsq(0), inside(true), origin(0, 0, 0), orientation(0, 0, 1),
          verts(new detail::PolyhedronVertices)
        {
        }
    CylinderWall(Scalar r, vec3<Scalar> origin_, vec3<Scalar> axis_, bool inside_ = true)
        : rsq(r * r), inside(inside_), origin(origin_), orientation(axis_),
          verts(new detail::PolyhedronVertices(2, false))
        {
        // normalize the orientation vector
        Scalar len = sqrt(dot(orientation, orientation));
        orientation /= len;

        // set the position of the vertices and the diameter later
        verts->N = 2;
        verts->sweep_radius = ShortReal(r);
        verts->ignore = 0;
        }
#ifndef __HIPCC__
    CylinderWall(Scalar r, pybind11::tuple origin, pybind11::tuple axis, bool inside = true)
        : CylinderWall(
              r,
              vec3<Scalar>(origin[0].cast<Scalar>(),
                           origin[1].cast<Scalar>(),
                           origin[2].cast<Scalar>()),
              vec3<Scalar>(axis[0].cast<Scalar>(), axis[1].cast<Scalar>(), axis[2].cast<Scalar>()),
              inside)
        {
        }

    pybind11::tuple getOrigin()
        {
        return pybind11::make_tuple(origin.x, origin.y, origin.z);
        }

    pybind11::tuple getAxis()
        {
        return pybind11::make_tuple(orientation.x, orientation.y, orientation.z);
        }
#endif
    CylinderWall(const CylinderWall& src)
        : rsq(src.rsq), inside(src.inside), origin(src.origin), orientation(src.orientation),
          verts(new detail::PolyhedronVertices(*src.verts))
        {
        }

    Scalar getRadius()
        {
        return sqrt(rsq);
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
    PlaneWall(vec3<Scalar> origin_, vec3<Scalar> normal_) : origin(origin_), normal(normal_)
        {
        Scalar len = sqrt(dot(normal, normal));
        normal /= len;
        d = -dot(normal, origin);
        }
#ifndef __HIPCC__
    PlaneWall(pybind11::tuple origin, pybind11::tuple normal)
        : PlaneWall(vec3<Scalar>(origin[0].cast<Scalar>(),
                                 origin[1].cast<Scalar>(),
                                 origin[2].cast<Scalar>()),
                    vec3<Scalar>(normal[0].cast<Scalar>(),
                                 normal[1].cast<Scalar>(),
                                 normal[2].cast<Scalar>()))
        {
        }

    pybind11::tuple getOrigin()
        {
        return pybind11::make_tuple(origin.x, origin.y, origin.z);
        }

    pybind11::tuple getNormal()
        {
        return pybind11::make_tuple(normal.x, normal.y, normal.z);
        }
#endif

    vec3<Scalar> origin; // we could remove this.
    vec3<Scalar> normal; // unit normal n = (a, b, c)
    Scalar d;            // ax + by + cz + d =  0
    };
    } // namespace hpmc
    } // namespace hoomd

#ifndef __HIPCC__
// This is required to be here before any uses of these vectors to work
// correctly.
PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::SphereWall>);
PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::CylinderWall>);
PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::PlaneWall>);
#endif

namespace hoomd
    {
namespace hpmc
    {
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
    Scalar max_dist = sqrt(rxyz_sq) + (shape.getCircumsphereDiameter() / ShortReal(2.0));
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
        }
    ~ExternalFieldWall() { }

    double energydiff(uint64_t timestep,
                      const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        const BoxDim box = this->m_pdata->getGlobalBox();
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

    //! Evaluate the energy of the force.
    /*! \param box The system box.
        \param type Particle type.
        \param r_i Particle position
        \param q_i Particle orientation.
        \param diameter Particle diameter.
        \param charge Particle charge.
        \returns Energy due to the force
    */
    virtual float energy(const BoxDim& box,
                         unsigned int type,
                         const vec3<Scalar>& r_i,
                         const quat<Scalar>& q_i,
                         Scalar diameter,
                         Scalar charge)
        {
        const std::vector<typename Shape::param_type,
                          hoomd::detail::managed_allocator<typename Shape::param_type>>& params
            = m_mc->getParams();
        Shape shape(q_i, params[type]);
        vec3<Scalar> origin(m_pdata->getOrigin());

        for (size_t i = 0; i < m_Spheres.size(); i++)
            {
            if (!test_confined(m_Spheres[i], shape, r_i, origin, box))
                {
                return INFINITY;
                }
            }

        for (size_t i = 0; i < m_Cylinders.size(); i++)
            {
            set_cylinder_wall_verts(m_Cylinders[i], shape);
            if (!test_confined(m_Cylinders[i], shape, r_i, origin, box))
                {
                return INFINITY;
                }
            }

        for (size_t i = 0; i < m_Planes.size(); i++)
            {
            if (!test_confined(m_Planes[i], shape, r_i, origin, box))
                {
                return INFINITY;
                }
            }

        return double(0.0);
        }

    double calculateDeltaE(uint64_t timestep,
                           const Scalar4* const position_old,
                           const Scalar4* const orientation_old,
                           const BoxDim& box_old,
                           const Scalar3& origin_old)
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

    std::vector<SphereWall>& GetSphereWalls()
        {
        return m_Spheres;
        }

    std::vector<CylinderWall>& GetCylinderWalls()
        {
        return m_Cylinders;
        }

    std::vector<PlaneWall>& GetPlaneWalls()
        {
        return m_Planes;
        }

    bool wall_overlap(uint64_t timestep,
                      const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        double energy
            = energydiff(timestep, index, position_old, shape_old, position_new, shape_new);
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

            if (wall_overlap(timestep, i, pos_i, shape_i, pos_i, shape_i))
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

    protected:
    void set_cylinder_wall_verts(CylinderWall& wall, const Shape& shape)
        {
        vec3<Scalar> v0;
        v0 = Scalar(shape.getCircumsphereDiameter()) * wall.orientation;
        wall.verts->x[0] = ShortReal(-v0.x);
        wall.verts->y[0] = ShortReal(-v0.y);
        wall.verts->z[0] = ShortReal(-v0.z);

        wall.verts->x[1] = ShortReal(v0.x);
        wall.verts->y[1] = ShortReal(v0.y);
        wall.verts->z[1] = ShortReal(v0.z);
        wall.verts->diameter
            = ShortReal(2.0 * (shape.getCircumsphereDiameter() + wall.verts->sweep_radius));
        }

    protected:
    std::vector<SphereWall> m_Spheres;
    std::vector<CylinderWall> m_Cylinders;
    std::vector<PlaneWall> m_Planes;
    Scalar m_Volume;

    private:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc; //!< integrator
    };
    } // namespace hpmc
    } // namespace hoomd

#ifndef __HIPCC__
namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
template<class Shape> void export_ExternalFieldWall(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ExternalFieldWall<Shape>,
                     ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalFieldWall<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def("numOverlaps", &ExternalFieldWall<Shape>::countOverlaps)
        .def_property_readonly("SphereWalls", &ExternalFieldWall<Shape>::GetSphereWalls)
        .def_property_readonly("CylinderWalls", &ExternalFieldWall<Shape>::GetCylinderWalls)
        .def_property_readonly("PlaneWalls", &ExternalFieldWall<Shape>::GetPlaneWalls);
    }

void export_wall_list(pybind11::module& m);
void export_wall_classes(pybind11::module& m);
    } // end namespace detail
    } // namespace hpmc
    } // end namespace hoomd
#endif
#undef DEVICE
#endif // inclusion guard
