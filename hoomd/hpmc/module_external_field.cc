// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <string.h>

#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"

#include "HPMCCounters.h" // do we need this to keep track of the statistics?

#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeEllipsoid.h"
#include "ShapePolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeSphere.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterExternalFieldWall.h"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
// NOTE: I am only exporting 3d shapes for now because I think the 2d ones need some tweaking (how
// to do this generally?)

SphereWall make_sphere_wall(Scalar r, pybind11::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = pybind11::cast<Scalar>(origin[0]);
    orig.y = pybind11::cast<Scalar>(origin[1]);
    orig.z = pybind11::cast<Scalar>(origin[2]);
    return SphereWall(r, orig, inside);
    }

CylinderWall
make_cylinder_wall(Scalar r, pybind11::list origin, pybind11::list orientation, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = pybind11::cast<Scalar>(origin[0]);
    orig.y = pybind11::cast<Scalar>(origin[1]);
    orig.z = pybind11::cast<Scalar>(origin[2]);
    vec3<Scalar> orient;
    orient.x = pybind11::cast<Scalar>(orientation[0]);
    orient.y = pybind11::cast<Scalar>(orientation[1]);
    orient.z = pybind11::cast<Scalar>(orientation[2]);
    return CylinderWall(r, orig, orient, inside);
    }

PlaneWall make_plane_wall(pybind11::list norm, pybind11::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = pybind11::cast<Scalar>(origin[0]);
    orig.y = pybind11::cast<Scalar>(origin[1]);
    orig.z = pybind11::cast<Scalar>(origin[2]);
    vec3<Scalar> normal;
    normal.x = pybind11::cast<Scalar>(norm[0]);
    normal.y = pybind11::cast<Scalar>(norm[1]);
    normal.z = pybind11::cast<Scalar>(norm[2]);
    return PlaneWall(normal, orig, inside);
    }

void export_walls(pybind11::module& m)
    {
    // export wall structs.
    pybind11::class_<SphereWall, std::shared_ptr<SphereWall>>(m, "sphere_wall_params");
    pybind11::class_<CylinderWall, std::shared_ptr<CylinderWall>>(m, "cylinder_wall_params");
    pybind11::class_<PlaneWall, std::shared_ptr<PlaneWall>>(m, "plane_wall_params");

    // export helper functions.
    m.def("make_sphere_wall", &make_sphere_wall);
    m.def("make_cylinder_wall", &make_cylinder_wall);
    m.def("make_plane_wall", &make_plane_wall);
    }

//! Export the external field classes to python
void export_external_fields(pybind11::module& m)
    {
    // export wall fields
    export_walls(m);
    }
    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
