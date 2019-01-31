// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <string.h>


#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"

#include "HPMCCounters.h"   // do we need this to keep track of the statistics?

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"

namespace py = pybind11;

namespace hpmc{
// NOTE: I am only exporting 3d shapes for now because I think the 2d ones need some tweaking (how to do this generally?)

SphereWall make_sphere_wall(Scalar r, py::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = py::cast<Scalar>(origin[0]);
    orig.y = py::cast<Scalar>(origin[1]);
    orig.z = py::cast<Scalar>(origin[2]);
    return SphereWall(r, orig, inside);
    }

CylinderWall make_cylinder_wall(Scalar r, py::list origin, py::list orientation, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = py::cast<Scalar>(origin[0]);
    orig.y = py::cast<Scalar>(origin[1]);
    orig.z = py::cast<Scalar>(origin[2]);
    vec3<Scalar> orient;
    orient.x = py::cast<Scalar>(orientation[0]);
    orient.y = py::cast<Scalar>(orientation[1]);
    orient.z = py::cast<Scalar>(orientation[2]);
    return CylinderWall(r, orig, orient, inside);
    }

PlaneWall make_plane_wall(py::list norm, py::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = py::cast<Scalar>(origin[0]);
    orig.y = py::cast<Scalar>(origin[1]);
    orig.z = py::cast<Scalar>(origin[2]);
    vec3<Scalar> normal;
    normal.x = py::cast<Scalar>(norm[0]);
    normal.y = py::cast<Scalar>(norm[1]);
    normal.z = py::cast<Scalar>(norm[2]);
    return PlaneWall(normal, orig, inside);
    }

void export_walls(py::module& m)
    {
    // export wall structs.
   py::class_<SphereWall, std::shared_ptr<SphereWall> >(m, "sphere_wall_params");
   py::class_<CylinderWall, std::shared_ptr<CylinderWall> >(m, "cylinder_wall_params");
   py::class_<PlaneWall, std::shared_ptr<PlaneWall> >(m, "plane_wall_params");

    // export helper functions.
    m.def("make_sphere_wall", &make_sphere_wall);
    m.def("make_cylinder_wall", &make_cylinder_wall);
    m.def("make_plane_wall", &make_plane_wall);
    }


//! Export the external field classes to python
void export_external_fields(py::module& m)
    {
    // export wall fields
    export_walls(m);
    }

} // namespace
