// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <string.h>


#include "hoomd/Compute.h"
#include "hoomd/extern/saruprng.h" // not sure if we need this for the accept method
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


namespace hpmc{
// NOTE: I am only exporting 3d shapes for now beacuse I think the 2d ones need some tweaking (how to do this generally?)

SphereWall make_sphere_wall(Scalar r, boost::python::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = boost::python::extract<Scalar>(origin[0]);
    orig.y = boost::python::extract<Scalar>(origin[1]);
    orig.z = boost::python::extract<Scalar>(origin[2]);
    return SphereWall(r, orig, inside);
    }

CylinderWall make_cylinder_wall(Scalar r, boost::python::list origin, boost::python::list orientation, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = boost::python::extract<Scalar>(origin[0]);
    orig.y = boost::python::extract<Scalar>(origin[1]);
    orig.z = boost::python::extract<Scalar>(origin[2]);
    vec3<Scalar> orient;
    orient.x = boost::python::extract<Scalar>(orientation[0]);
    orient.y = boost::python::extract<Scalar>(orientation[1]);
    orient.z = boost::python::extract<Scalar>(orientation[2]);
    return CylinderWall(r, orig, orient, inside);
    }

PlaneWall make_plane_wall(boost::python::list norm, boost::python::list origin, bool inside)
    {
    vec3<Scalar> orig;
    orig.x = boost::python::extract<Scalar>(origin[0]);
    orig.y = boost::python::extract<Scalar>(origin[1]);
    orig.z = boost::python::extract<Scalar>(origin[2]);
    vec3<Scalar> normal;
    normal.x = boost::python::extract<Scalar>(norm[0]);
    normal.y = boost::python::extract<Scalar>(norm[1]);
    normal.z = boost::python::extract<Scalar>(norm[2]);
    return PlaneWall(normal, orig, inside);
    }

void export_walls()
    {
    // export wall structs.
    class_<SphereWall, std::shared_ptr<SphereWall> >("sphere_wall_params");
    class_<CylinderWall, std::shared_ptr<CylinderWall> >("cylinder_wall_params");
    class_<PlaneWall, std::shared_ptr<PlaneWall> >("plane_wall_params", no_init);

    // export helper functions.
    def("make_sphere_wall", &make_sphere_wall);
    def("make_cylinder_wall", &make_cylinder_wall);
    def("make_plane_wall", &make_plane_wall);
    }


//! Export the external field classes to python
void export_external_fields()
    {
    // export wall fields
    export_walls();
    }

} // namespace
