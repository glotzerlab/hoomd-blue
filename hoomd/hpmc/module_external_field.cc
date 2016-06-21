// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <boost/python.hpp>
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

void export_lattice()
    {
    export_LatticeField<ShapePolyhedron>("ExternalFieldLatticePolyhedron");

    export_LatticeField<ShapeConvexPolyhedron<8> >("ExternalFieldLatticeConvexPolyhedron8");
    export_LatticeField<ShapeConvexPolyhedron<16> >("ExternalFieldLatticeConvexPolyhedron16");
    export_LatticeField<ShapeConvexPolyhedron<32> >("ExternalFieldLatticeConvexPolyhedron32");
    export_LatticeField<ShapeConvexPolyhedron<64> >("ExternalFieldLatticeConvexPolyhedron64");
    export_LatticeField<ShapeConvexPolyhedron<128> >("ExternalFieldLatticeConvexPolyhedron128");

    export_LatticeField<ShapeSpheropolyhedron<8> >("ExternalFieldLatticeSpheropolyhedron8");
    export_LatticeField<ShapeSpheropolyhedron<16> >("ExternalFieldLatticeSpheropolyhedron16");
    export_LatticeField<ShapeSpheropolyhedron<32> >("ExternalFieldLatticeSpheropolyhedron32");
    export_LatticeField<ShapeSpheropolyhedron<64> >("ExternalFieldLatticeSpheropolyhedron64");
    export_LatticeField<ShapeSpheropolyhedron<128> >("ExternalFieldLatticeSpheropolyhedron128");

    export_LatticeField<ShapeSpheropolygon>("ExternalFieldLatticeSpheropolygon");
    export_LatticeField<ShapeEllipsoid>("ExternalFieldLatticeEllipsoid");
    export_LatticeField<ShapeSphinx>("ExternalFieldLatticeSphinx");
    export_LatticeField<ShapeUnion<ShapeSphere> >("ExternalFieldLatticeUnionSphere");
    }

void export_walls()
    {
    // export wall structs.
    class_<SphereWall, boost::shared_ptr<SphereWall> >("sphere_wall_params");
    class_<CylinderWall, boost::shared_ptr<CylinderWall> >("cylinder_wall_params");
    class_<PlaneWall, boost::shared_ptr<PlaneWall> >("plane_wall_params", no_init);

    // export helper functions.
    def("make_sphere_wall", &make_sphere_wall);
    def("make_cylinder_wall", &make_cylinder_wall);
    def("make_plane_wall", &make_plane_wall);

    // export wall class
    export_ExternalFieldWall<ShapeConvexPolyhedron<8> >("WallConvexPolyhedron8");
    export_ExternalFieldWall<ShapeConvexPolyhedron<16> >("WallConvexPolyhedron16");
    export_ExternalFieldWall<ShapeConvexPolyhedron<32> >("WallConvexPolyhedron32");
    export_ExternalFieldWall<ShapeConvexPolyhedron<64> >("WallConvexPolyhedron64");
    export_ExternalFieldWall<ShapeConvexPolyhedron<128> >("WallConvexPolyhedron128");

    //export the updater
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<8> >("UpdaterExternalFieldWallConvexPolyhedron8");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<16> >("UpdaterExternalFieldWallConvexPolyhedron16");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<32> >("UpdaterExternalFieldWallConvexPolyhedron32");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<64> >("UpdaterExternalFieldWallConvexPolyhedron64");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<128> >("UpdaterExternalFieldWallConvexPolyhedron128");

    }


//! Export the external field classes to python
void export_external_fields()
    {
    // export the external field interfaces.
    export_ExternalFieldInterface<ShapePolyhedron>("ExternalFieldPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<8> >("ExternalFieldConvexPolyhedron8");
    export_ExternalFieldInterface<ShapeConvexPolyhedron<16> >("ExternalFieldConvexPolyhedron16");
    export_ExternalFieldInterface<ShapeConvexPolyhedron<32> >("ExternalFieldConvexPolyhedron32");
    export_ExternalFieldInterface<ShapeConvexPolyhedron<64> >("ExternalFieldConvexPolyhedron64");
    export_ExternalFieldInterface<ShapeConvexPolyhedron<128> >("ExternalFieldConvexPolyhedron128");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<8> >("ExternalFieldSpheropolyhedron8");
    export_ExternalFieldInterface<ShapeSpheropolyhedron<16> >("ExternalFieldSpheropolyhedron16");
    export_ExternalFieldInterface<ShapeSpheropolyhedron<32> >("ExternalFieldSpheropolyhedron32");
    export_ExternalFieldInterface<ShapeSpheropolyhedron<64> >("ExternalFieldSpheropolyhedron64");
    export_ExternalFieldInterface<ShapeSpheropolyhedron<128> >("ExternalFieldSpheropolyhedron128");

    export_ExternalFieldInterface<ShapeSpheropolygon>("ExternalFieldSpheropolygon");
    export_ExternalFieldInterface<ShapeEllipsoid>("ExternalFieldEllipsoid");
    export_ExternalFieldInterface<ShapeSphinx>("ExternalFieldSphinx");
    export_ExternalFieldInterface<ShapeUnion<ShapeSphere> >("ExternalFieldUnionSphere");

    // export wall fields
    export_walls();
    //export lattice fields
    export_lattice();

    //export composite fields
    export_ExternalFieldComposite<ShapePolyhedron>("ExternalFieldCompositePolyhedron");

    export_ExternalFieldComposite<ShapeConvexPolyhedron<8> >("ExternalFieldCompositeConvexPolyhedron8");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<16> >("ExternalFieldCompositeConvexPolyhedron16");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<32> >("ExternalFieldCompositeConvexPolyhedron32");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<64> >("ExternalFieldCompositeConvexPolyhedron64");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<128> >("ExternalFieldCompositeConvexPolyhedron128");

    export_ExternalFieldComposite<ShapeSpheropolyhedron<8> >("ExternalFieldCompositeSpheropolyhedron8");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<16> >("ExternalFieldCompositeSpheropolyhedron16");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<32> >("ExternalFieldCompositeSpheropolyhedron32");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<64> >("ExternalFieldCompositeSpheropolyhedron64");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<128> >("ExternalFieldCompositeSpheropolyhedron128");

    export_ExternalFieldComposite<ShapeSpheropolygon>("ExternalFieldCompositeSpheropolygon");
    export_ExternalFieldComposite<ShapeEllipsoid>("ExternalFieldCompositeEllipsoid");
    export_ExternalFieldComposite<ShapeSphinx>("ExternalFieldCompositeSphinx");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere> >("ExternalFieldCompositeUnionSphere");

    //export drift remover
    export_RemoveDriftUpdater<ShapePolyhedron>("RemoveDriftUpdaterPolyhedron");

    export_RemoveDriftUpdater<ShapeConvexPolyhedron<8> >("RemoveDriftUpdaterConvexPolyhedron8");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<16> >("RemoveDriftUpdaterConvexPolyhedron16");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<32> >("RemoveDriftUpdaterConvexPolyhedron32");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<64> >("RemoveDriftUpdaterConvexPolyhedron64");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<128> >("RemoveDriftUpdaterConvexPolyhedron128");

    export_RemoveDriftUpdater<ShapeSpheropolyhedron<8> >("RemoveDriftUpdaterSpheropolyhedron8");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<16> >("RemoveDriftUpdaterSpheropolyhedron16");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<32> >("RemoveDriftUpdaterSpheropolyhedron32");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<64> >("RemoveDriftUpdaterSpheropolyhedron64");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<128> >("RemoveDriftUpdaterSpheropolyhedron128");

    export_RemoveDriftUpdater<ShapeSpheropolygon>("RemoveDriftUpdaterSpheropolygon");
    export_RemoveDriftUpdater<ShapeEllipsoid>("RemoveDriftUpdaterEllipsoid");
    export_RemoveDriftUpdater<ShapeSphinx>("RemoveDriftUpdaterSphinx");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere> >("RemoveDriftUpdaterUnionSphere");
    }

} // namespace
