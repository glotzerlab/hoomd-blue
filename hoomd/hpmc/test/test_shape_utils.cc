// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/hpmc/ShapeConvexPolyhedron.h"
#include "hoomd/hpmc/ShapeEllipsoid.h"
#include "hoomd/hpmc/ShapeSphere.h"
#include "hoomd/hpmc/ShapeUtils.h"

#include "hoomd/extern/quickhull/QuickHull.hpp"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

UP_TEST(convex_polyhedron_mass_properties)
    {
    vector<vec3<OverlapReal>> vlist;
    vlist.push_back(vec3<OverlapReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<OverlapReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<OverlapReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<OverlapReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<OverlapReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<OverlapReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<OverlapReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<OverlapReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    Scalar V = 1.0;
    Scalar S = 6.0;
    Scalar Q = 36.0 * M_PI * V * V / (S * S * S);
    Scalar I = 1 / 6;
    Scalar detI = I * I * I;

    MassProperties<ShapeConvexPolyhedron> mp(verts, true);

    UP_ASSERT(mp.getNumFaces() == 6);
    MY_CHECK_CLOSE(mp.getVolume(), V, tol);
    MY_CHECK_CLOSE(mp.getSurfaceArea(), S, tol);
    MY_CHECK_CLOSE(mp.getIsoperimetricQuotient(), Q, tol);

    vec3<Scalar> com = mp.getCenterOfMass();
    MY_CHECK_CLOSE(com.x, 0.0, tol);
    MY_CHECK_CLOSE(com.y, 0.0, tol);
    MY_CHECK_CLOSE(com.z, 0.0, tol);

    MY_CHECK_CLOSE(mp.getInertiaTensor(0), I, tol);
    MY_CHECK_CLOSE(mp.getInertiaTensor(1), I, tol);
    MY_CHECK_CLOSE(mp.getInertiaTensor(2), I, tol);
    MY_CHECK_CLOSE(mp.getInertiaTensor(3), 0, tol);
    MY_CHECK_CLOSE(mp.getInertiaTensor(4), 0, tol);
    MY_CHECK_CLOSE(mp.getInertiaTensor(5), 0, tol);

    MY_CHECK_CLOSE(mp.getDetInertiaTensor(), detI, tol);

    vec3<Scalar> new_com(0.1, -0.2, 0.5);
    for (unsigned int i = 0; i < vlist.size(); i++)
        {
        vlist[i] += new_com;
        }
    verts.updateParam(vlist);
    com = mp.getCenterOfMass();
    MY_CHECK_CLOSE(com.x, new_com.x, tol);
    MY_CHECK_CLOSE(com.y, new_com.y, tol);
    MY_CHECK_CLOSE(com.z, new_com.z, tol);
    }

UP_TEST(convex_spheropolyhedron_mass_properties)
    {
    //
    }

UP_TEST(ellipsoid_mass_properties)
    {
    // build an Ellipsoid
    EllipsoidParams axes;
    axes.x = 0.5;
    axes.y = 0.5;
    axes.z = 1;

    axes.x = 1;
    axes.y = 1.3;
    axes.z = 1;
    }
