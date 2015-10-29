/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//! Name the unit test module
#define BOOST_TEST_MODULE walldata
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdlib>
#include <vector>

#include "WallData.h"

BOOST_AUTO_TEST_CASE( construction )
    {
    // test each geometry definition structure constructors
    SphereWall defSphere = SphereWall();
    MY_BOOST_CHECK_SMALL(defSphere.r, tol_small);
    MY_BOOST_CHECK_SMALL(defSphere.origin.x, tol_small);
    MY_BOOST_CHECK_SMALL(defSphere.origin.y, tol_small);
    MY_BOOST_CHECK_SMALL(defSphere.origin.z, tol_small);
    BOOST_CHECK(defSphere.inside==true);

    SphereWall Sphere = SphereWall(1.25,make_scalar3(1.0,2.8,39.27),false);
    MY_BOOST_CHECK_CLOSE(Sphere.r, 1.25, tol);
    MY_BOOST_CHECK_CLOSE(Sphere.origin.x, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(Sphere.origin.y, 2.8, tol);
    MY_BOOST_CHECK_CLOSE(Sphere.origin.z, 39.27, tol);
    BOOST_CHECK(Sphere.inside==false);

    CylinderWall defCylinder = CylinderWall();
    MY_BOOST_CHECK_SMALL(defCylinder.r, tol_small);
    MY_BOOST_CHECK_SMALL(defCylinder.origin.x, tol_small);
    MY_BOOST_CHECK_SMALL(defCylinder.origin.y, tol_small);
    MY_BOOST_CHECK_SMALL(defCylinder.origin.z, tol_small);
    MY_BOOST_CHECK_SMALL(defCylinder.axis.x, tol_small);
    MY_BOOST_CHECK_SMALL(defCylinder.axis.y, tol_small);
    MY_BOOST_CHECK_CLOSE(defCylinder.axis.z, 1.0, tol);
    BOOST_CHECK(defCylinder.inside==true);

    CylinderWall Cylinder = CylinderWall(1.25,make_scalar3(584.1,27.6,14.1),make_scalar3(4.2,57.21,0.21),false);
    MY_BOOST_CHECK_CLOSE(Cylinder.r, 1.25, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.origin.x, 584.1, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.origin.y, 27.6, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.origin.z, 14.1, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.axis.x, 4.2, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.axis.y, 57.21, tol);
    MY_BOOST_CHECK_CLOSE(Cylinder.axis.z, 0.21, tol);
    BOOST_CHECK(Cylinder.inside==false);

    PlaneWall defPlane = PlaneWall();
    MY_BOOST_CHECK_SMALL(defPlane.origin.x, tol_small);
    MY_BOOST_CHECK_SMALL(defPlane.origin.y, tol_small);
    MY_BOOST_CHECK_SMALL(defPlane.origin.z, tol_small);
    MY_BOOST_CHECK_SMALL(defPlane.normal.x, tol_small);
    MY_BOOST_CHECK_SMALL(defPlane.normal.y, tol_small);
    MY_BOOST_CHECK_CLOSE(defPlane.normal.z, 1.0, tol);

    PlaneWall Plane = PlaneWall(make_scalar3(5.214,1.32,8.143),make_scalar3(3.62,2.151,8.1));
    MY_BOOST_CHECK_CLOSE(Plane.origin.x, 5.214, tol);
    MY_BOOST_CHECK_CLOSE(Plane.origin.y, 1.32, tol);
    MY_BOOST_CHECK_CLOSE(Plane.origin.z, 8.143, tol);
    MY_BOOST_CHECK_CLOSE(Plane.normal.x, 0.39653241, tol);
    MY_BOOST_CHECK_CLOSE(Plane.normal.y, 0.23561912, tol);
    MY_BOOST_CHECK_CLOSE(Plane.normal.z, 0.88726866, tol);
    }

BOOST_AUTO_TEST_CASE( sphere_wall_math )
    {
    SphereWall Sphere = SphereWall(5.0,make_scalar3(2.0,3.0,1.0));

    // test inside
    vec3<Scalar> x = vec3<Scalar>(1.2,1.6,2.0);
    vec3<Scalar> vx = vecInsPtToWall(Sphere, x);
    Scalar dx = distWall(Sphere, x);
    MY_BOOST_CHECK_CLOSE(vx.x, -1.30818511, tol);
    MY_BOOST_CHECK_CLOSE(vx.y, -2.28932394, tol);
    MY_BOOST_CHECK_CLOSE(vx.z, 1.63523138, tol);
    MY_BOOST_CHECK_CLOSE(dx, 3.1026334, tol);

    // test outside
    x = vec3<Scalar>(13.2,3.0,1.0);
    vx = vecInsPtToWall(Sphere, x);
    dx = distWall(Sphere, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, 6.2, tol);

    // test center
    x = vec3<Scalar>(2.0,3.0,1.0);
    vx = vecInsPtToWall(Sphere, x);
    dx = distWall(Sphere, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, 5.0, tol);

    // test on surface
    x = vec3<Scalar>(2.0,3.0,6.0);
    vx = vecInsPtToWall(Sphere, x);
    dx = distWall(Sphere, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_SMALL(dx, tol_small);

    SphereWall invSphere = SphereWall(0.5,make_scalar3(1.0,2.0,3.0),false);

    // test "inside"
    x = vec3<Scalar>(1.2,3.6,2.0);
    vx = vecInsPtToWall(invSphere, x);
    dx = distWall(Sphere, x);
    MY_BOOST_CHECK_CLOSE(vx.x, -0.14729537, tol);
    MY_BOOST_CHECK_CLOSE(vx.y, -1.17836298, tol);
    MY_BOOST_CHECK_CLOSE(vx.z, 0.73647686, tol);
    MY_BOOST_CHECK_CLOSE(dx, 1, tol);

    // test "outside"
    x = vec3<Scalar>(1.2,2.1,3.0);
    vx = vecInsPtToWall(invSphere, x);
    dx = distWall(Sphere, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, 1, tol);
    }

BOOST_AUTO_TEST_CASE( cylinder_wall_math )
    {
    CylinderWall Cylinder = CylinderWall(3.5,make_scalar3(2.1,1.1,0.7),make_scalar3(1.5,0,0));

    // test inside
    vec3<Scalar> x = vec3<Scalar>(30.2,1.6,2.0);
    vec3<Scalar> vx = vecInsPtToWall(Cylinder, x);
    Scalar dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_CLOSE(vx.y, 0.75642678, tol);
    MY_BOOST_CHECK_CLOSE(vx.z, 1.96670962, tol);
    MY_BOOST_CHECK_CLOSE(dx, 2.10716117, tol);

    // test outside
    x = vec3<Scalar>(2.2,5.6,1.0);
    vx = vecInsPtToWall(Cylinder, x);
    dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, -1.00998891, tol);

    // test center
    x = vec3<Scalar>(2.1,1.1,0.7);
    vx = vecInsPtToWall(Cylinder, x);
    dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, 3.5, tol);

    // test on surface
    x = vec3<Scalar>(2.1,1.1,0.7);
    vx = vecInsPtToWall(Cylinder, x);
    dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_SMALL(dx, tol_small);


    CylinderWall invCylinder = CylinderWall(3.5,make_scalar3(2.1,1.1,0.7),make_scalar3(1.5,0,0),false);

    // test "inside"
    x = vec3<Scalar>(2.2,5.6,1.0);
    vx = vecInsPtToWall(invCylinder, x);
    dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_CLOSE(vx.y, -1.00775195, tol);
    MY_BOOST_CHECK_CLOSE(vx.z, -0.06718346, tol);
    MY_BOOST_CHECK_CLOSE(dx, 1.00998891, tol);

    // test "outside"
    x = vec3<Scalar>(30.2,1.6,2.0);
    vx = vecInsPtToWall(invCylinder, x);
    dx = distWall(Cylinder, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, -2.10716117, tol);
    }


BOOST_AUTO_TEST_CASE( plane_wall_math )
    {
    PlaneWall Plane = PlaneWall(make_scalar3(4.1,2.1,8.7),make_scalar3(-1.1,-0.3,-5.0));

    // test inside
    vec3<Scalar> x = vec3<Scalar>(30.2,1.6,2.0);
    vec3<Scalar> vx = vecInsPtToWall(Plane, x);
    Scalar dx = distWall(Plane, x);
    MY_BOOST_CHECK_CLOSE(vx.x, 1.28319392, tol);
    MY_BOOST_CHECK_CLOSE(vx.y, 0.34996198, tol);
    MY_BOOST_CHECK_CLOSE(vx.z, 5.83269962, tol);
    MY_BOOST_CHECK_CLOSE(dx, 5.9824280, tol);

    // test outside
    x = vec3<Scalar>(5.1,2.2,8.9);
    vx = vecInsPtToWall(Plane, x);
    dx = distWall(Plane, x);
    MY_BOOST_CHECK_SMALL(vx.x, tol_small);
    MY_BOOST_CHECK_SMALL(vx.y, tol_small);
    MY_BOOST_CHECK_SMALL(vx.z, tol_small);
    MY_BOOST_CHECK_CLOSE(dx, -0.4153380592, tol);
    }
