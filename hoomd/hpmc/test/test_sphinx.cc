
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/extern/saruprng.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

//! Name the unit test module
#define BOOST_TEST_MODULE ShapeSphinx
#include "boost_utf_configure.h"

#include "hoomd/hpmc/ShapeSphinx.h"

#include <iostream>
#include <string>

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/function.hpp>
#include <memory>

using namespace hpmc;
using namespace hpmc::detail;

unsigned int err_count;



BOOST_AUTO_TEST_CASE( construction )
    {
    //parameters for constructing a P2N
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));

    sphinx3d_params data;
    data.N = 3;
    data.diameter[0] = 2.0;
    data.diameter[1] = -2.2;
    data.diameter[2] = -2.2;
    data.center[0] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[1] = vec3<Scalar>(0.0,0.0,1.15);
    data.center[2] = vec3<Scalar>(0,0,-1.15);

    data.circumsphereDiameter = 2.0;
    data.ignore = 0;


    // construct and check
    ShapeSphinx a(o, data);

    MY_BOOST_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    BOOST_REQUIRE_EQUAL(a.spheres.diameter, data.diameter);
        for (unsigned int i = 0; i < data.N; i++)
        {
        MY_BOOST_CHECK_CLOSE(a.spheres.diameter[i], data.diameter[i], tol);
        }

    BOOST_REQUIRE_EQUAL(a.spheres.center, data.center);
    for (unsigned int i = 0; i < data.N; i++)
        {
        MY_BOOST_CHECK_CLOSE(a.spheres.center[i].x, data.center[i].x,tol);
        MY_BOOST_CHECK_CLOSE(a.spheres.center[i].y, data.center[i].y,tol);
        MY_BOOST_CHECK_CLOSE(a.spheres.center[i].z, data.center[i].z,tol);
        }


    BOOST_CHECK(a.hasOrientation());

    MY_BOOST_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.0, tol);
    }


BOOST_AUTO_TEST_CASE( overlap_P2N_no_rot )
    {
    // first set of simple overlap checks is two double dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a double dimpled sphinx - P2N
    sphinx3d_params data;
    BOOST_REQUIRE(MAX_SPHERE_CENTERS >= 3);
    data.N = 3;
    data.diameter[0] = 2.0;
    data.diameter[1] = -2.2;
    data.diameter[2] = -2.2;
    data.center[0] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[1] = vec3<Scalar>(0.0,0.0,1.15);
    data.center[2] = vec3<Scalar>(0.0,0.0,-1.15);

    data.circumsphereDiameter = 2.0;
    data.ignore = 0;

    ShapeSphinx a(o, data);
    ShapeSphinx b(o, data);

    // zeroth test: exactly overlapping shapes
    r_ij =  vec3<Scalar>(0.0625, 0.0625, 0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0078125, 0.0078125, 0.0078125);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P2N sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(2.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,2.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-2.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,1.00);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,-1.00);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));


    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(2.1,0.2,0.0625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0.2,0.0625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,2.1,0.0625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-2.1,0.0625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.0625,1.0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.0625,-1.0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.3,0.6,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.9,0.5,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,1.95,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,1.96,0.0625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,-1.6,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.2,0.9);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.2,-0.9);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));


    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(0.0625,0.0625,0.95);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,1.975,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }

BOOST_AUTO_TEST_CASE( overlap_P4Nth_no_rot )
    {
    // simple overlap checks of two tetrahedrally dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a tetrahedrally dimpled sphinx - P4Nth
    sphinx3d_params data;
    BOOST_REQUIRE(MAX_SPHERE_CENTERS >= 5);
    data.N = 5;
    data.diameter[0] = 2.0;
    data.diameter[1] = -2.0;
    data.diameter[2] = -2.0;
    data.diameter[3] = -2.0;
    data.diameter[4] = -2.0;
    data.center[0] = vec3<Scalar>(0,0,0);
    data.center[1] = vec3<Scalar>(0.7396,0.7396,0.7396);
    data.center[2] = vec3<Scalar>(-0.7396,-0.7396,0.7396);
    data.center[3] = vec3<Scalar>(-0.7396,0.7396,-0.7396);
    data.center[4] = vec3<Scalar>(0.7396,-0.7396,-0.7396);
    data.circumsphereDiameter = 2.0;
    data.ignore = 0;

    ShapeSphinx a(o, data);
    ShapeSphinx b(o, data);

    // zeroth test: exactly overlapping shapes
    r_ij =  vec3<Scalar>(0.0, 0.0, 0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P4Nth sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(2.005,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.005,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,2.005,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-2.005,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,2.005);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,-2.005);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));


    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(2.1,0.2,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0.2,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,2.1,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-2.1,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.1,2.005);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.1,-2.005);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.8,0.6,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.8,0.6,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.6,1.8,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.6,-1.8,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0,1.8,0.6);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0,1.8,-0.6);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));


    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.8,0.6,0.2);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    //r_ij =  vec3<Scalar>(0.0,1.995,0.0);
    //BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    //BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }

BOOST_AUTO_TEST_CASE( overlap_P4Nth_P2N )
    {
    // simple overlap checks of one tetrahedrally dimpled sphinx and one double dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a tetrahedrally dimpled sphinx - P4Nth
    sphinx3d_params data;
    BOOST_REQUIRE(MAX_SPHERE_CENTERS >= 5);
    data.N = 5;
    data.diameter[0] = 2.0;
    data.diameter[1] = -2.0;
    data.diameter[2] = -2.0;
    data.diameter[3] = -2.0;
    data.diameter[4] = -2.0;
    data.center[0] = vec3<Scalar>(0,0,0);
    data.center[1] = vec3<Scalar>(0.7396,0.7396,0.7396);
    data.center[2] = vec3<Scalar>(-0.7396,-0.7396,0.7396);
    data.center[3] = vec3<Scalar>(-0.7396,0.7396,-0.7396);
    data.center[4] = vec3<Scalar>(0.7396,-0.7396,-0.7396);
    data.circumsphereDiameter = 2.0;
    data.ignore = 0;

    ShapeSphinx a(o, data);

    // build a double dimpled sphinx - P2N
    sphinx3d_params data_P2N;
    BOOST_REQUIRE(MAX_SPHERE_CENTERS >= 3);
    data_P2N.N = 3;
    data_P2N.diameter[0] = 2.0;
    data_P2N.diameter[1] = -2.2;
    data_P2N.diameter[2] = -2.2;
    data_P2N.center[0] = vec3<Scalar>(0.0,0.0,0.0);
    data_P2N.center[1] = vec3<Scalar>(0.0,0.0,1.15);
    data_P2N.center[2] = vec3<Scalar>(0.0,0.0,-1.15);

    data_P2N.circumsphereDiameter = 2.0;
    data_P2N.ignore = 0;

    ShapeSphinx b(o, data_P2N);


    // zeroth test: overlapping shapes
    r_ij =  vec3<Scalar>(0.0625, 0.0625, 0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P4Nth sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(1.95,0.625,0.625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,0.625,-1.95);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,1.95,0.625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,-1.95,0.625);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(0.425,0.425,-1.55);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.425,-0.425,1.55);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.0625,1.975,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.975,0.0625,0.0625);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.0625,1.975);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,0.625,1.215);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    }

