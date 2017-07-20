
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/ShapeSphinx.h"

#include <iostream>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

using namespace hpmc;
using namespace hpmc::detail;

unsigned int err_count;



UP_TEST( construction )
    {
    //parameters for constructing a P2N
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));

    sphinx3d_params data;
    data.N = 3;
    data.diameter[0] = 2.0;
    data.diameter[1] = -2.2;
    data.diameter[2] = -2.2;
    data.diameter[3] = 0;
    data.diameter[4] = 0;
    data.diameter[5] = 0;
    data.diameter[6] = 0;
    data.diameter[7] = 0;
    data.center[0] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[1] = vec3<Scalar>(0.0,0.0,1.15);
    data.center[2] = vec3<Scalar>(0,0,-1.15);
    data.center[3] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[4] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[5] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[6] = vec3<Scalar>(0.0,0.0,0.0);
    data.center[7] = vec3<Scalar>(0.0,0.0,0.0);

    data.circumsphereDiameter = 2.0;
    data.ignore = 0;


    // construct and check
    ShapeSphinx a(o, data);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.spheres.diameter, data.diameter);
        for (unsigned int i = 0; i < data.N; i++)
        {
        MY_CHECK_CLOSE(a.spheres.diameter[i], data.diameter[i], tol);
        }

    MY_ASSERT_EQUAL(a.spheres.center, data.center);
    for (unsigned int i = 0; i < data.N; i++)
        {
        MY_CHECK_CLOSE(a.spheres.center[i].x, data.center[i].x,tol);
        MY_CHECK_CLOSE(a.spheres.center[i].y, data.center[i].y,tol);
        MY_CHECK_CLOSE(a.spheres.center[i].z, data.center[i].z,tol);
        }


    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.0, tol);
    }


UP_TEST( overlap_P2N_no_rot )
    {
    // first set of simple overlap checks is two double dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a double dimpled sphinx - P2N
    sphinx3d_params data;
    UP_ASSERT(MAX_SPHERE_CENTERS >= 3);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0078125, 0.0078125, 0.0078125);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P2N sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(2.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,2.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-2.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,1.00);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,-1.00);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));


    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(2.1,0.2,0.0625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0.2,0.0625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,2.1,0.0625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-2.1,0.0625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.0625,1.0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.0625,-1.0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.3,0.6,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.9,0.5,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,1.95,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,1.96,0.0625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.4,-1.6,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.2,0.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.2,-0.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));


    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(0.0625,0.0625,0.95);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,1.975,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_P4Nth_no_rot )
    {
    // simple overlap checks of two tetrahedrally dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a tetrahedrally dimpled sphinx - P4Nth
    sphinx3d_params data;
    UP_ASSERT(MAX_SPHERE_CENTERS >= 5);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P4Nth sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(2.005,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.005,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,2.005,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-2.005,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,2.005);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0,-2.005);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));


    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(2.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-2.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,2.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-2.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.1,2.005);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.1,-2.005);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.8,0.6,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.8,0.6,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.6,1.8,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.6,-1.8,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0,1.8,0.6);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0,1.8,-0.6);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));


    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.8,0.6,0.2);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    //r_ij =  vec3<Scalar>(0.0,1.995,0.0);
    //UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_P4Nth_P2N )
    {
    // simple overlap checks of one tetrahedrally dimpled sphinx and one double dimpled sphinx at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
    BoxDim box(100);

    // build a tetrahedrally dimpled sphinx - P4Nth
    sphinx3d_params data;
    UP_ASSERT(MAX_SPHERE_CENTERS >= 5);
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
    UP_ASSERT(MAX_SPHERE_CENTERS >= 3);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate P4Nth sphinx by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides of base
    r_ij =  vec3<Scalar>(1.95,0.625,0.625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,0.625,-1.95);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,1.95,0.625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,-1.95,0.625);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all six sides
    r_ij =  vec3<Scalar>(0.425,0.425,-1.55);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.425,-0.425,1.55);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));


    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.0625,1.975,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.975,0.0625,0.0625);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0625,0.0625,1.975);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.625,0.625,1.215);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    }
