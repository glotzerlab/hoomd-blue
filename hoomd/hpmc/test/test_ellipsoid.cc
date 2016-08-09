

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeEllipsoid.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




#include <iostream>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

unsigned int err_count;

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    ell_params axes;
    axes.x = 3;
    axes.y = 1;
    axes.z = 2;
    axes.ignore = 0;
    ShapeEllipsoid a(o, axes);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    MY_CHECK_CLOSE(a.axes.x, axes.x, tol);
    MY_CHECK_CLOSE(a.axes.y, axes.y, tol);
    MY_CHECK_CLOSE(a.axes.z, axes.z, tol);

    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 6, tol);
    }

UP_TEST( overlap_ellipsoid_sphere )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build two ellipsoids
    ell_params axes;
    axes.x = 1;
    axes.y = 1;
    axes.z = 1;
    axes.ignore = 0;
    ShapeEllipsoid a(o, axes);
    ShapeEllipsoid b(o, axes);

    // first test, separate shapes by a large distance
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, verify overlap
    r_ij = vec3<Scalar>(1.9,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_no_rot )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build two ellipsoids
    ell_params axes_a;
    axes_a.x = 3;
    axes_a.y = 1;
    axes_a.z = 2;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 2;
    axes_b.y = 3;
    axes_b.z = 1;
    axes_b.ignore=0;
    ShapeEllipsoid a(o, axes_a);
    ShapeEllipsoid b(o, axes_b);

    // first test, separate shapes by a large distance
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides
    r_ij = vec3<Scalar>(5.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-5.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,4.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-4.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,3.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-3.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(4.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-4.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,3.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-3.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,2.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-2.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_mystery )
    {
     //first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    Scalar4 s4a;
    s4a.x = -0.5508406838001192;
    s4a.y = -0.5799554923140041;
    s4a.z =  0.4835456040882306;
    s4a.w = -0.3555415823393748;
    quat<Scalar> o_a(s4a);

    Scalar4 s4b;
    s4b.x = 1.0;
    s4b.y = 0.0;
    s4b.z = 0.0;
    s4b.w = 0.0;
    quat<Scalar> o_b(s4b);

    BoxDim box(100);

    ell_params axes_a;
    axes_a.x = 0.5;
    axes_a.y = 0.5;
    axes_a.z = 0.25;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 0.1;
    axes_b.y = 0.1;
    axes_b.z = 0.1;
    axes_b.ignore=0;
    ShapeEllipsoid a(o_a, axes_a);
    ShapeEllipsoid b(o_b, axes_b);

    r_ij = vec3<Scalar>(-0.0498649748800466,-0.3447528941432312,-0.0521417988793118);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    //set up config 2
    //s4a.x = 0.4114528487037498;
    //s4a.y = 0.5965286240678213;
    //s4a.z =  -0.6744056999702521;
    //s4a.w = 0.1415524842910064;
    //o_a=quat<Scalar>(s4a);

    //s4b.x = 1.0;
    //s4b.y = 0.0;
    //s4b.z = 0.0;
    //s4b.w = 0.0;
    //o_b=quat<Scalar>(s4b);

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(0.1301787374866989, 0.2395105367655792, 0.2211425037603733);
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    ////set up config 3
    //s4a.x = -0.0165028459295592;
    //s4a.y = -0.4043395276916358;
    //s4a.z =  0.2005837748298367;
    //s4a.w =  0.8921901992833947;
    //o_a=quat<Scalar>(s4a);

    //s4b.x = 1.0;
    //s4b.y = 0.0;
    //s4b.z = 0.0;
    //s4b.w = 0.0;
    //o_b=quat<Scalar>(s4b);

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(-0.2556580599312241, 0.1217761596058726, 0.2075825377324516);
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    ////set up config 4
    //s4a.x = -0.4614112635908928;
    //s4a.y = -0.1992192177315461;
    //s4a.z =  0.7256085949560850;
    //s4a.w = -0.4700037404571829;
    //o_a=quat<Scalar>(s4a);

    //s4b.x =0.1646737869651115;
    //s4b.y = -0.3012433618229696;
    //s4b.z = 0.9392204110154538;
    //s4b.w = 0.0000193939844737;
    //o_b=quat<Scalar>(s4b);
    //// change shape of b
    //axes_b.z=0.05;

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(0.3848695657165745, -0.1329840102836337, -0.3920737379791253 );
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_mystery_2 )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    Scalar4 s4a;

    s4a.x = -0.7852640377890254;
    s4a.y = 0.3024125779495512;
    s4a.z =  0.0043495675322003;
    s4a.w = -0.5402666979515444;
    quat<Scalar> o_a(s4a);

    Scalar4 s4b;
    s4b.x = -0.4082065233536581;
    s4b.y = -0.1532432383675237;
    s4b.z = -0.4917815736020184;
    s4b.w = 0.7536808529156596;
    quat<Scalar> o_b(s4b);

    BoxDim box(100);

    ell_params axes_a;
    axes_a.x = 0.5;
    axes_a.y = 0.5;
    axes_a.z = 0.25;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 0.5;
    axes_b.y = 0.5;
    axes_b.z = 0.25;
    axes_b.ignore=0;
    ShapeEllipsoid a(o_a, axes_a);
    ShapeEllipsoid b(o_b, axes_b);

    r_ij = vec3<Scalar>(0.0389568514454071, -0.5427436433641971, 0.4444722354577015);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));


    s4a.x = 0.9842368915478391;
    s4a.y = 0.05896400025767973;
    s4a.z = -0.16508386511473788;
    s4a.w = 0.023415923399224714;
    o_a = quat<Scalar>(s4a);

    s4b.x = 0.991613015572359;
    s4b.y = -0.024799134952532047;
    s4b.z = -0.018859144475762063;
    s4b.w = 0.12543110827358975;
    o_b = quat<Scalar>(s4b);

    axes_a.x = 0.5;
    axes_a.y = 0.25;
    axes_a.z = 0.15;
    axes_a.ignore=0;

    axes_b.x = 0.5;
    axes_b.y = 0.25;
    axes_b.z = 0.15;
    axes_b.ignore=0;

    a = ShapeEllipsoid(o_a, axes_a);
    b = ShapeEllipsoid(o_b, axes_b);

    r_ij = vec3<Scalar>(-0.0152884, -0.00495342, 0.0171991);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_sanity )
    {
     //first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    Scalar4 s4a;
    s4a.x = -0.5;
    s4a.y = -0.5;
    s4a.z =  0.5;
    s4a.w = -0.5;
    quat<Scalar> o_a(s4a);

    Scalar4 s4b;
    s4b.x = 1.0;
    s4b.y = 0.0;
    s4b.z = 0.0;
    s4b.w = 0.0;
    quat<Scalar> o_b(s4b);

    BoxDim box(100);

    ell_params axes_a;
    axes_a.x = 1;
    axes_a.y = 2;
    axes_a.z = 3;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 3;
    axes_b.y = 2;
    axes_b.z = 1;
    axes_b.ignore=0;
    ShapeEllipsoid a(o_a, axes_a);
    ShapeEllipsoid b(o_b, axes_b);

    r_ij = vec3<Scalar>(5, 0, 0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    //set up config 2
    //s4a.x = 0.4114528487037498;
    //s4a.y = 0.5965286240678213;
    //s4a.z =  -0.6744056999702521;
    //s4a.w = 0.1415524842910064;
    //o_a=quat<Scalar>(s4a);

    //s4b.x = 1.0;
    //s4b.y = 0.0;
    //s4b.z = 0.0;
    //s4b.w = 0.0;
    //o_b=quat<Scalar>(s4b);

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(0.1301787374866989, 0.2395105367655792, 0.2211425037603733);
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    ////set up config 3
    //s4a.x = -0.0165028459295592;
    //s4a.y = -0.4043395276916358;
    //s4a.z =  0.2005837748298367;
    //s4a.w =  0.8921901992833947;
    //o_a=quat<Scalar>(s4a);

    //s4b.x = 1.0;
    //s4b.y = 0.0;
    //s4b.z = 0.0;
    //s4b.w = 0.0;
    //o_b=quat<Scalar>(s4b);

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(-0.2556580599312241, 0.1217761596058726, 0.2075825377324516);
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    ////set up config 4
    //s4a.x = -0.4614112635908928;
    //s4a.y = -0.1992192177315461;
    //s4a.z =  0.7256085949560850;
    //s4a.w = -0.4700037404571829;
    //o_a=quat<Scalar>(s4a);

    //s4b.x =0.1646737869651115;
    //s4b.y = -0.3012433618229696;
    //s4b.z = 0.9392204110154538;
    //s4b.w = 0.0000193939844737;
    //o_b=quat<Scalar>(s4b);
    //// change shape of b
    //axes_b.z=0.05;

    //a = ShapeEllipsoid(o_a,axes_a);
    //b = ShapeEllipsoid(o_a,axes_b);

    //r_ij = vec3<Scalar>(0.3848695657165745, -0.1329840102836337, -0.3920737379791253 );
    //UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    //UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_range )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build two ellipsoids
    ell_params axes;
    axes.x=2;
    axes.y=1;
    axes.z=1;
    axes.ignore=0;
    ShapeEllipsoid a(o,axes);
    ShapeEllipsoid b(o, axes);

    //sweep from overlapping to non-overlapping configs in the y and z directions
    Scalar overlapping_distances[] = {1.0, 1.95, 1.9999};
    for (unsigned int i = 0; i < 3; i++)
        {
        r_ij = vec3<Scalar>(0,overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(2.0*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-2.0*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
        }

    Scalar nonoverlapping_distances[] = {2.0001, 2.001, 2.05, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for (unsigned int i = 0; i < 10; i++)
        {
        r_ij = vec3<Scalar>(0,nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(2.0*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-2.0*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
        }
    }

UP_TEST( overlap_sphere_as_ellipsoid_range )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build two ellipsoids
    ell_params axes;
    axes.x=1.125;
    axes.y=1;
    axes.z=1;
    axes.ignore=0;
    ShapeEllipsoid a(o, axes);
    ShapeEllipsoid b(o, axes);

    //sweep from overlapping to non-overlapping configs in the y and z directions
    Scalar overlapping_distances[] = {1.0, 1.95, 1.9999};
    for (unsigned int i = 0; i < 3; i++)
        {
        r_ij = vec3<Scalar>(0,overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.125*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.125*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
        }

    Scalar nonoverlapping_distances[] = {2.0001, 2.001, 2.05, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for (unsigned int i = 0; i < 10; i++)
        {
        r_ij = vec3<Scalar>(0,nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.125*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.125*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
        }
    }

UP_TEST( overlap_sphere_as_ellipsoid_range_rot1 )
    {
    // first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/2.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion
    BoxDim box(100);

    // build two ellipsoids
    ell_params axes;
    axes.x=1.125;
    axes.y=1;
    axes.z=1;
    axes.ignore=0;
    ShapeEllipsoid a(o_a, axes);
    ShapeEllipsoid b(o_b, axes);

    //sweep from overlapping to non-overlapping configs in the y and z directions
    Scalar overlapping_distances[] = {1.0, 1.95, 1.9999};
    for (unsigned int i = 0; i < 3; i++)
        {
        r_ij = vec3<Scalar>(0,1.0625*overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-1.0625*overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.0625*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.0625*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
        }

    Scalar nonoverlapping_distances[] = {2.0001, 2.001, 2.05, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for (unsigned int i = 0; i < 10; i++)
        {
        r_ij = vec3<Scalar>(0,1.0625*nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-1.0625*nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.0625*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.0625*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
        }
    }

UP_TEST( overlap_sphere_as_ellipsoid_range_crazy )
    {
     //first set of simple overlap checks is two ellipsoids at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/2.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion
    BoxDim box(100);

     //build two ellipsoids. These have different axes setups, so that when the 2nd is rotated they
     //are the same. This tests the ability of the numerical stability checker to properly handle cases
     //when different types of ellipsoids are compared. Because it sets up two ellipsoids so that the
     //rotation to fix the stability issues makes the ellipsoids have the same orientation
    ell_params axes_a;
    axes_a.x=1.125;
    axes_a.y=1;
    axes_a.z=1;
    axes_a.ignore=0;
    ShapeEllipsoid a(o_a,axes_a);
    ell_params axes_b;
    axes_b.x=1.0;
    axes_b.y=1.125;
    axes_b.z=1;
    axes_b.ignore=0;
    ShapeEllipsoid b(o_b,axes_b);

    //sweep from overlapping to non-overlapping configs in the y and z directions
    Scalar overlapping_distances[] = {1.0, 1.95, 1.9999, 1.99999};
    for (unsigned int i = 0; i < 4; i++)
        {
        r_ij = vec3<Scalar>(0,overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-overlapping_distances[i],0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-overlapping_distances[i]);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.125*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.125*overlapping_distances[i],0,0);
        UP_ASSERT(test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
        }

    Scalar nonoverlapping_distances[] = {2.0001, 2.001, 2.05, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for (unsigned int i = 0; i < 10; i++)
        {
        r_ij = vec3<Scalar>(0,nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,-nonoverlapping_distances[i],0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(0,0,-nonoverlapping_distances[i]);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(1.125*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

        r_ij = vec3<Scalar>(-1.125*nonoverlapping_distances[i],0,0);
        UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
        }
    }

UP_TEST( overlap_ellipsoid_rot1 )
    {
    // try one of the ellipsoids rotated
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/2.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build two ellipsoids
    ell_params axes_a;
    axes_a.x = 3;
    axes_a.y = 1;
    axes_a.z = 2;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 2;
    axes_b.y = 3;
    axes_b.z = 1;
    axes_b.ignore=0;
    ShapeEllipsoid a(o_a, axes_a);
    ShapeEllipsoid b(o_b, axes_b);

    // first test, separate shapes by a large distance
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides
    r_ij = vec3<Scalar>(6.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-6.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,3.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-3.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,3.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-3.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(5.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-5.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,2.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-2.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,2.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-2.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_ellipsoid_rot2 )
    {
    // try both of the ellipsoids rotated
    vec3<Scalar> r_ij;
    Scalar alpha = M_PI/2.0;
    quat<Scalar> o_a(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0)); // rotation quaternion

    BoxDim box(100);

    // build two ellipsoids
    ell_params axes_a;
    axes_a.x = 3;
    axes_a.y = 1;
    axes_a.z = 2;
    axes_a.ignore=0;
    ell_params axes_b;
    axes_b.x = 2;
    axes_b.y = 3;
    axes_b.z = 1;
    axes_b.ignore=0;
    ShapeEllipsoid a(o_a, axes_a);
    ShapeEllipsoid b(o_b, axes_b);

    // first test, separate shapes by a large distance
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all six sides
    r_ij = vec3<Scalar>(3.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-3.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,4.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-4.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,5.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-5.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(2.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-2.9,0.0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,3.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-3.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,4.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-4.9);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }
