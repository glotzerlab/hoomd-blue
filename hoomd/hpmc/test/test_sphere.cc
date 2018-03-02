
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/ShapeSphere.h"

#include <iostream>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

using namespace hpmc;
using namespace hpmc::detail;

unsigned int err_count;

UP_TEST( construction )
    {
    // parameters
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));
    o = o * (Scalar)(Scalar(1.0)/sqrt(norm2(o)));
    sph_params par;
    par.radius = 1.25;
    par.ignore = 0;
    par.isOriented = false;

    // construct and check
    ShapeSphere a(o, par);
    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    MY_CHECK_CLOSE(a.params.radius, par.radius, tol);

    UP_ASSERT(!a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST( overlap_sphere)
    {
    // parameters
    vec3<Scalar> r_i;
    vec3<Scalar> r_j;
    quat<Scalar> o;
    BoxDim box(100);

    sph_params par;
    par.radius=1.25;
    par.ignore=0;
    par.isOriented = false;

    // place test spheres
    ShapeSphere a(o, par);
    r_i = vec3<Scalar>(1,2,3);

    par.radius = 1.75;
    ShapeSphere b(o, par);
    r_j = vec3<Scalar>(5,-2,-1);
    UP_ASSERT(!test_overlap(r_j - r_i, a,b,err_count));
    UP_ASSERT(!test_overlap(r_i - r_j, b,a,err_count));

    ShapeSphere c(o, par);
    r_j = vec3<Scalar>(3.9,2,3);
    UP_ASSERT(test_overlap(r_j - r_i, a,c,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, c,a,err_count));

    ShapeSphere d(o, par);
    r_j = vec3<Scalar>(1,-0.8,3);
    UP_ASSERT(test_overlap(r_j - r_i, a,d,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, d,a,err_count));

    ShapeSphere e(o, par);
    r_j = vec3<Scalar>(1,2,0.1);
    UP_ASSERT(test_overlap(r_j - r_i, a,e,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, e,a,err_count));
    }

UP_TEST( overlap_boundaries )
    {
    // parameters
    quat<Scalar> o;
    BoxDim box(20);
    sph_params par;
    par.radius=1.0;
    par.ignore = 0;
    par.isOriented = false;

    // place test spheres
    vec3<Scalar> pos_a(9,0,0);
    vec3<Scalar> pos_b(-8,-2,-1);
    vec3<Scalar> rij = pos_b - pos_a;
    rij = vec3<Scalar>(box.minImage(vec_to_scalar3(rij)));
    ShapeSphere a(o, par);
    ShapeSphere b(o, par);
    UP_ASSERT(!test_overlap(rij,a,b,err_count));
    UP_ASSERT(!test_overlap(-rij,b,a,err_count));

    vec3<Scalar> pos_c(-9.1,0,0);
    rij = pos_c - pos_a;
    rij = vec3<Scalar>(box.minImage(vec_to_scalar3(rij)));
    ShapeSphere c(o, par);
    UP_ASSERT(test_overlap(rij,a,c,err_count));
    UP_ASSERT(test_overlap(-rij,c,a,err_count));
    }
