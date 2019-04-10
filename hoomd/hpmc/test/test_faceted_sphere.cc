#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/ShapeFacetedEllipsoid.h"

#include <iostream>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hpmc;

unsigned int err_count;

UP_TEST( construction )
    {
    // parameters
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));
    o = o * (Scalar)(Scalar(1.0)/sqrt(norm2(o)));
    Scalar radius = 1.25;

    detail::faceted_ellipsoid_params p(0, false);
    p.N = 0;
    p.a = p.b = p.c = radius;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    // construct and check
    ShapeFacetedEllipsoid a(o, p);
    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    MY_CHECK_CLOSE(a.params.a, radius, tol);
    MY_CHECK_CLOSE(a.params.b, radius, tol);
    MY_CHECK_CLOSE(a.params.c, radius, tol);

    UP_ASSERT(!a.hasOrientation());

    UP_ASSERT(a.params.verts.N == 0);
    UP_ASSERT(a.params.additional_verts.N == 0);

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST( overlap )
    {
    // parameters
    vec3<Scalar> r_i;
    vec3<Scalar> r_j;
    quat<Scalar> o;
    BoxDim box(100);

    // place test spheres
    detail::faceted_ellipsoid_params p(0, false);
    p.a = p.b = p.c = 1.25;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    ShapeFacetedEllipsoid a(o, p);
    r_i = vec3<Scalar>(1,2,3);

    detail::faceted_ellipsoid_params p2(0, false);
    p2.a = p2.b = p2.c = 1.75;
    p2.ignore = 0;
    p2.verts.N = 0;
    p2.additional_verts.N = 0;

    ShapeFacetedEllipsoid b(o, p2);
    r_j = vec3<Scalar>(5,-2,-1);
    UP_ASSERT(!test_overlap(r_j - r_i, a,b,err_count));
    UP_ASSERT(!test_overlap(r_i - r_j, b,a,err_count));

    ShapeFacetedEllipsoid c(o, p2);
    r_j = vec3<Scalar>(3.9,2,3);
    UP_ASSERT(test_overlap(r_j - r_i, a,c,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, c,a,err_count));

    ShapeFacetedEllipsoid d(o, p2);
    r_j = vec3<Scalar>(1,-0.8,3);
    UP_ASSERT(test_overlap(r_j - r_i, a,d,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, d,a,err_count));

    ShapeFacetedEllipsoid e(o, p2);
    r_j = vec3<Scalar>(1,2,0.1);
    UP_ASSERT(test_overlap(r_j - r_i, a,e,err_count));
    UP_ASSERT(test_overlap(r_i - r_j, e,a,err_count));
    }

UP_TEST( overlap_boundaries )
    {
    // parameters
    quat<Scalar> o;
    BoxDim box(20);

    // place test spheres
    vec3<Scalar> pos_a(9,0,0);
    vec3<Scalar> pos_b(-8,-2,-1);
    vec3<Scalar> rij = pos_b - pos_a;
    rij = vec3<Scalar>(box.minImage(vec_to_scalar3(rij)));

    detail::faceted_ellipsoid_params p(0, false);
    p.a = p.b = p.c = 1.00;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    ShapeFacetedEllipsoid a(o, p);
    ShapeFacetedEllipsoid b(o, p);
    UP_ASSERT(!test_overlap(rij,a,b,err_count));
    UP_ASSERT(!test_overlap(-rij,b,a,err_count));

    vec3<Scalar> pos_c(-9.1,0,0);
    rij = pos_c - pos_a;
    rij = vec3<Scalar>(box.minImage(vec_to_scalar3(rij)));
    ShapeFacetedEllipsoid c(o, p);
    UP_ASSERT(test_overlap(rij,a,c,err_count));
    UP_ASSERT(test_overlap(-rij,c,a,err_count));
    }

UP_TEST( overlap_faceted )
    {
    // parameters
    vec3<Scalar> r_i;
    vec3<Scalar> r_j;
    quat<Scalar> o(1,vec3<Scalar>(0,0,0));
    BoxDim box(100);

    // place test spheres
    detail::faceted_ellipsoid_params p(1, false);
    p.a = p.b = p.c  = 0.5;
    p.n[0] = vec3<OverlapReal>(1,0,0);
    p.offset[0] = -.3;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    ShapeFacetedEllipsoid a(o, p);

    detail::faceted_ellipsoid_params p2(1, false);
    p2.N = 0;
    p2.a = p2.b = p2.c = 0.5;
    p2.ignore = 0;
    p2.verts.N = 0;
    p2.additional_verts.N = 0;

    ShapeFacetedEllipsoid b(o, p2);
    vec3<Scalar> r_ij = vec3<Scalar>(2,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.85,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.75,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    p2.N = 1;
    p2.offset[0] = -.3;

    // facet particle b, but place it so that b's circumsphere doesn't intersect a
    for (unsigned int i = 0; i < 100; ++i)
        {
        // rotate b around z
        Scalar phi = 2.0*M_PI/100.0*i;
        p2.n[0].x = cos(phi);
        p2.n[0].y = sin(phi);
        p2.n[0].z = 0.0;
        r_ij = vec3<Scalar>(0.81,0,0);
        UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));
        }

    p2.n[0].x = -1;
    p2.n[0].y = 0.0;
    p2.n[0].z = 0.0;

    // place b close to a along x, with facing facets, then translate in y-z plane
    for (unsigned int i = 0; i < 100; ++i)
        for (unsigned int j = 0; j < 100; ++j)
            {
            Scalar y = -5.0+0.1*i;
            Scalar z = -5.0+0.1*j;
            r_ij = vec3<Scalar>(.6001,y,z);
            UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
            UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));
            }

    // place b close to a along x, with facing facets, then rotate slightly around z (1deg)
    for (unsigned int i = 0; i < 10; ++i)
        {
        Scalar phi = -0.5/180.0*M_PI+1.0/180.0*M_PI/10.0*i;
        p2.n[0].x = -cos(phi);
        p2.n[0].y = -sin(phi);
        p2.n[0].z = 0.0;
        r_ij = vec3<Scalar>(.61,0,0);
        UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
        UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));
        }

    // get a vertex on the intersection circle of sphere a
    hpmc::detail::SupportFuncFacetedEllipsoid S_a(p);
    vec3<OverlapReal> v_or = S_a(vec3<OverlapReal>(1,-.3,0));
    vec3<Scalar> v(v_or.x, v_or.y, v_or.z);

    // place particle b along that axis, with patch normal to it,
    // but barely not touching
    r_ij = v+v*fast::rsqrt(dot(v,v))*Scalar(0.3001);
    p2.n[0] = -v*fast::rsqrt(dot(v,v));
    p2.offset[0] = -.3;
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    // place particle b along that axis, with patch normal to it,
    // barely overlapping
    r_ij = v+v*fast::rsqrt(dot(v,v))*Scalar(0.2999);
    p2.n[0] = -v*fast::rsqrt(dot(v,v));
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));
    }

UP_TEST( overlap_faceted_twofacets )
    {
    // parameters
    vec3<Scalar> r_i;
    vec3<Scalar> r_j;
    quat<Scalar> o(1,vec3<Scalar>(0,0,0));
    BoxDim box(100);

    // place test spheres
    detail::faceted_ellipsoid_params p(2, false);
    p.a = p.b = p.c = 0.5;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.verts.diameter = 1.0;
    p.origin = vec3<OverlapReal>(0,0,0);

    // this shape has two facets intersecting inside the sphere
    p.n[0] = vec3<OverlapReal>(1/sqrt(2),1/sqrt(2),0);
    p.offset[0] = -0.9*1/(2*sqrt(2));
    p.n[1] = vec3<OverlapReal>(1/sqrt(2),-1/sqrt(2),0);
    p.offset[1] = -0.9*1/(2*sqrt(2));
    ShapeFacetedEllipsoid::initializeVertices(p,false);
    ShapeFacetedEllipsoid a(o, p);

    detail::faceted_ellipsoid_params p2(0, false);
    p2.a = p2.b = p2.c = 0.5;
    p2.ignore = 0;
    p2.verts.N = 0;
    p2.additional_verts.N = 0;

    ShapeFacetedEllipsoid b(o, p2);
    vec3<Scalar> r_ij = vec3<Scalar>(2,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(1,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.5+0.905*0.5,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.5+0.895*0.5,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.5,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));
    }

UP_TEST( overlap_faceted_threefacets )
    {
    // parameters
    quat<Scalar> o(1,vec3<Scalar>(0,0,0));
    BoxDim box(100);

    // place test spheres
    detail::faceted_ellipsoid_params p(3, false);
    p.a = p.b = p.c = 0.5;
    p.ignore = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    // this shape has three facets coming together in a corner inside the sphere
    OverlapReal phi(2.0*M_PI/3.0);
    OverlapReal theta(M_PI/4.0);
    p.n[0] = vec3<OverlapReal>(sin(theta)*cos(0*phi),sin(theta)*sin(0*phi),cos(theta));
    p.offset[0] = -0.9*cos(theta)/2;
    p.n[1] = vec3<OverlapReal>(sin(theta)*cos(1*phi),sin(theta)*sin(1*phi),cos(theta));
    p.offset[1] = -0.9*cos(theta)/2;
    p.n[2] = vec3<OverlapReal>(sin(theta)*cos(2*phi),sin(theta)*sin(2*phi),cos(theta));
    p.offset[2] = -0.9*cos(theta)/2;

    p.verts = detail::poly3d_verts(1,false);
    p.verts.diameter = 1.0;
    p.verts.x[0] = 0;
    p.verts.y[0] = 0;
    p.verts.z[0] = 0.9/2/p.c;

    ShapeFacetedEllipsoid::initializeVertices(p,false);

    ShapeFacetedEllipsoid a(o, p);

    detail::faceted_ellipsoid_params p2(0, false);
    p2.a = p2.b = p2.c = 0.5;
    p2.ignore = 0;
    p2.verts.N = 0;
    p2.additional_verts.N = 0;


    ShapeFacetedEllipsoid b(o, p2);
    vec3<Scalar> r_ij = vec3<Scalar>(0,0,2);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,1);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,0.5+0.905*0.5);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,0.5+0.895*0.5);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,0.5);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-.99);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0,0,-1.01);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));
    }

UP_TEST( overlap_faceted_offset )
    {
    // parameters
    quat<Scalar> o(1,vec3<Scalar>(0,0,0));
    BoxDim box(100);

    // place test spheres
    detail::faceted_ellipsoid_params p(1, false);
    p.a = p.b = p.c = 0.5;
    p.n[0] = vec3<OverlapReal>(1,0,0);
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);


    ShapeFacetedEllipsoid a(o, p);

    detail::faceted_ellipsoid_params p2(0, false);
    p2.a = p2.b = p2.c = 0.5;
    p2.ignore = 0;
    p2.verts.N = 0;
    p2.additional_verts.N = 0;

    p.offset[0] = -.25;

    ShapeFacetedEllipsoid b(o, p2);
    vec3<Scalar> r_ij = vec3<Scalar>(.76,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.74,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    p.offset[0] = 0;

    r_ij = vec3<Scalar>(.51,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(0.49,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));

    p.offset[0] = .25;

    r_ij = vec3<Scalar>(.26,0,0);
    UP_ASSERT(!test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij, b,a,err_count));

    r_ij = vec3<Scalar>(.24,0,0);
    UP_ASSERT(test_overlap(r_ij, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij, b,a,err_count));
    }

UP_TEST( random_support_test )
    {
    detail::faceted_ellipsoid_params p(6, false);
    p.a = p.b = p.c = 0.5;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    // this shape has three facets coming together in a corner inside the sphere
    unsigned int n = 6;
    //p.N = n + 1;
    p.N = n;
    OverlapReal phi(2.0*M_PI/n);
    OverlapReal theta(M_PI/4.0);
    for (unsigned int i = 0; i < n; ++i)
        {
        p.n[i] = vec3<OverlapReal>(sin(theta)*cos(i*phi),sin(theta)*sin(i*phi),cos(theta));
        p.offset[i] = -1.1*cos(theta)/2;
        }
    //p.n[n] = vec3<OverlapReal>(0,0,1);
    //p.offset[n] = -0.35;

    ShapeFacetedEllipsoid::initializeVertices(p,false);

    hoomd::RandomGenerator rng;

    detail::SupportFuncFacetedEllipsoid support(p);
    for (unsigned int i = 0; i < 10000; ++i)
        {
        // draw a random vector in the excluded volume sphere of the colloid
        OverlapReal theta = hoomd::UniformDistribution<Scalar>(OverlapReal(0.0),OverlapReal(2.0*M_PI))(rng);
        OverlapReal z = hoomd::UniformDistribution<Scalar>(OverlapReal(-1.0),OverlapReal(1.0))(rng);

        // random normalized vector
        vec3<OverlapReal> n(fast::sqrt(OverlapReal(1.0)-z*z)*fast::cos(theta),fast::sqrt(OverlapReal(1.0)-z*z)*fast::sin(theta),z);

        vec3<OverlapReal> s = support(n);
        //printf("%f %f %f\n", s.x, s.y, s.z);
        UP_ASSERT(dot(s,s) <= 0.5);
        }
    }

UP_TEST( random_support_test_2 )
    {
    detail::faceted_ellipsoid_params p(2, false);
    p.a = p.b = p.c = 0.5;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    unsigned int n = 2;
    p.N = n;
    OverlapReal phi(M_PI*20.0/180.0);
    OverlapReal theta(M_PI/2.0);
    for (unsigned int i = 0; i < n; ++i)
        {
        p.n[i] = vec3<OverlapReal>(sin(theta)*cos(i*phi),sin(theta)*sin(i*phi),cos(theta));
        p.offset[i] = 0;
        }
    //p.n[n] = vec3<OverlapReal>(0,0,1);
    //p.offset[n] = -0.35;

    ShapeFacetedEllipsoid::initializeVertices(p,false);

    hoomd::RandomGenerator rng;

    detail::SupportFuncFacetedEllipsoid support(p);
    for (unsigned int i = 0; i < 10000; ++i)
        {
        // draw a random vector in the excluded volume sphere of the colloid
        OverlapReal theta = hoomd::UniformDistribution<Scalar>(OverlapReal(0.0),OverlapReal(2.0*M_PI))(rng);
        OverlapReal z = hoomd::UniformDistribution<Scalar>(OverlapReal(-1.0),OverlapReal(1.0))(rng);

        // random normalized vector
        vec3<OverlapReal> n(fast::sqrt(OverlapReal(1.0)-z*z)*fast::cos(theta),fast::sqrt(OverlapReal(1.0)-z*z)*fast::sin(theta),z);

        vec3<OverlapReal> s = support(n);
        //printf("%f %f %f\n", s.x, s.y, s.z);
        UP_ASSERT(dot(s,s) <= 0.5);
        }
    }

UP_TEST( overlap_special_case )
    {
    // parameters
    BoxDim box(100);

    detail::faceted_ellipsoid_params p(2, false);
    p.a = p.b = p.c = 0.5;
    p.ignore = 0;
    p.verts.N = 0;
    p.additional_verts.N = 0;
    p.origin = vec3<OverlapReal>(0,0,0);

    unsigned int n = 2;
    p.N = n;
    OverlapReal phi(M_PI*20.0/180.0);
    OverlapReal theta(M_PI/2.0);
    for (unsigned int i = 0; i < n; ++i)
        {
        p.n[i] = vec3<OverlapReal>(sin(theta)*cos(i*phi),sin(theta)*sin(i*phi),cos(theta));
        p.offset[i] = 0;
        }

    ShapeFacetedEllipsoid::initializeVertices(p,false);

    // place test spheres
    ShapeFacetedEllipsoid a(quat<Scalar>(.3300283551216,vec3<Scalar>(0.01934501715004,-0.9390037059784, 0.09475778788328)),p);
    ShapeFacetedEllipsoid b(quat<Scalar>(-0.225227072835,vec3<Scalar>(-0.3539296984673,-0.8667258024216,-0.269801825285)),p);

    vec3<Scalar> r_a(-0.1410365402699,-0.3096362948418,-0.04636116325855);
    vec3<Scalar> r_b(-0.7674461603165,-0.5918425917625,-0.3122854232788);
    vec3<Scalar> r_ab = r_b-r_a;
    UP_ASSERT(test_overlap(r_ab, a,b,err_count));
    UP_ASSERT(test_overlap(-r_ab,a,b,err_count));
    }
