// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "ExecutionConfiguration.h"

#include <iostream>
#include <math.h>
#include <memory>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "VectorMath.h"


using namespace std;


/*! \file test_messenger.cc
	\brief Unit test for Messenger
	\ingroup unit_tests
*/


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

//! Tolerance setting for near-zero comparisons
const Scalar tol_small = Scalar(1e-3);

//! Tolerance setting for comparisons
const Scalar tol = Scalar(1e-2);

//! Loose tolerance to be used with randomly generated and unpredictable comparisons
Scalar loose_tol = Scalar(10);


TEST_CASE("construction")
{
	// test each constructor separately
	vec3<Scalar> a;
	REQUIRE(std::abs(a.x) < tol_small);
	REQUIRE(std::abs(a.y) < tol_small);
	REQUIRE(std::abs(a.z) < tol_small);

	vec3<Scalar> b(123, 86, -103);
	REQUIRE(b.x == Approx(123));
	REQUIRE(b.y == Approx(86));
	REQUIRE(b.z == Approx(-103));

	Scalar3 s3 = make_scalar3(-10, 25, 92);
	vec3<Scalar> c(s3);
	REQUIRE(c.x == Approx(s3.x));
	REQUIRE(c.y == Approx(s3.y));
	REQUIRE(c.z == Approx(s3.z));

	Scalar4 s4 = make_scalar4(18, -22, 78, 12);
	vec3<Scalar> d(s4);
	REQUIRE(d.x == Approx(s4.x));
	REQUIRE(d.y == Approx(s4.y));
	REQUIRE(d.z == Approx(s4.z));

	vec3<float> e(123, 86, -103);
	REQUIRE(vec3<float>(e).x == Approx(123));
	REQUIRE(vec3<float>(e).y == Approx(86));
	REQUIRE(vec3<float>(e).z == Approx(-103));
	REQUIRE(vec3<double>(e).x == Approx(123));
	REQUIRE(vec3<double>(e).y == Approx(86));
	REQUIRE(vec3<double>(e).z == Approx(-103));

	vec3<double> f(123, 86, -103);
	REQUIRE(vec3<float>(f).x == Approx(123));
	REQUIRE(vec3<float>(f).y == Approx(86));
	REQUIRE(vec3<float>(f).z == Approx(-103));
	REQUIRE(vec3<double>(f).x == Approx(123));
	REQUIRE(vec3<double>(f).y == Approx(86));
	REQUIRE(vec3<double>(f).z == Approx(-103));

	// Test assignment
	vec3<float> g;
	vec3<double> h;
	g = vec3<float>(121, 12, -10);
	REQUIRE(g.x == Approx(121));
	REQUIRE(g.y == Approx(12));
	REQUIRE(g.z == Approx(-10));
	g = vec3<double>(-122, 15, 3);
	REQUIRE(g.x == Approx(-122));
	REQUIRE(g.y == Approx(15));
	REQUIRE(g.z == Approx(3));
	h = vec3<float>(18, 12, -1000);
	REQUIRE(h.x == Approx(18));
	REQUIRE(h.y == Approx(12));
	REQUIRE(h.z == Approx(-1000));
	h = vec3<double>(55, -64, 1);
	REQUIRE(h.x == Approx(55));
	REQUIRE(h.y == Approx(-64));
	REQUIRE(h.z == Approx(1));
}

TEST_CASE("component_wise")
{
	vec3<Scalar> a(1, 2, 3);
	vec3<Scalar> b(4, 6, 8);
	vec3<Scalar> c;

	// test each component-wise operator separately
	c = a + b;
	REQUIRE(c.x == Approx(5));
	REQUIRE(c.y == Approx(8));
	REQUIRE(c.z == Approx(11));

	c = a - b;
	REQUIRE(c.x == Approx(-3));
	REQUIRE(c.y == Approx(-4));
	REQUIRE(c.z == Approx(-5));

	c = a * b;
	REQUIRE(c.x == Approx(4));
	REQUIRE(c.y == Approx(12));
	REQUIRE(c.z == Approx(24));

	c = a / b;
	REQUIRE(c.x == Approx(1.0 / 4.0));
	REQUIRE(c.y == Approx(2.0 / 6.0));
	REQUIRE(c.z == Approx(3.0 / 8.0));

	c = -a;
	REQUIRE(c.x == Approx(-1));
	REQUIRE(c.y == Approx(-2));
	REQUIRE(c.z == Approx(-3));
}

TEST_CASE("assignment_component_wise")
{
	vec3<Scalar> a(1, 2, 3);
	vec3<Scalar> b(4, 6, 8);
	vec3<Scalar> c;

	// test each component-wise operator separately
	c = a += b;
	REQUIRE(c.x == Approx(5));
	REQUIRE(c.y == Approx(8));
	REQUIRE(c.z == Approx(11));
	REQUIRE(a.x == Approx(5));
	REQUIRE(a.y == Approx(8));
	REQUIRE(a.z == Approx(11));

	a = vec3<Scalar>(1, 2, 3);
	c = a -= b;
	REQUIRE(c.x == Approx(-3));
	REQUIRE(c.y == Approx(-4));
	REQUIRE(c.z == Approx(-5));
	REQUIRE(a.x == Approx(-3));
	REQUIRE(a.y == Approx(-4));
	REQUIRE(a.z == Approx(-5));

	a = vec3<Scalar>(1, 2, 3);
	c = a *= b;
	REQUIRE(c.x == Approx(4));
	REQUIRE(c.y == Approx(12));
	REQUIRE(c.z == Approx(24));
	REQUIRE(a.x == Approx(4));
	REQUIRE(a.y == Approx(12));
	REQUIRE(a.z == Approx(24));

	a = vec3<Scalar>(1, 2, 3);
	c = a /= b;
	REQUIRE(c.x == Approx(1.0 / 4.0));
	REQUIRE(c.y == Approx(2.0 / 6.0));
	REQUIRE(c.z == Approx(3.0 / 8.0));
	REQUIRE(a.x == Approx(1.0 / 4.0));
	REQUIRE(a.y == Approx(2.0 / 6.0));
	REQUIRE(a.z == Approx(3.0 / 8.0));
}

TEST_CASE("scalar")
{
	vec3<Scalar> a(1, 2, 3);
	Scalar b(4);
	vec3<Scalar> c;

	// test each component-wise operator separately
	c = a * b;
	REQUIRE(c.x == Approx(4));
	REQUIRE(c.y == Approx(8));
	REQUIRE(c.z == Approx(12));

	c = b * a;
	REQUIRE(c.x == Approx(4));
	REQUIRE(c.y == Approx(8));
	REQUIRE(c.z == Approx(12));

	c = a / b;
	REQUIRE(c.x == Approx(1.0 / 4.0));
	REQUIRE(c.y == Approx(2.0 / 4.0));
	REQUIRE(c.z == Approx(3.0 / 4.0));
}

TEST_CASE("assignment_scalar")
{
	vec3<Scalar> a(1, 2, 3);
	Scalar b(4);

	// test each component-wise operator separately
	a = vec3<Scalar>(1, 2, 3);
	a *= b;
	REQUIRE(a.x == Approx(4));
	REQUIRE(a.y == Approx(8));
	REQUIRE(a.z == Approx(12));

	a = vec3<Scalar>(1, 2, 3);
	a /= b;
	REQUIRE(a.x == Approx(1.0 / 4.0));
	REQUIRE(a.y == Approx(2.0 / 4.0));
	REQUIRE(a.z == Approx(3.0 / 4.0));
}

TEST_CASE("vector_ops")
{
	vec3<Scalar> a(1, 2, 3);
	vec3<Scalar> b(6, 5, 4);
	vec3<Scalar> c;
	Scalar d;

	// test each vector operation
	d = dot(a, b);
	REQUIRE(d == Approx(28));

	c = cross(a, b);
	REQUIRE(c.x == Approx(-7));
	REQUIRE(c.y == Approx(14));
	REQUIRE(c.z == Approx(-7));
}

TEST_CASE("vec_to_scalar")
{
	vec3<Scalar> a(1, 2, 3);
	Scalar w(4);
	Scalar3 m;
	Scalar4 n;

	// test convenience functions for converting between types
	m = vec_to_scalar3(a);
	REQUIRE(m.x == Approx(1));
	REQUIRE(m.y == Approx(2));
	REQUIRE(m.z == Approx(3));

	n = vec_to_scalar4(a, w);
	REQUIRE(n.x == Approx(1));
	REQUIRE(n.y == Approx(2));
	REQUIRE(n.z == Approx(3));
	REQUIRE(n.w == Approx(4));

	// test mapping of Scalar{3,4} to vec3
	a = vec3<Scalar>(0.0, 0.0, 0.0);
	a = vec3<Scalar>(m);
	REQUIRE(a.x == Approx(1));
	REQUIRE(a.y == Approx(2));
	REQUIRE(a.z == Approx(3));

	a = vec3<Scalar>(0.0, 0.0, 0.0);
	a = vec3<Scalar>(n);
	REQUIRE(a.x == Approx(1.0));
	REQUIRE(a.y == Approx(2.0));
	REQUIRE(a.z == Approx(3.0));
}

TEST_CASE("comparison")
{
	vec3<Scalar> a(1.1, 2.1, .1);
	vec3<Scalar> b = a;
	vec3<Scalar> c(.1, 1.1, 2.1);

	// test equality
	REQUIRE(a == b);
	REQUIRE(!(a == c));

	// test inequality
	REQUIRE(!(a != b));
	REQUIRE(a != c);
}

TEST_CASE("test_swap")
{
	vec3<Scalar> a(1.1, 2.2, 0.0);
	vec3<Scalar> b(3.3, 4.4, 0.0);
	vec3<Scalar> c(1.1, 2.2, 0.0);
	vec3<Scalar> d(3.3, 4.4, 0.0);

	// test swap
	a.swap(b);
	REQUIRE(a == d);
	REQUIRE(b == c);
}
