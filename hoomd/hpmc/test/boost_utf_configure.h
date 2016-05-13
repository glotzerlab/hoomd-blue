// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



/*! \file boost_utf_configure.h
    \brief Helps unit tests setup the boost unit testing framework
    \details boost, for whatever reason, has greatly changed the interface
        for the unit testing framework over the versions. This header file
        does it's best to setup BOOST_AUTO_TEST_CASE( ) to work for
        any version of boost from 1.32 to 1.34 ... hopefully not much
        changes in 1.35
    \note This file should be included only once and by a file that will
        compile into a unit test executable
*/

// unit testing has changed much over the different boost versions: try to
// handle all the way back to 1.32 (included RHEL4 based installs which are still quite common
// these days)

// setup is tested and works on:
// boost 1.36.0 (Mac OS X)
// boost 1.35.1 (gentoo)
// boost 1.34.1 (gentoo)
// boost 1.33.1 (fedora 7)
// boost 1.32.0 (RHEL4)
// it may work on other versions or it may need tweaking

#include "hoomd/HOOMDMath.h"


#include <boost/version.hpp>
#if (BOOST_VERSION >= 103400)
#include <boost/test/unit_test.hpp>
#else
//! Macro needed to define the main() function on older versions of boost
#define BOOST_AUTO_TEST_MAIN
#include <boost/test/auto_unit_test.hpp>

#if (BOOST_VERSION < 103300)
//! Compatibility macro to make older versions of boost look like newer ones
#define BOOST_AUTO_TEST_CASE( f ) BOOST_AUTO_UNIT_TEST( f )
//! Compatibility macro to make older versions of boost look like newer ones
#define BOOST_CHECK_SMALL( f, tol ) BOOST_CHECK( (f) < (tol) )
//! Compatibility macro to make older versions of boost look like newer ones
#define BOOST_REQUIRE_EQUAL( a, b) BOOST_REQUIRE( (a) == (b) )
#endif

#endif

#include <boost/test/floating_point_comparison.hpp>

// ******** helper macros
//! Helper macro for checking if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for checking if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))
//! Need a simple define for requireing two values which are unsigned
#define BOOST_REQUIRE_EQUAL_UINT(a,b) BOOST_REQUIRE_EQUAL(a,(unsigned int)(b))
//! Need a simple define for checking two values which are unsigned
#define BOOST_CHECK_EQUAL_UINT(a,b) BOOST_CHECK_EQUAL(a,(unsigned int)(b))

//! Tolerance setting for near-zero comparisons
const Scalar tol_small = Scalar(1e-3);

//! Tolerance setting for comparisons
const Scalar tol = Scalar(1e-2);

//! Loose tolerance to be used with randomly generated and unpredictable comparisons
Scalar loose_tol = Scalar(10);

// helper functions to set up MPI environment
#include "MPITestSetup.h"
