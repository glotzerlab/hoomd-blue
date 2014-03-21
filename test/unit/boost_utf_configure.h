/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

#include "HOOMDMath.h"


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
