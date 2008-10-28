/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

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
// boost 1.34.1 (gentoo)
// boost 1.33.1 (fedora 7)
// boost 1.32.0 (RHEL4)
// it may work on other versions or it may need tweaking

#if !defined(USE_STATIC)
//! Macro needed to correctly define things when boost_utf is linked dynamically
#define BOOST_TEST_DYN_LINK
#endif

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
