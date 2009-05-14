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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE GPUArrayTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "GPUArray.h"

#ifdef ENABLE_CUDA
#include "gpu_array_test.cuh"
#endif

using namespace std;
using namespace boost;

/*! \file gpu_array_test.cc
	\brief Implements unit tests for GPUArray
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! boost test case for testing the basic operation of GPUArray
BOOST_AUTO_TEST_CASE( GPUArray_basic_tests )
	{
	ExecutionConfiguration exec_conf;
	GPUArray<int> gpu_array(100, exec_conf);
	
	// basic check: ensure that the number of elements is set correctly
	BOOST_CHECK_EQUAL((int)gpu_array.getNumElements(), 100);

	// basic check 2: acquire the data on the host and fill out a pattern
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			h_handle.data[i] = i;
		}
		
	// basic check 3: verify the data set in check 2
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::read);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			BOOST_CHECK_EQUAL(h_handle.data[i], i);
		}
		
	// basic check 3.5: test the construction of a 2-D GPUArray
	GPUArray<int> gpu_array_2d(63, 120, exec_conf);
	BOOST_CHECK_EQUAL((int)gpu_array_2d.getPitch(), 64);
	BOOST_CHECK_EQUAL((int)gpu_array_2d.getHeight(), 120);
	BOOST_CHECK_EQUAL((int)gpu_array_2d.getNumElements(), 7680);
		
	// basic check 4: verify the copy constructor
	GPUArray<int> array_b(gpu_array);
	BOOST_CHECK_EQUAL((int)array_b.getNumElements(), 100);
	BOOST_CHECK_EQUAL((int)array_b.getPitch(), 100);
	BOOST_CHECK_EQUAL((int)array_b.getHeight(), 1);
	
		{
		ArrayHandle<int> h_handle(array_b, access_location::host, access_mode::read);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)array_b.getNumElements(); i++)
			BOOST_CHECK_EQUAL(h_handle.data[i], i);
		}
		
	// basic check 5: verify the = operator
	GPUArray<int> array_c(1, exec_conf);
	array_c = gpu_array;

	BOOST_CHECK_EQUAL((int)array_c.getNumElements(), 100);
	BOOST_CHECK_EQUAL((int)array_c.getPitch(), 100);
	BOOST_CHECK_EQUAL((int)array_c.getHeight(), 1);	
	
		{
		ArrayHandle<int> h_handle(array_c, access_location::host, access_mode::read);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)array_c.getNumElements(); i++)
			BOOST_CHECK_EQUAL(h_handle.data[i], i);
		}
	
	}
	
#ifdef ENABLE_CUDA
//! boost test case for testing device to/from host transfers
BOOST_AUTO_TEST_CASE( GPUArray_transfer_tests )
	{
	ExecutionConfiguration exec_conf;
	BOOST_REQUIRE(exec_conf.gpu.size() == 1);
	
	GPUArray<int> gpu_array(100, exec_conf);
	
	// initialize the data on the device
		{
		ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
		BOOST_REQUIRE(d_handle.data != NULL);
		
		exec_conf.gpu[0]->call(bind(gpu_fill_test_pattern, d_handle.data, gpu_array.getNumElements()));
		}
		
	// copy it to the host and verify
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			{
			BOOST_CHECK_EQUAL(h_handle.data[i], i*i);
			// overwrite the data as we verify
			h_handle.data[i] = 100+i;
			}
		}
		
	// data has been overwitten on the host. Increment it on the device in overwrite mode
	// and verify that the data was not copied from the host to device
		{
		ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::overwrite);
		BOOST_REQUIRE(d_handle.data != NULL);
		
		exec_conf.gpu[0]->call(bind(gpu_add_one, d_handle.data, gpu_array.getNumElements()));
		}
		
	// copy it back to the host and verify
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			{
			BOOST_CHECK_EQUAL(h_handle.data[i], i*i+1);
			// overwrite the data as we verify
			h_handle.data[i] = 100+i;
			}
		}
		
	// access it on the device in read only mode, but be a bad boy and overwrite the data
	// the verify on the host should then still show the overwritten data as the internal state
	// should still be hostdevice and not copy the data back from the device
		{
		ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::read);
		BOOST_REQUIRE(d_handle.data != NULL);
		
		exec_conf.gpu[0]->call(bind(gpu_add_one, d_handle.data, gpu_array.getNumElements()));
		}
		
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			{
			BOOST_CHECK_EQUAL(h_handle.data[i], 100+i);
			}
		}
		
	// finally, test host-> device copies
		{
		ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
		BOOST_REQUIRE(d_handle.data != NULL);
		
		exec_conf.gpu[0]->call(bind(gpu_add_one, d_handle.data, gpu_array.getNumElements()));
		}
		
	// via the read access mode
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::read);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			{
			BOOST_CHECK_EQUAL(h_handle.data[i], 100+i+1);
			}
		}
		
		{
		ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
		BOOST_REQUIRE(d_handle.data != NULL);
		
		exec_conf.gpu[0]->call(bind(gpu_add_one, d_handle.data, gpu_array.getNumElements()));
		}
		
	// and via the readwrite access mode
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		BOOST_REQUIRE(h_handle.data != NULL);
		for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
			{
			BOOST_CHECK_EQUAL(h_handle.data[i], 100+i+1+1);
			}
		}
	}
	
//! Tests operations on NULL GPUArrays
BOOST_AUTO_TEST_CASE( GPUArray_null_tests )
	{
	// Construct a NULL GPUArray
	GPUArray<int> a;
	
	BOOST_CHECK(a.isNull());
	BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned)0);
	
	// check copy construction of a NULL GPUArray
	GPUArray<int> b(a);
	
	BOOST_CHECK(b.isNull());
	BOOST_CHECK_EQUAL(b.getNumElements(), (unsigned)0);
	
	// check assignment of a NULL GPUArray
	ExecutionConfiguration exec_conf;
	GPUArray<int> c(1000, exec_conf);
	c = a;
	
	BOOST_CHECK(c.isNull());
	BOOST_CHECK_EQUAL(c.getNumElements(), (unsigned)0);
	
	// check swapping of a NULL GPUArray
	GPUArray<int> d(1000, exec_conf);
	
	d.swap(a);
	BOOST_CHECK(d.isNull());
	BOOST_CHECK_EQUAL(d.getNumElements(), (unsigned)0);
	
	BOOST_CHECK(!a.isNull());
	BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned)1000);
	}


#endif
	
#ifdef WIN32
#pragma warning( pop )
#endif
