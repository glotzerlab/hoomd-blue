// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include "hoomd/ClockSource.h"
#include <math.h>

#include "upp11_config.h"

/*! \file utils_test.cc
    \brief Unit tests for ClockSource, and Variant
    \ingroup unit_tests
*/

using namespace std;
using namespace hoomd;

HOOMD_UP_MAIN();

// the clock test depends on timing and thus should not be run in automatic builds.
// uncomment to test by hand if the test seems to be behaving poorly

//! perform some simple checks on the clock source code
/*UP_TEST(ClockSource_test)
    {
    ClockSource c1;
    int64_t t = c1.getTime();
    // c.getTime() should read 0, but we can't expect it to be exact, so allow a tolerance
    UP_ASSERT(abs(int(t)) <= 1000000);

    // test timing a whole second
    ClockSource c2;
    int64_t t1 = c2.getTime();
    Sleep(1000);
    int64_t t2 = c2.getTime();

    UP_ASSERT(abs(int(t2 - t1 - int64_t(1000000000))) <= 20000000);*/

// unfortunately, testing of microsecond timing with a sleep routine is out of the question
// the following test code tests the ability of the timer to read nearby values
/*ClockSource c4;
int64_t times[100];
for (int i = 0; i < 100; i++)
    {
    times[i] = c4.getTime();
    }
for (int i = 0; i < 100; i++)
    {
    cout << times[i] << endl;
    }*/

// test copying timers
// operator=
/*  c1 = c2;
    t1 = c1.getTime();
    t2 = c2.getTime();
    UP_ASSERT(abs(int(t1-t2)) <= 1000000);
    // copy constructor
    ClockSource c3(c1);
    t1 = c1.getTime();
    t2 = c3.getTime();
    UP_ASSERT(abs(int(t1-t2)) <= 1000000);

    // test the ability of the clock source to format values
    UP_ASSERT_EQUAL(ClockSource::formatHMS(0), string("00:00:00"));
    UP_ASSERT_EQUAL(ClockSource::formatHMS(int64_t(1000000000)), string("00:00:01"));
    UP_ASSERT_EQUAL(ClockSource::formatHMS(int64_t(1000000000)*int64_t(11)), string("00:00:11"));
    UP_ASSERT_EQUAL(ClockSource::formatHMS(int64_t(1000000000)*int64_t(65)), string("00:01:05"));
    UP_ASSERT_EQUAL(ClockSource::formatHMS(int64_t(1000000000)*int64_t(3678)), string("01:01:18"));
    }*/
