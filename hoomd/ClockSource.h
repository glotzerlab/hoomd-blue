// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ClockSource.h
    \brief Declares the ClockSource class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __CLOCK_SOURCE_H__
#define __CLOCK_SOURCE_H__

#include "HOOMDMath.h"

// The clock code uses 64 bit integers for big numbers of nanoseconds.
#include <stdint.h>

#include <string>

#include <iostream>

#ifdef __GNUC__
#include <sys/time.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <Windows.h>
#include <time.h>
#include <io.h>
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


#ifdef _WIN32
inline void usleep(__int64 usec)
{
	HANDLE timer;
	LARGE_INTEGER ft;

	ft.QuadPart = -(10 * usec); // Convert to 100 nanosecond interval, negative value indicates relative time

	timer = CreateWaitableTimer(NULL, TRUE, NULL);
	SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
	WaitForSingleObject(timer, INFINITE);
	CloseHandle(timer);
}


#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

inline int gettimeofday(struct timeval *tv, struct timezone *tz)
{
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag = 0;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		tmpres /= 10;  /*convert into microseconds*/
		/*converting file time to unix epoch*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS;
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}
#endif

//! Sleep for for a time
/*! \param msec Number of milliseconds to sleep for
	\ingroup utils
*/
inline void Sleepms(int msec)
    {
    usleep(msec*1000);
    }

//! Source of time measurements
/*! Access the operating system's timer and reports a time since construction in nanoseconds.
    The resolution of the timer is system dependent, though typically around 10 microseconds
    or better. Critical accessor methods are inlined for low overhead
    \ingroup utils
*/
class PYBIND11_EXPORT ClockSource
    {
    public:
        //! Construct a ClockSource
        ClockSource();
        //! Get the current time in nanoseconds
        int64_t getTime() const;

        //! Formats a given time value to HH:MM:SS
        static std::string formatHMS(int64_t t);
    private:
        int64_t m_start_time; //!< Stores a base time to reference from
    };

//! Exports the ClockSource class to python
#ifndef NVCC
void export_ClockSource(pybind11::module& m);
#endif

inline int64_t ClockSource::getTime() const
    {
    timeval t;
    gettimeofday(&t, NULL);

    int64_t nsec = int64_t(t.tv_sec) * int64_t(1000000000) + int64_t(t.tv_usec)*int64_t(1000);
    return nsec - m_start_time;
    }

#endif
