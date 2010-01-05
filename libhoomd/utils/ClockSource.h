/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file ClockSource.h
    \brief Declares the ClockSource class
*/

#ifndef __CLOCK_SOURCE_H__
#define __CLOCK_SOURCE_H__

// The clock code uses 64 bit integers for big numbers of nanoseconds. Include the
// cross-platform int64_t types from boost
#include <boost/cstdint.hpp>
using boost::int64_t;
using boost::uint64_t;

#include <string>

#include <iostream>

// If this is being compiled on a windows platform, we need to define a surrogate
// gettimeofday
#ifdef WIN32
#include <windows.h>
#define EPOCHFILETIME (116444736000000000i64)

//! Emulation class for timezone on windows
/*! \ingroup utils
*/
struct timezone
    {
    int tz_minuteswest;
    int tz_dsttime;
    };

//! Emulation for gettimeofday in windows
/*! \param tv timeval to return current time in
    \param tz unused
    \ingroup utils
*/
__inline int gettimeofday(struct timeval *tv, struct timezone *tz)
    {
    FILETIME        ft;
    LARGE_INTEGER   li;
    __int64         t;
    static int      tzflag;
    
    if (tv)
        {
        GetSystemTimeAsFileTime(&ft);
        li.LowPart  = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;
        t  = li.QuadPart;       /* In 100-nanosecond intervals */
        t -= EPOCHFILETIME;     /* Offset to the Epoch time */
        t /= 10;                /* In microseconds */
        tv->tv_sec  = (long)(t / 1000000);
        tv->tv_usec = (long)(t % 1000000);
        }
        
    // lets just ignore the whole timezone thing
    /*if (tz)
        {
        if (!tzflag)
            {
            _tzset();
            tzflag++;
            }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
        }*/
    
    return 0;
    }

#else
// else, we are on a unix-ish platform and need to include a few files to get
// gettimeofday
#include <sys/time.h>
#include <unistd.h>


// unix-ish systems also need a Sleep function
//! Sleep for for a time
/*! \param msec Number of milliseconds to sleep for
    \ingroup utils
*/
inline void Sleep(int msec)
    {
    usleep(msec*1000);
    }
#endif

//! Source of time measurements
/*! Access the operating system's timer and reports a time since construction in nanoseconds.
    The resolution of the timer is system dependant, though typically around 10 microseconds
    or better. Critical accessor methods are inlined for low overhead
    \ingroup utils
*/
class ClockSource
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
void export_ClockSource();

inline int64_t ClockSource::getTime() const
    {
    timeval t;
    gettimeofday(&t, NULL);
    
    int64_t nsec = int64_t(t.tv_sec) * int64_t(1000000000) + int64_t(t.tv_usec)*int64_t(1000);
    return nsec - m_start_time;
    }

#endif



