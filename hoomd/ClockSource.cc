// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ClockSource.cc
    \brief Defines the ClockSource class
*/


#include "ClockSource.h"

#include <sstream>
#include <iomanip>

using namespace std;
namespace py = pybind11;

/*! A newly constructed ClockSource should read ~0 when getTime() is called. There is no other way to reset the clock*/
ClockSource::ClockSource() : m_start_time(0)
    {
    // take advantage of the initial 0 start time to assign a new start time
    m_start_time = getTime();
    }

/*! \param t the time to format
*/
std::string ClockSource::formatHMS(int64_t t)
    {
    // separate out into hours minutes and seconds
    int hours = int(t / (int64_t(3600) * int64_t(1000000000)));
    t -= hours * int64_t(3600) * int64_t(1000000000);
    int minutes = int(t / (int64_t(60) * int64_t(1000000000)));
    t -= minutes * int64_t(60) * int64_t(1000000000);
    int seconds = int(t / int64_t(1000000000));

    // format the string
    ostringstream str;
    str <<  setfill('0') << setw(2) << hours << ":" << setw(2) << minutes << ":" << setw(2) << seconds;
    return str.str();
    }

#ifndef NVCC
void export_ClockSource(py::module& m)
    {
    py::class_<ClockSource>(m,"ClockSource")
    .def("getTime", &ClockSource::getTime)
    ;
    }
#endif
