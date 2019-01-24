// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Profiler.h
    \brief Declares the Profiler class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ExecutionConfiguration.h"
#include "ClockSource.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_NVTOOLS
#include <nvToolsExt.h>
#endif

#include <string>
#include <stack>
#include <map>
#include <iostream>
#include <cassert>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Allow score-p instrumentation
#ifdef SCOREP_USER_ENABLE
#include <scorep/SCOREP_User.h>
#endif

#ifndef __PROFILER_H__
#define __PROFILER_H__

// forward declarations
class ProfileDataElem;
class Profiler;

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup utils Utility classes
    \brief General purpose utility classes
*/

/*! @}
*/

//! Internal class for storing profile data
/*! This is a simple utility class, so it is fully public. It is really only designed to be used in
    concert with the Profiler class.
    \ingroup utils
*/
class PYBIND11_EXPORT ProfileDataElem
    {
    public:
        //! Constructs an element with zeroed counters
        ProfileDataElem() : m_start_time(0), m_elapsed_time(0), m_flop_count(0), m_mem_byte_count(0)
            #ifdef SCOREP_USER_ENABLE
            , m_scorep_region(SCOREP_USER_INVALID_REGION)
            #endif
            {}

        //! Returns the total elapsed time of this nodes children
        int64_t getChildElapsedTime() const;
        //! Returns the total flop count of this node + children
        int64_t getTotalFlopCount() const;
        //! Returns the total memory byte count of this node + children
        int64_t getTotalMemByteCount() const;

        //! Output helper function
        void output(std::ostream &o, const std::string &name, int tab_level, int64_t total_time, int name_width) const;
        //! Another output helper function
        void output_line(std::ostream &o,
                         const std::string &name,
                         double sec,
                         double perc,
                         double flops,
                         double bytes,
                         unsigned int name_width) const;

        std::map<std::string, ProfileDataElem> m_children; //!< Child nodes of this profile

        int64_t m_start_time;   //!< The start time of the most recent timed event
        int64_t m_elapsed_time; //!< A running total of elapsed running time
        int64_t m_flop_count;   //!< A running total of floating point operations
        int64_t m_mem_byte_count;   //!< A running total of memory bytes transferred

        #ifdef SCOREP_USER_ENABLE
        SCOREP_User_RegionHandle m_scorep_region;   //!< ScoreP region identifier
        #endif
    };



//! A class for doing coarse-level profiling of code
/*! Stores and organizes a tree of profiles that can be created with a simple push/pop
    type interface. Any number of root profiles can be created via the default constructor
    Profiler::Profiler(). Sub-profiles are created with the push() member. They take a time sample on creation
    and take a second time sample when they are pop() ed. One can perform a guesstimate
    on memory bandwidth and FLOPS by calling the appropriate pop() member that takes
    a number of operations executed as a parameter.

    Pushing and popping the same profile tree over and over is important to do,
    the system will tally up total time in each slot. Pushing and popping different
    names on each pass will generate a jumbled mess.

    There are versions of push() and pop() that take in a reference to an ExecutionConfiguration.
    These methods automatically synchronize with the asynchronous GPU execution stream in order
    to provide accurate timing information.

    These profiles can of course be output via normal ostream operators.
    \ingroup utils
    */
class PYBIND11_EXPORT Profiler
    {
    public:
        //! Constructs an empty profiler and starts its timer ticking
        Profiler(const std::string& name = "Profile");
        //! Pushes a new sub-category into the current category
        void push(const std::string& name);
        //! Pops back up to the next super-category
        void pop(uint64_t flop_count = 0, uint64_t byte_count = 0);

        //! Pushes a new sub-category into the current category & syncs the GPUs
        void push(std::shared_ptr<const ExecutionConfiguration> exec_conf, const std::string& name);
        //! Pops back up to the next super-category & syncs the GPUs
        void pop(std::shared_ptr<const ExecutionConfiguration> exec_conf, uint64_t flop_count = 0, uint64_t byte_count = 0);

    private:
        ClockSource m_clk;  //!< Clock to provide timing information
        std::string m_name; //!< The name of this profile
        ProfileDataElem m_root; //!< The root profile element
        std::stack<ProfileDataElem *> m_stack;  //!< A stack of data elements for the push/pop structure

        //! Output helper function
        void output(std::ostream &o);

        //! friend operator to enable stream output
        friend std::ostream& operator<<(std::ostream &o, Profiler& prof);
    };

//! Exports the Profiler class to python
#ifndef NVCC
void export_Profiler(pybind11::module& m);
#endif


//! Output operator for Profiler
PYBIND11_EXPORT std::ostream& operator<<(std::ostream &o, Profiler& prof);

/////////////////////////////////////
// Profiler inlines

inline void Profiler::push(std::shared_ptr<const ExecutionConfiguration> exec_conf, const std::string& name)
    {
#if defined(ENABLE_CUDA) && !defined(ENABLE_NVTOOLS)
    // nvtools profiling disables synchronization so that async CPU/GPU overlap can be seen
    if(exec_conf->isCUDAEnabled())
        {
        exec_conf->multiGPUBarrier();
        cudaDeviceSynchronize();
        }
#endif
    push(name);
   }

inline void Profiler::pop(std::shared_ptr<const ExecutionConfiguration> exec_conf, uint64_t flop_count, uint64_t byte_count)
    {
#if defined(ENABLE_CUDA) && !defined(ENABLE_NVTOOLS)
    // nvtools profiling disables synchronization so that async CPU/GPU overlap can be seen
    if(exec_conf->isCUDAEnabled())
        {
        exec_conf->multiGPUBarrier();
        cudaDeviceSynchronize();
        }
#endif
    pop(flop_count, byte_count);
    }

inline void Profiler::push(const std::string& name)
    {
    // sanity checks
    assert(!m_stack.empty());

    #ifdef ENABLE_NVTOOLS
    nvtxRangePush(name.c_str());
    #endif

    // pushing a new record on to the stack involves taking a time sample
    int64_t t = m_clk.getTime();

    ProfileDataElem *cur = m_stack.top();

    // then creating (or accessing) the named sample and setting the start time
    cur->m_children[name].m_start_time = t;

    // and updating the stack
    m_stack.push(&cur->m_children[name]);

    #ifdef SCOREP_USER_ENABLE
    // log Score-P region
    SCOREP_USER_REGION_BEGIN( cur->m_children[name].m_scorep_region, name.c_str(),SCOREP_USER_REGION_TYPE_COMMON )
    #endif
    }

inline void Profiler::pop(uint64_t flop_count, uint64_t byte_count)
    {
    // sanity checks
    assert(!m_stack.empty());
    assert(!(m_stack.top() == &m_root));

    #ifdef ENABLE_NVTOOLS
    nvtxRangePop();
    #endif

    // popping up a level in the profile stack involves taking a time sample
    int64_t t = m_clk.getTime();

    // then increasing the elapsed time for the current item
    ProfileDataElem *cur = m_stack.top();
    #ifdef SCOREP_USER_ENABLE
    SCOREP_USER_REGION_END(cur->m_scorep_region)
    #endif
    cur->m_elapsed_time += t - cur->m_start_time;

    // and increasing the flop and mem counters
    cur->m_flop_count += flop_count;
    cur->m_mem_byte_count += byte_count;

    // and finally popping the stack so that the next pop will access the correct element
    m_stack.pop();
    }

#endif
