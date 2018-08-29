// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file MemoryTraceback.cc
    \brief Implements a class to track memory allocations
*/

#include "MemoryTraceback.h"

#include <execinfo.h>

//! Maximum number of symbols to trace back
#define MAX_TRACEBACK 4

void MemoryTraceback::registerAllocation(void *ptr, unsigned int nbytes, const std::string& type_hint, const std::string& tag) const
    {
    // insert element into list of allocations
    std::pair<void *, unsigned int> idx = std::make_pair(ptr, nbytes);

    m_traces[idx] = std::vector<void *>(MAX_TRACEBACK, nullptr);
    m_type_hints[idx] = type_hint;
    m_tags[idx] = tag;

    // obtain a traceback
    int num_symbols = backtrace(&m_traces[idx].front(), MAX_TRACEBACK);

    m_traces[idx].resize(num_symbols);
    }

void MemoryTraceback::unregisterAllocation(void *ptr, unsigned int nbytes) const
    {
    // remove element from list of allocations
    std::pair<void *, unsigned int> idx = std::make_pair(ptr, nbytes);

    m_traces.erase(idx);
    m_type_hints.erase(idx);
    m_tags.erase(idx);
    }

void MemoryTraceback::outputTraces(std::shared_ptr<Messenger> msg) const
    {
    msg->notice(2) << "List of memory allocations and " << MAX_TRACEBACK-1 << " last functions called at time of allocation" << std::endl;

    for (auto it_trace = m_traces.begin(); it_trace != m_traces.end(); ++it_trace)
        {
        msg->notice(2) << "** Address " << it_trace->first.first << ", " << it_trace->first.second
            << " bytes, data type " << m_type_hints[it_trace->first]
            << " [" << m_tags[it_trace->first] << "]" << std::endl;

        // translate symbol addresses into array of strings
        unsigned int size = it_trace->second.size();
        char **symbols = backtrace_symbols(&it_trace->second.front(), size);

        if (! symbols)
            throw std::runtime_error("Out of memory while trying to obtain stacktrace.");

        // begin trace in the calling function
        for (unsigned int i = 1; i < size; ++i)
            {
            msg->notice(2) << "(" << i << ") " << symbols[i] << std::endl;
            }
        }
    }
