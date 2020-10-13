// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file MemoryTraceback.cc
    \brief Implements a class to track memory allocations
*/

#include "MemoryTraceback.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>

//! Maximum number of symbols to trace back
#define MAX_TRACEBACK 4

void MemoryTraceback::registerAllocation(const void *ptr, unsigned int nbytes, const std::string& type_hint, const std::string& tag) const
    {
    // insert element into list of allocations
    std::pair<const void *, unsigned int> idx = std::make_pair(ptr, nbytes);

    m_traces[idx] = std::vector<void *>(MAX_TRACEBACK, nullptr);
    m_type_hints[idx] = type_hint;
    m_tags[idx] = tag;

    // obtain a traceback
    int num_symbols = backtrace(&m_traces[idx].front(), MAX_TRACEBACK);

    m_traces[idx].resize(num_symbols);
    }

void MemoryTraceback::unregisterAllocation(const void *ptr, unsigned int nbytes) const
    {
    // remove element from list of allocations
    std::pair<const void *, unsigned int> idx = std::make_pair(ptr, nbytes);

    m_traces.erase(idx);
    m_type_hints.erase(idx);
    m_tags.erase(idx);
    }

void MemoryTraceback::updateTag(const void *ptr, unsigned int nbytes, const std::string& tag) const
    {
    std::pair<const void *, unsigned int> idx = std::make_pair(ptr, nbytes);

    if (m_tags.find(idx) != m_tags.end())
        m_tags[idx] = tag;
    }

//! Pretty print number of bytes
inline std::string pretty_bytes(double size_bytes)
    {
    int i = 0;
    const char* units[] = {"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
    std::string unit = units[0];
    std::ostringstream oss;
    while (size_bytes > 1024)
        {
        size_bytes /= 1024;
        i++;
        if (i >= 9)
            {
            unit = std::string("?B");
            break;
            }
        else
            unit = std::string(units[i]);
        }

    oss << std::setprecision(3) << size_bytes << unit;
    return oss.str();
    }

void MemoryTraceback::outputTraces(std::shared_ptr<Messenger> msg) const
    {
    // reduce total memory
    unsigned long int nbytes_tot = 0;

    for (auto it_trace = m_traces.begin(); it_trace != m_traces.end(); ++it_trace)
        {
        nbytes_tot += it_trace->first.second;
        }

    msg->notice(2) << "Total amount of managed memory allocated through Global[Array,Vector]: " << pretty_bytes(nbytes_tot) << std::endl;
    msg->notice(2) << "Actual allocation sizes may be larger by up to the OS page size due to alignment." << std::endl;
    msg->notice(2) << "List of memory allocations and last " << MAX_TRACEBACK-1 << " functions called at time of (re-)allocation" << std::endl;

    for (auto it_trace = m_traces.begin(); it_trace != m_traces.end(); ++it_trace)
        {
        std::ostringstream oss;

        oss << "** Address " << it_trace->first.first << ", " << pretty_bytes(it_trace->first.second);

        char *realname;
        int status;
        realname = abi::__cxa_demangle(m_type_hints[it_trace->first].c_str(), 0, 0, &status);
        if (status)
            throw std::runtime_error("Status "+std::to_string(status)+" while trying to demangle data type.");

        oss << ", data type " << realname;
        free(realname);

        if (! m_tags[it_trace->first].empty())
            oss << " [" << m_tags[it_trace->first] << "]";
        msg->notice(2) << oss.str() << std::endl;

        // translate symbol addresses into array of strings
        unsigned int size = it_trace->second.size();
        char **symbols = backtrace_symbols(&it_trace->second.front(), size);

        if (! symbols)
            throw std::runtime_error("Out of memory while trying to obtain stacktrace.");

        // https://gist.github.com/fmela/591333
        // begin trace in the calling function
        for (unsigned int i = 1; i < size; ++i)
            {
            Dl_info info;
            std::ostringstream oss;
            oss << "(" << i << ") ";
            if (dladdr(it_trace->second[i], &info) && info.dli_sname)
                {
                status = -1;
                char *demangled = NULL;
                demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
                oss << ((status == 0) ? std::string(demangled) : std::string(info.dli_sname));
                if (status == 0)
                    free(demangled);
                }
            else
                {
                oss << symbols[i];
                }
            msg->notice(2) << oss.str() << std::endl;
            }
        free(symbols);
        }
    }
