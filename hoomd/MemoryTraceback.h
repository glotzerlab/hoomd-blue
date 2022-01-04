// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: jglaser

#pragma once

/*! \file MemoryTraceback.h
    \brief Declares a class for memory allocation tracking
*/

#include <map>

#include "Messenger.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
class PYBIND11_EXPORT MemoryTraceback
    {
    public:
    //! Register a memory allocation along with a stacktrace
    /*! \param ptr The pointer to the memory address being allocated
        \param nbytes The size of the allocation in bytes
        \param type_hint A string describing the data type used
     */
    void registerAllocation(const void* ptr,
                            size_t nbytes,
                            const std::string& type_hint = std::string(),
                            const std::string& tag = std::string()) const;

    //! Unregister a memory allocation
    /*! \param ptr The pointer to the memory address being allocated
        \param nbytes The size of the allocation in bytes
     */
    void unregisterAllocation(const void* ptr, size_t nbytes) const;

    //! Output the list of pointers along with their stack traces
    void outputTraces(std::shared_ptr<Messenger> msg) const;

    //! Update the name of an allocation
    /*! \param tag The new tag
     */
    void updateTag(const void* ptr, size_t nbytes, const std::string& tag) const;

    private:
    mutable std::map<std::pair<const void*, size_t>, std::vector<void*>>
        m_traces; //!< A stacktrace per memory allocation
    mutable std::map<std::pair<const void*, size_t>, std::string>
        m_type_hints; //!< Types of memory allocations
    mutable std::map<std::pair<const void*, size_t>, std::string>
        m_tags; //!< Tags of memory allocations
    };

    } // end namespace hoomd
