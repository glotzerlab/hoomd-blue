// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#pragma once

/*! \file MemoryTraceback.h
    \brief Declares a class for memory allocation tracking
*/

#include <map>

#include "Messenger.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

class PYBIND11_EXPORT MemoryTraceback
    {
    public:
        //! Register a memory allocation along with a stacktrace
        /*! \param ptr The pointer to the memory address being allocated
            \param nbytes The size of the allocation in bytes
            \param type_hint A string describing the data type used
         */
        void registerAllocation(const void *ptr, unsigned int nbytes, const std::string& type_hint = std::string(),
            const std::string& tag = std::string()) const;

        //! Unregister a memory allocation
        /*! \param ptr The pointer to the memory address being allocated
            \param nbytes The size of the allocation in bytes
         */
        void unregisterAllocation(const void *ptr, unsigned int nbytes ) const;

        //! Output the list of pointers along with their stack traces
        void outputTraces(std::shared_ptr<Messenger> msg) const;

        //! Update the name of an allocation
        /*! \param tag The new tag
         */
        void updateTag(const void *ptr, unsigned int nbytes, const std::string& tag) const;

    private:
        mutable std::map<std::pair<const void *,unsigned int>, std::vector<void *> > m_traces;  //!< A stacktrace per memory allocation
        mutable std::map<std::pair<const void *,unsigned int>, std::string > m_type_hints;      //!< Types of memory allocations
        mutable std::map<std::pair<const void *,unsigned int>, std::string > m_tags;            //!< Tags of memory allocations
    };
