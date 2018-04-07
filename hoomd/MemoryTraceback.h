// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#pragma once

/*! \file MemoryTraceback.h
    \brief Declares a class for memory allocation tracking
*/

#include <map>

#include "Messenger.h"

class MemoryTraceback
    {
    public:
        //! Constructor
        MemoryTraceback() {}

        //! Register a memory allocation along with a stacktrace
        /*! \param ptr The pointer to the memory address being allocated
            \param nbytes The size of the allocation in bytes
         */
        void registerAllocation(void *ptr, unsigned int nbytes) const;

        //! Output the list of pointers along with their stack traces
        void outputTraces(std::shared_ptr<Messenger> msg) const;

    private:
        mutable std::map<std::pair<void *,unsigned int>, std::vector<void *> > m_traces;  //! A stacktrace per memory allocation
    };
