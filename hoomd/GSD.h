// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/extern/gsd.h"
#include <sstream>
#include <stdexcept>
#include <string>

namespace hoomd
    {
namespace detail
    {
/// Utility class to collect common GSD file operations.
class GSDUtils
    {
    public:
    /// Check and raise an exception if an error occurs
    static void checkError(int retval, const std::string& fname)
        {
        // checkError prints errors and then throws exceptions for common gsd error codes
        if (retval == GSD_ERROR_IO)
            {
            std::ostringstream s;
            s << "GSD: " << strerror(errno) << " - " << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_INVALID_ARGUMENT)
            {
            std::ostringstream s;
            s << "GSD: Invalid argument"
                 " - "
              << fname;
            throw std::invalid_argument(s.str());
            }
        else if (retval == GSD_ERROR_NOT_A_GSD_FILE)
            {
            std::ostringstream s;
            s << "GSD: Not a GSD file"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_INVALID_GSD_FILE_VERSION)
            {
            std::ostringstream s;
            s << "GSD: Invalid GSD file version"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_FILE_CORRUPT)
            {
            std::ostringstream s;
            s << "GSD: File corrupt"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_MEMORY_ALLOCATION_FAILED)
            {
            std::ostringstream s;
            s << "GSD: Memory allocation failed"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_NAMELIST_FULL)
            {
            std::ostringstream s;
            s << "GSD: Namelist full"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_FILE_MUST_BE_WRITABLE)
            {
            std::ostringstream s;
            s << "GSD: File must be writable"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval == GSD_ERROR_FILE_MUST_BE_READABLE)
            {
            std::ostringstream s;
            s << "GSD: File must be readable"
                 " - "
              << fname;
            throw std::runtime_error(s.str());
            }
        else if (retval != GSD_SUCCESS)
            {
            std::ostringstream s;
            s << "GSD: " << "Unknown error " << retval << " writing: " << fname;
            throw std::runtime_error(s.str());
            }
        }
    };
    } // namespace detail
    } // namespace hoomd
