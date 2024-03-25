// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Filesystem.h
    \brief Helper functions for working with the file system
*/

#include <sys/stat.h>

namespace hoomd
    {
namespace filesystem
    {
//! Test if a file exists
/*! \param name File name
 */
inline bool exists(const std::string& name)
    {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
    }
    } // namespace filesystem
    } // end namespace hoomd
