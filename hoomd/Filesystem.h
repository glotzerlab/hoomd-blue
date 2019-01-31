// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Filesystem.h
    \brief Helper functions for working with the file system
*/

#include <sys/stat.h>

namespace filesystem {

//! Test if a file exists
/*! \param name File name
*/
inline bool exists(const std::string& name)
    {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
    }
}
