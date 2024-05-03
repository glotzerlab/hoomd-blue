// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <string>

// Full text HOOMD version string
#define HOOMD_VERSION "${HOOMD_VERSION}"
// Set to the src dir to be used as the root to read data files from
#define HOOMD_SOURCE_DIR "${HOOMD_SOURCE_DIR}"
// Set to the binary dir
#define HOOMD_BINARY_DIR "${HOOMD_BINARY_DIR}"
// Set the installation directory as a hint for locating libraries
#define HOOMD_INSTALL_PREFIX "${HOOMD_INSTALL_PREFIX}"
// The python site directory
#define PYTHON_SITE_INSTALL_DIR "${PYTHON_SITE_INSTALL_DIR}"

// clang-format off

// hoomd major version
#define HOOMD_VERSION_MAJOR ${HOOMD_VERSION_MAJOR}
// hoomd minor version
#define HOOMD_VERSION_MINOR ${HOOMD_VERSION_MINOR}
// hoomd patch version
#define HOOMD_VERSION_PATCH ${HOOMD_VERSION_PATCH}

// clang-format on

/*! \file HOOMDVersion.h.in
    \brief Functions and variables for printing compile time build information of HOOMD
    \details This file is configured to HOOMDVersion.h by CMake, that is the file that should
        be included in any code needing it.

    \ingroup utils
*/

namespace hoomd
    {
/// Collect all build information together in one class
class BuildInfo
    {
    public:
    /// Return the version as a string
    static std::string getVersion();

    /// Format the HOOMD compilation options as a string
    static std::string getCompileFlags();

    /// Determine if ENABLE_GPU is set
    static bool getEnableGPU();

    /// Format the cuda API version as a string
    static std::string getGPUAPIVersion();

    /// Get the GPU platform
    static std::string getGPUPlatform();

    /// Get the c++ compiler description
    static std::string getCXXCompiler();

    /// Determine if ENABLE_TBB is set
    static bool getEnableTBB();

    /// Determine if ENABLE_MPI is set
    static bool getEnableMPI();

    /// Get the source directory
    static std::string getSourceDir();

    /// Get the installation directory
    static std::string getInstallDir();

    /// Get the floating point precision
    static std::pair<unsigned int, unsigned int> getFloatingPointPrecision();
    };
    } // namespace hoomd
