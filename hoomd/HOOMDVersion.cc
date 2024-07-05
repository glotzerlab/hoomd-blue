// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HOOMDVersion.h"
#include <iostream>
#include <sstream>
#include <string>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace hoomd
    {
std::string BuildInfo::getCompileFlags()
    {
    std::ostringstream o;

#ifdef ENABLE_HIP
    int hip_major = HIP_VERSION_MAJOR;
    int hip_minor = HIP_VERSION_MINOR;

    o << "GPU [";
#if defined(__HIP_PLATFORM_NVCC__)
    o << "CUDA";
#elif defined(__HIP_PLATFORM_HCC__)
    o << "ROCm";
#endif
    o << "] (" << hip_major << "." << hip_minor << ") ";
#endif

#if HOOMD_LONGREAL_SIZE == 32
    o << "SINGLE";
#else
    o << "DOUBLE";
#endif

#if HOOMD_SHORTREAL_SIZE == 32
    o << "[SINGLE] ";
#else
    o << "[DOUBLE] ";
#endif

#ifdef ENABLE_MPI
    o << "MPI ";
#endif

#ifdef ENABLE_TBB
    o << "TBB ";
#endif

#ifdef __SSE__
    o << "SSE ";
#endif

#ifdef __SSE2__
    o << "SSE2 ";
#endif

#ifdef __SSE3__
    o << "SSE3 ";
#endif

#ifdef __SSE4_1__
    o << "SSE4_1 ";
#endif

#ifdef __SSE4_2__
    o << "SSE4_2 ";
#endif

#ifdef __AVX__
    o << "AVX ";
#endif

#ifdef __AVX2__
    o << "AVX2 ";
#endif

#ifdef ALWAYS_USE_MANAGED_MEMORY
    o << "ALWAYS_MANAGED ";
#endif

    return o.str();
    }

std::string BuildInfo::getVersion()
    {
    return std::string(HOOMD_VERSION);
    }

bool BuildInfo::getEnableGPU()
    {
#ifdef ENABLE_HIP
    return true;
#else
    return false;
#endif
    }

std::string BuildInfo::getGPUAPIVersion()
    {
#ifdef ENABLE_HIP
    int major = HIP_VERSION_MAJOR;
    int minor = HIP_VERSION_MINOR;
    std::ostringstream s;
    s << major << "." << minor;
    return s.str();
#else
    return "0.0";
#endif
    }

std::string BuildInfo::getGPUPlatform()
    {
#if defined(__HIP_PLATFORM_NVCC__)
    return std::string("CUDA");
#elif defined(__HIP_PLATFORM_HCC__)
    return std::string("ROCm");
#else
    return "";
#endif
    }

std::string BuildInfo::getCXXCompiler()
    {
#if defined(__GNUC__) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    std::ostringstream o;
    o << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return o.str();

#elif defined(__clang__)
    std::ostringstream o;
    o << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    return o.str();

#elif defined(__INTEL_COMPILER)
    std::ostringstream o;
    o << "icc " << __INTEL_COMPILER;
    return o.str();

#else
    return string("unknown");
#endif
    }

bool BuildInfo::getEnableTBB()
    {
#ifdef ENABLE_TBB
    return true;
#else
    return false;
#endif
    }

bool BuildInfo::getEnableMPI()
    {
#ifdef ENABLE_MPI
    return true;
#else
    return false;
#endif
    }

std::string BuildInfo::getSourceDir()
    {
    return std::string(HOOMD_SOURCE_DIR);
    }

std::string BuildInfo::getInstallDir()
    {
    return std::string(HOOMD_INSTALL_PREFIX) + "/" + std::string(PYTHON_SITE_INSTALL_DIR);
    }

std::pair<unsigned int, unsigned int> BuildInfo::getFloatingPointPrecision()
    {
    return std::make_pair(HOOMD_LONGREAL_SIZE, HOOMD_SHORTREAL_SIZE);
    }

    } // namespace hoomd
