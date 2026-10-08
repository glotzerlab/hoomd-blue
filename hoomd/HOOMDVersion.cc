// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HOOMDVersion.h"
#include <iostream>
#include <sstream>
#include <string>

#ifdef ENABLE_GPU
#include <cuda_runtime.h>
#endif

#define CUDA_VERSION_MAJOR (CUDART_VERSION / 1000)
#define CUDA_VERSION_MINOR ((CUDART_VERSION % 1000) / 10)

namespace hoomd
    {
std::string BuildInfo::getCompileFlags()
    {
    std::ostringstream o;

#ifdef ENABLE_GPU
    int major = CUDA_VERSION_MAJOR;
    int minor = CUDA_VERSION_MINOR;

    o << "GPU [";
    o << "CUDA";
    o << "] (" << major << "." << minor << ") ";
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

    return o.str();
    }

std::string BuildInfo::getVersion()
    {
    return std::string(HOOMD_VERSION);
    }

bool BuildInfo::getEnableGPU()
    {
#ifdef ENABLE_GPU
    return true;
#else
    return false;
#endif
    }

std::string BuildInfo::getGPUAPIVersion()
    {
#ifdef ENABLE_GPU
    int major = CUDA_VERSION_MAJOR;
    int minor = CUDA_VERSION_MINOR;
    std::ostringstream s;
    s << major << "." << minor;
    return s.str();
#else
    return "0.0";
#endif
    }

std::string BuildInfo::getGPUPlatform()
    {
#if ENABLE_GPU
    return std::string("CUDA");
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
