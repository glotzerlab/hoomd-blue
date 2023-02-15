// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/VirtualSiteGPU.cuh"
#include "hoomd/md/VirtualSiteMappings.h"

#define VS_CLASS hoomd::md::virtualsites::@_vstype@
#define EXPORT_FUNCTION export_virtual_site@_vstype@

// clang-format on

namespace hoomd
{
    namespace md
    {
        namespace kernel
        {
            template __attribute__((visibility("default"))) hipError_t
                    gpu_update_virtual_sites<VS_CLASS>(const VirtualSiteUpdateKernelArgs& args, const typename VS_CLASS::param_type params);

            template __attribute__((visibility("default"))) hipError_t
                    gpu_decompose_virtual_sites<VS_CLASS>(const VirtualSiteDecomposeKernelArgs& args, const typename VS_CLASS::param_type params);
        } // end namespace kernel
    } // end namespace md
} // end namespace hoomd
