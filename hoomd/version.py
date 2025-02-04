# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Version and build information.

Use the values in `hoomd.version` to query properties of the package set at
compile time.

See Also:
    `features`

    `building`

Attributes:
    build_dir (str): The directory where this build was compiled.

    compile_date (str): The date this build was compiled.

    compile_flags (str): Human readable summary of compilation flags.

    cuda_include_path (str): CUDA toolkit include directory.

    cuda_devrt_library (str): CUDA devrt library.

    cxx_compiler (str): Name and version of the C++ compiler used to build
        HOOMD.

    floating_point_precision (tuple[int, int]): The **high precision** floating
        point width in bits  (element 0) and the **reduced precision** width in
        bits (element 1).

    git_branch (str):  Name of the git branch used when compiling this build.

    git_sha1 (str):  SHA1 of the git commit used when compiling this build.

    gpu_api_version (str): The GPU API version this build was compiled against.

    gpu_enabled (bool): ``True`` when this build supports GPUs.

    gpu_platform (str): Name of the GPU platform this build was compiled
        against.

    hpmc_built (bool): ``True`` when the ``hpmc`` component is built.

    install_dir (str): The installation directory.

    llvm_enabled (bool): ``True`` when this build supports LLVM run time
        compilation.

    metal_built (bool): ``True`` when the ``metal`` component is built.

    md_built (bool): ``True`` when the `md` component is built.

    mpcd_built (bool): ``True`` when the ``mpcd`` component is built.

    mpi_enabled (bool): ``True`` when this build supports MPI parallel runs.

    source_dir (str): The source directory.

    tbb_enabled (bool): ``True`` when this build supports TBB threads.

    version (str): HOOMD-blue package version, following semantic versioning.
"""
from hoomd import _hoomd

from hoomd.version_config import (
    build_dir,
    compile_date,
    cuda_include_path,
    cuda_devrt_library,
    git_branch,
    git_sha1,
    hpmc_built,
    llvm_enabled,
    md_built,
    metal_built,
    mpcd_built,
)

version = _hoomd.BuildInfo.getVersion()
compile_flags = _hoomd.BuildInfo.getCompileFlags()
gpu_enabled = _hoomd.BuildInfo.getEnableGPU()
gpu_api_version = _hoomd.BuildInfo.getGPUAPIVersion()
gpu_platform = _hoomd.BuildInfo.getGPUPlatform()
cxx_compiler = _hoomd.BuildInfo.getCXXCompiler()
tbb_enabled = _hoomd.BuildInfo.getEnableTBB()
mpi_enabled = _hoomd.BuildInfo.getEnableMPI()
source_dir = _hoomd.BuildInfo.getSourceDir()
install_dir = _hoomd.BuildInfo.getInstallDir()
floating_point_precision = _hoomd.BuildInfo.getFloatingPointPrecision()
