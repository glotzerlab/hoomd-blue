# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pathlib


def _get_hoomd_include_path():
    """Get the base directory for the HOOMD source include path."""
    current_module_path = pathlib.Path(hoomd.__file__).parent.resolve()
    build_module_path = (pathlib.Path(hoomd.version.build_dir)
                         / 'hoomd').resolve()

    # use the source directory if this module is in the build directory
    if current_module_path == build_module_path:
        hoomd_include_path = pathlib.Path(hoomd.version.source_dir)
    else:
        # otherwise, use the installation directory
        hoomd_include_path = current_module_path / 'include'

    return hoomd_include_path.resolve()


def get_cpu_compiler_arguments():
    """Get the arguments to pass to the compiler.

    These arguments must include the include patch for HOOMD's include files.
    """
    return ['-I', str(_get_hoomd_include_path()), '-O3']


def get_gpu_compilation_settings(gpu):
    """Helper function to set CUDA libraries for GPU execution."""
    hoomd_include_path = _get_hoomd_include_path()

    includes = [
        "-I" + str(hoomd_include_path),
        "-I" + str(hoomd_include_path / 'hoomd' / 'extern' / 'HIP' / 'include'),
        "-I" + str(hoomd.version.cuda_include_path),
    ]

    # compile JIT code for the current device
    compute_major, compute_minor = gpu.compute_capability
    return {
        "includes": includes,
        "cuda_devrt_lib_path": hoomd.version.cuda_devrt_library,
        "max_arch": compute_major * 10 + compute_minor
    }
