# Maintainer: joaander

#################################
## Optional static build
## ENABLE_STATIC is an option to control whether HOOMD is built as a statically linked exe or as a python module.
OPTION(ENABLE_STATIC "Link as many libraries as possible statically, cannot be changed after the first run of CMake" OFF)

mark_as_advanced(ENABLE_STATIC)

#################################
## Optional single/double precision build
option(SINGLE_PRECISION "Use single precision math" OFF)

option(ENABLE_HPMC_MIXED_PRECISION "Enable mixed precision computations in HPMC" ON)
if (ENABLE_HPMC_MIXED_PRECISION)
    add_definitions(-DENABLE_HPMC_MIXED_PRECISION)
endif()

#####################3
## CUDA related options
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)

option(ALWAYS_USE_MANAGED_MEMORY "Use CUDA managed memory also when running on single GPU" OFF)
MARK_AS_ADVANCED(ALWAYS_USE_MANAGED_MEMORY)

if (ENABLE_CUDA)
    option(ENABLE_NVTOOLS "Enable NVTools profiler integration" off)
endif (ENABLE_CUDA)

############################
## MPI related options
option (ENABLE_MPI "Enable the compilation of the MPI communication code" off)

#################################
## Optionally enable documentation build
OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" OFF)
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # get the doxygen version
        exec_program(${DOXYGEN_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE DOXYGEN_VERSION)

        if (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
        else (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
            message(STATUS "Doxygen version less than 1.8.5, documentation may not build correctly")
        endif (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
    endif ()
endif ()

option(COPY_HEADERS "Copy the headers into the build directories for plugins" off)

###################################
## Components
option(BUILD_MD "Build the md package" on)
if (NOT SINGLE_PRECISION)
option(BUILD_HPMC "Build the hpmc package" on)
else ()
option(BUILD_HPMC "Build the hpmc package" off)
endif()
option(BUILD_DEPRECATED "Build the deprecated package" on)
option(BUILD_METAL "Build the metal package" on)
option(BUILD_DEM "Build the dem package" on)
option(BUILD_CGCMM "Build the cgcmm package" on)
option(BUILD_MPCD "Build the mpcd package" on)
option(BUILD_JIT "Build the jit package" off)

###############################
## In jenkins tests on multiple build configurations, it is wasteful to run CPU tests on CPU and all GPU test paths
## this option turns off CPU only tests in builds with ENABLE_CUDA=ON
option(TEST_CPU_IN_GPU_BUILDS "Test CPU code path in GPU enabled builds" on)
mark_as_advanced(TEST_CPU_IN_GPU_BUILDS)
if (NOT TEST_CPU_IN_GPU_BUILDS AND ENABLE_CUDA)
    message(STATUS "Warning: Skipping CPU tests")
endif()
