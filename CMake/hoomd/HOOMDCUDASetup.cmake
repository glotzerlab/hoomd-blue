# Maintainer: joaander

##################################
## Find CUDA
# If CUDA is enabled, set it up
if (ENABLE_CUDA)
	# the package is needed
	find_package(CUDA REQUIRED REQUIRED)

	if (${CUDA_VERSION} VERSION_LESS 5.0)
		message(SEND_ERROR "CUDA 5.0 or newer is required")
	endif (${CUDA_VERSION} VERSION_LESS 5.0)

    # Find Thrust
    find_package(Thrust)

    if (${THRUST_VERSION} VERSION_LESS 1.5.0)
        message(SEND_ERROR "Thrust version ${THRUST_VERSION} found, >= 1.5.0 is required")
    endif (${THRUST_VERSION} VERSION_LESS 1.5.0)

    # first thrust, then CUDA (to allow for local thrust installation
    # that overrides CUDA toolkit)
    include_directories(${THRUST_INCLUDE_DIR})

	include_directories(${CUDA_INCLUDE_DIRS})

    get_directory_property(DIRS INCLUDE_DIRECTORIES SYSTEM)
    # hide some variables users don't need to see
    mark_as_advanced(CUDA_SDK_ROOT_DIR)
    if (CUDA_TOOLKIT_ROOT_DIR)
        mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
    endif (CUDA_TOOLKIT_ROOT_DIR)
    mark_as_advanced(CUDA_VERBOSE_BUILD)
    mark_as_advanced(CUDA_BUILD_EMULATION)
    mark_as_advanced(CUDA_HOST_COMPILER)
    mark_as_advanced(CUDA_dl_LIBRARY)
    mark_as_advanced(CUDA_rt_LIBRARY)
    mark_as_advanced(THRUST_INCLUDE_DIR)

    if (ENABLE_NVTOOLS)
        find_library(CUDA_nvToolsExt_LIBRARY
                     NAMES nvToolsExt
                     PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                           "${CUDA_TOOLKIT_ROOT_DIR}/lib"
                     ENV CUDA_LIB_PATH
                     DOC "nvTools library"
                     NO_DEFAULT_PATH
                     )

        mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    endif()
endif (ENABLE_CUDA)

# setup CUDA compile options
if (ENABLE_CUDA)
    # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
    if (CUDA_VERSION VERSION_GREATER 5.99)
        set(CUDA_ARCH_LIST 20 30 35 50 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    else (CUDA_VERSION VERSION_GREATER 4.99)
        set(CUDA_ARCH_LIST 20 30 35 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    endif()

    foreach(_cuda_arch ${CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
    endforeach (_cuda_arch)

    # need to know the minumum supported CUDA_ARCH
    set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
    list(SORT _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
    add_definitions(-DCUDA_ARCH=${_cuda_min_arch})

    if (_cuda_min_arch LESS 20)
        message(SEND_ERROR "SM1x builds are not supported")
    endif ()

    # only generage ptx code for the maximum supported CUDA_ARCH (saves on file size)
    list(REVERSE _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_max_arch)
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_max_arch},code=compute_${_cuda_max_arch}")

# embed the CUDA libraries into the lib dir
if (ENABLE_EMBED_CUDA)
    # determine the directory of the found cuda libs
    get_filename_component(_cuda_libdir ${CUDA_CUDART_LIBRARY} PATH)
    FILE(GLOB _cuda_libs ${_cuda_libdir}/libcurand.* ${_cuda_libdir}/libcufft.* ${_cuda_libdir}/libcusolver.* ${_cuda_libdir}/libcusparse.*)
    install(PROGRAMS ${_cuda_libs} DESTINATION ${PYTHON_MODULE_BASE_DIR})
endif ()

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        # clang does not seem to work with host flag passing
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    endif()

endif (ENABLE_CUDA)

# automatically handle setting ccbin to /usr when needed
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    # CMAKE_CXX_COMPILER_VERSION is only available on 2.8.8 and newer

    # need to set ccbin to  when gcc is unsupported
    # this assumes that the user is on a system where CUDA is supported and /usr/bin/gcc will work - if they aren't, then it is their problem

    if (CUDA_VERSION VERSION_EQUAL 4.1)
        if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.5.99)
            message(STATUS "CUDA 4.1 doesn't support gcc >= 4.6, falling back on /usr/bin/gcc")
            list(APPEND CUDA_NVCC_FLAGS "-ccbin;/usr/bin/gcc")
        endif()
    endif()

    if (CUDA_VERSION VERSION_EQUAL 4.2)
        if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6.99)
            message(STATUS "CUDA 4.2 doesn't support gcc >= 4.7, falling back on /usr/bin/gcc")
            list(APPEND CUDA_NVCC_FLAGS "-ccbin;/usr/bin/gcc")
        endif()
    endif()

    if (CUDA_VERSION VERSION_EQUAL 5.0)
        if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6.99)
            message(STATUS "CUDA 5.0 doesn't support gcc >= 4.7, falling back on /usr/bin/gcc")
            list(APPEND CUDA_NVCC_FLAGS "-ccbin;/usr/bin/gcc")
        endif()
    endif()
endif()

# set CUSOLVER_AVAILABLE depending on CUDA Toolkit version
if (ENABLE_CUDA)
    if(${CUDA_VERSION} VERSION_LESS 7.5)
        set(CUSOLVER_AVAILABLE FALSE CACHE BOOL "TRUE if cusolver library is available")
    elseif(${CUDA_VERSION} VERSION_GREATER 7.5)
        set(CUSOLVER_AVAILABLE FALSE CACHE BOOL "TRUE if cusolver library is available")
    else()
        if (NOT CUSOLVER_AVAILABLE)
            # message at first time
            message(STATUS "CUDA version >= 7.5, looking for cusolver library")
            if (NOT ${CUDA_cusolver_LIBRARY} STREQUAL "")
                set(CUSOLVER_AVAILABLE TRUE CACHE BOOL "TRUE if cusolver library is available")
                message(STATUS "cuSolver library found.")
            else()
                set(CUSOLVER_AVAILABLE FALSE CACHE BOOL "TRUE if cusolver library is available")
                message(STATUS "Could not find cusolver library, constraints will be slower, perhaps old CMake?")
            endif()
        endif()
    endif()
    mark_as_advanced(CUSOLVER_AVAILABLE)
endif()
