option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)

option(ALWAYS_USE_MANAGED_MEMORY "Use CUDA managed memory also when running on single GPU" OFF)
MARK_AS_ADVANCED(ALWAYS_USE_MANAGED_MEMORY)

if (ENABLE_CUDA)
    option(ENABLE_NVTOOLS "Enable NVTools profiler integration" off)
endif (ENABLE_CUDA)

if (ENABLE_CUDA)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.0)
        message(SEND_ERROR "HOOMD-blue requires CUDA 9.0 or newer")
    endif()

    # find CUDA library path
    get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_LIB_PATH "${CUDA_BIN_PATH}/../lib64/" ABSOLUTE)

    # find libraries
    find_library(CUDA_cudart_LIBRARY cudart HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cudart_LIBRARY)
    find_library(CUDA_cudadevrt_LIBRARY cudadevrt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cudadevrt_LIBRARY)
    find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cufft_LIBRARY)
    find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cusolver_LIBRARY)
    find_library(CUDA_cusparse_LIBRARY cusparse HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cusparse_LIBRARY)

    # find cuda-memcheck
    find_program(CUDA_MEMCHECK_EXECUTABLE
      NAMES cuda-memcheck
      HINTS "${CUDA_BIN_PATH}"
      NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    list(APPEND HOOMD_COMMON_LIBS ${CUDA_cudart_LIBRARY} ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})

    add_definitions (-DENABLE_CUDA)

    if(ALWAYS_USE_MANAGED_MEMORY)
        add_definitions(-DALWAYS_USE_MANAGED_MEMORY)
    endif()

    if (ENABLE_NVTOOLS)
        list(APPEND HOOMD_COMMON_LIBS ${CUDA_nvToolsExt_LIBRARY})
        add_definitions(-DENABLE_NVTOOLS)
    endif()
endif (ENABLE_CUDA)

# setup CUDA compile options
if (ENABLE_CUDA)
    # supress warnings in random123
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=code_is_unreachable")

    # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 8.99)
        set(CUDA_ARCH_LIST 35 50 60 70 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 7.99)
        set(CUDA_ARCH_LIST 35 50 60 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    endif()

    foreach(_cuda_arch ${CUDA_ARCH_LIST})
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
    endforeach (_cuda_arch)

    # need to know the minimum supported CUDA_ARCH
    set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
    list(SORT _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
    list(GET _cuda_arch_list_sorted -1 _cuda_max_arch)
    add_definitions(-DCUDA_ARCH=${_cuda_min_arch})

    if (_cuda_min_arch LESS 35)
        message(SEND_ERROR "HOOMD requires compute 3.5 or newer")
    endif ()

    # only generate ptx code for the maximum supported CUDA_ARCH (saves on file size)
    list(REVERSE _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_max_arch)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${_cuda_max_arch},code=compute_${_cuda_max_arch}")

endif (ENABLE_CUDA)

# set CUSOLVER_AVAILABLE depending on CUDA Toolkit version
if (ENABLE_CUDA)
    # CUDA 8.0 requires that libgomp be linked in - see if we can link it
    try_compile(_can_link_gomp
                ${CMAKE_CURRENT_BINARY_DIR}/tmp
                ${CMAKE_CURRENT_LIST_DIR}/test.cc
                LINK_LIBRARIES gomp
               )

    if (NOT ${CUDA_cusolver_LIBRARY} STREQUAL "" AND _can_link_gomp)
        set(CUSOLVER_AVAILABLE TRUE)
    else()
        set(CUSOLVER_AVAILABLE FALSE)
    endif()

if (NOT CUSOLVER_AVAILABLE)
    message(STATUS "Could not find cusolver library, constraints will be slower. Perhaps old CMake or missing gomp library.")
endif()

if (CUSOLVER_AVAILABLE)
    add_definitions(-DCUSOLVER_AVAILABLE)
endif()

endif()

###############################
# Helper macros
macro(fix_cudart_rpath target)
if (ENABLE_CUDA AND APPLE)
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcudart.dylib ${CUDA_CUDART_LIBRARY} ${_target_exe})
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcufft.dylib ${CUDA_cufft_LIBRARY} ${_target_exe})
endif (ENABLE_CUDA AND APPLE)
endmacro(fix_cudart_rpath)
