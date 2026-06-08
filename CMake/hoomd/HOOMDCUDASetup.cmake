# setup CUDA compile options
if (ENABLE_HIP)
    if (HIP_PLATFORM STREQUAL "nvcc")
        # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            set(CUDA_ARCH_LIST 80 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
        elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
            set(CUDA_ARCH_LIST 60 70 80 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
        endif()

        # ignore warnings about unused results
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -diag-suppress 2810")

        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUSPARSE_NEW_API")
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUSPARSE_NEW_API")
        endif()

        # need to know the minimum supported CUDA_ARCH
        set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
        list(SORT _cuda_arch_list_sorted)
        list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
        list(GET _cuda_arch_list_sorted -1 _cuda_max_arch)

        if (_cuda_min_arch LESS 60)
            message(SEND_ERROR "HOOMD requires compute 6.0 or newer")
        endif()

        # only generate ptx code for the maximum supported CUDA_ARCH (saves on file size)
        list(REVERSE _cuda_arch_list_sorted)
        list(GET _cuda_arch_list_sorted 0 _cuda_max_arch)

        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
            # CMAKE 3.18 handles CUDA ARCHITECTURES with CMAKE_CUDA_ARCHITECTURES
            set(CMAKE_CUDA_ARCHITECTURES "")
            foreach(_cuda_arch ${CUDA_ARCH_LIST})
                list(APPEND CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}-real")
            endforeach()
            list(APPEND CMAKE_CUDA_ARCHITECTURES "${_cuda_max_arch}-virtual")
        else()
            foreach(_cuda_arch ${CUDA_ARCH_LIST})
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
            endforeach()
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${_cuda_max_arch},code=compute_${_cuda_max_arch}")
        endif()

    elseif(HIP_PLATFORM STREQUAL "amd")
        set(_cuda_min_arch 35)

        # ignore warnings about unused results and set HIP_PLATFORM_HCC (which was previously
        # set by rocm < 6.0.0)
        set(CMAKE_HIP_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -D__HIP_PLATFORM_HCC__")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")
    endif()
endif (ENABLE_HIP)

# set CUSOLVER_AVAILABLE depending on CUDA Toolkit version
if (ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc")
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

endif()
