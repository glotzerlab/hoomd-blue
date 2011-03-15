# $Id$
# $URL$
# Maintainer: joaander

##################################
## Find CUDA
if (SINGLE_PRECISION)
    # If CUDA is enabled, set it up
    if (ENABLE_CUDA)
        # the package is needed
        find_package(CUDA REQUIRED REQUIRED)
        
        if (${CUDA_VERSION} VERSION_LESS 2.3)
            message(SEND_ERROR "CUDA 2.2 and older are not supported")
        endif (${CUDA_VERSION} VERSION_LESS 2.3)

        include_directories(${CUDA_INCLUDE_DIRS})

        # hide some variables users don't need to see
        mark_as_advanced(CUDA_SDK_ROOT_DIR)
        if (CUDA_TOOLKIT_ROOT_DIR)
            mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
        endif (CUDA_TOOLKIT_ROOT_DIR)
        mark_as_advanced(CUDA_VERBOSE_BUILD)
        mark_as_advanced(CUDA_BUILD_EMULATION)
    endif (ENABLE_CUDA)
endif (SINGLE_PRECISION)

# setup CUDA compile options
if (ENABLE_CUDA)
    # setup nvcc to build for all CUDA architectures. Allow user to modify the list if desired
    if (CUDA_VERSION VERSION_GREATER 2.99) 
        set(CUDA_ARCH_LIST 11 12 13 20 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    else (CUDA_VERSION VERSION_GREATER 2.99)
        set(CUDA_ARCH_LIST 11 12 13 CACHE STRING "List of target sm_ architectures to compile CUDA code for. Separate with semicolons.")
    endif (CUDA_VERSION VERSION_GREATER 2.99)
    
    foreach(_cuda_arch ${CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_arch},code=sm_${_cuda_arch}")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${_cuda_arch},code=compute_${_cuda_arch}")
    endforeach (_cuda_arch)

    if (CUDA_VERSION VERSION_EQUAL 3.1 OR CUDA_VERSION VERSION_EQUAL 3.2) 
        message(STATUS "Enabling reg usage workaround for CUDA 3.1/3.2") 
        list(APPEND CUDA_NVCC_FLAGS "-Xptxas;-abi=no")
    endif (CUDA_VERSION VERSION_EQUAL 3.1 OR CUDA_VERSION VERSION_EQUAL 3.2) 
    
    # need to know the minumum supported CUDA_ARCH
    set(_cuda_arch_list_sorted ${CUDA_ARCH_LIST})
    list(SORT _cuda_arch_list_sorted)
    list(GET _cuda_arch_list_sorted 0 _cuda_min_arch)
    add_definitions(-DCUDA_ARCH=${_cuda_min_arch})
endif (ENABLE_CUDA)
