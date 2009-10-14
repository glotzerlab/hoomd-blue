# $Id$
# $URL$
# Maintainer: joaander

##################################
## Find CUDA
if (SINGLE_PRECISION)
    # add ENABLE_CUDA definition to compiler to enable #ifdef'd code
    if (ENABLE_CUDA)
        find_package(CUDA REQUIRED)

        mark_as_advanced(CUDA_SDK_ROOT_DIR)
        if (CUDA_TOOLKIT_ROOT_DIR)
            mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
        endif (CUDA_TOOLKIT_ROOT_DIR)
        mark_as_advanced(CUDA_VERBOSE_BUILD)

        if (CUDA_BUILD_EMULATION)
            add_definitions(-D_DEVICEEMU)
        endif (CUDA_BUILD_EMULATION)
    endif (ENABLE_CUDA)
endif (SINGLE_PRECISION)

