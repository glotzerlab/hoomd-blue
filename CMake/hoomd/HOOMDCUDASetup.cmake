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

        # define _DEVICEEMU in emulation mode
        if (CUDA_BUILD_EMULATION)
            add_definitions(-D_DEVICEEMU)
        endif (CUDA_BUILD_EMULATION)
    endif (ENABLE_CUDA)
endif (SINGLE_PRECISION)

