# Find CUDA libraries and binaries used by HOOMD

# find CUDA library path
get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_LIB_PATH "${CUDA_BIN_PATH}/../lib64/" ABSOLUTE)

set(REQUIRED_CUDA_LIB_VARS "")
if (HIP_PLATFORM STREQUAL "nvcc")
    # find libraries that go with this compiler
    find_library(CUDA_cudart_LIBRARY cudart HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cudart_LIBRARY)
    if(CUDA_cudart_LIBRARY AND NOT TARGET CUDA::cudart)
      add_library(CUDA::cudart UNKNOWN IMPORTED)
      set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "${CUDA_cudart_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS "CUDA_cudart_LIBRARY")
else()
    # define empty target
    add_library(CUDA::cudart UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "hip-clang" OR HIP_PLATFORM STREQUAL "hcc")
    # find libraries that go with this compiler
    find_library(HIP_hip_hcc_LIBRARY hip_hcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    mark_as_advanced(HIP_hip_hcc_LIBRARY)
    find_library(HIP_hiprtc_LIBRARY hiprtc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    mark_as_advanced(HIP_hiprtc_LIBRARY)

    if(HIP_hip_hcc_LIBRARY AND NOT TARGET HIP::hiprt)
      add_library(HIP::hiprt UNKNOWN IMPORTED)
      set_target_properties(HIP::hiprt PROPERTIES
        IMPORTED_LOCATION "${HIP_hip_hcc_LIBRARY}"
        INTERFACE_LINK_LIBRARIES ${HIP_hiprtc_LIBRARY}
      )
    endif()
    list(APPEND REQUIRED_HIP_LIB_VARS "HIP_hip_hcc_LIBRARY")
    list(APPEND REQUIRED_HIP_LIB_VARS "HIP_hiprtc_LIBRARY")
else()
    # define empty target
    add_library(HIP::hiprt UNKNOWN IMPORTED)
endif()


if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cudadevrt_LIBRARY cudadevrt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cudadevrt_LIBRARY)
    if(CUDA_cudadevrt_LIBRARY AND NOT TARGET CUDA::cudadevrt)
      add_library(CUDA::cudadevrt UNKNOWN IMPORTED)
      set_target_properties(CUDA::cudadevrt PROPERTIES
        IMPORTED_LOCATION "${CUDA_cudadevrt_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cudadevrt_LIBRARY)  
else()
    # separable compilation not supported with HIP
    add_library(CUDA::cudadevrt UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cufft_LIBRARY)
    if(CUDA_cufft_LIBRARY AND NOT TARGET CUDA::cufft)
      add_library(CUDA::cufft UNKNOWN IMPORTED)
      set_target_properties(CUDA::cufft PROPERTIES
        IMPORTED_LOCATION "${CUDA_cufft_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cufft_LIBRARY)
else()
    # cufft API is supported natively by HIP
    add_library(CUDA::cufft UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    if(CUDA_nvToolsExt_LIBRARY AND NOT TARGET CUDA::nvToolsExt)
      add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION "${CUDA_nvToolsExt_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_nvToolsExt_LIBRARY)
else()
    # nvtools not supported by HIP
    add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cusolver_LIBRARY)
    if(CUDA_cusolver_LIBRARY AND NOT TARGET CUDA::cusolver)
      add_library(CUDA::cusolver UNKNOWN IMPORTED)
      set_target_properties(CUDA::cusolver PROPERTIES
        IMPORTED_LOCATION "${CUDA_cusolver_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cusolver_LIBRARY)
else()
    # cusolver not offered by HIP (?)
    add_library(CUDA::cusolver UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cusparse_LIBRARY cusparse HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_cusparse_LIBRARY)
    if(CUDA_cusparse_LIBRARY AND NOT TARGET CUDA::cusparse)
      add_library(CUDA::cusparse UNKNOWN IMPORTED)
      set_target_properties(CUDA::cusparse PROPERTIES
        IMPORTED_LOCATION "${CUDA_cusparse_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cusparse_LIBRARY)
else()
    # cusparse not supported by HIP (?)
    add_library(CUDA::cusparse UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "hip-clang" OR HIP_PLATFORM STREQUAL "hcc")
    find_path(HIP_hipfft_INCLUDE_DIR
        NAMES hipfft.h
        PATHS
        ${HIP_ROOT_DIR}/rocfft/include
        $ENV{ROCM_PATH}/hipfft/include
        $ENV{HIP_PATH}/hipfft/include
        /opt/rocm/include
        /opt/rocm/hipfft/include
        NO_DEFAULT_PATH)

    list(APPEND REQUIRED_CUDA_LIB_VARS HIP_hipfft_INCLUDE_DIR)

    find_library(HIP_rocfft_LIBRARY rocfft
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/rocfft
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)

    find_path(HIP_rocfft_INCLUDE_DIR
        NAMES rocfft.h
        PATHS
        ${HIP_ROOT_DIR}/rocfft
        $ENV{ROCM_PATH}/rocfft
        $ENV{HIP_PATH}/rocfft
        /opt/rocm
        /opt/rocm/rocfft
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)

    mark_as_advanced(HIP_rocfft_LIBRARY)
    if(HIP_rocfft_LIBRARY AND NOT TARGET HIP::hipfft)
      add_library(HIP::hipfft UNKNOWN IMPORTED)
      set_target_properties(HIP::hipfft PROPERTIES
        IMPORTED_LOCATION "${HIP_rocfft_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_hipfft_INCLUDE_DIR};${HIP_rocfft_INCLUDE_DIR}"
        )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS HIP_rocfft_LIBRARY)
endif()

if (HIP_PLATFORM STREQUAL "hip-clang" OR HIP_PLATFORM STREQUAL "hcc")
    find_library(HIP_roctracer_LIBRARY roctracer64
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/roctracer
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)

    find_path(HIP_roctracer_INCLUDE_DIR
        NAMES roctracer.h
        PATHS
        ${HIP_ROOT_DIR}/roctracer
        $ENV{ROCM_PATH}/roctracer
        $ENV{HIP_PATH}/roctracer
        /opt/rocm
        /opt/rocm/roctracer
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)

    mark_as_advanced(HIP_roctracer_LIBRARY)
    if(HIP_roctracer_LIBRARY AND NOT TARGET HIP::roctracer)
      add_library(HIP::roctracer UNKNOWN IMPORTED)
      set_target_properties(HIP::roctracer PROPERTIES
        IMPORTED_LOCATION "${HIP_roctracer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_roctracer_INCLUDE_DIR};${HIP_roctracer_INCLUDE_DIR}"
        )
    endif()
    list(APPEND REQUIRED_CUDA_LIB_VARS HIP_roctracer_LIBRARY)
endif()


#find_library(HIP_hipsparse_LIBRARY hipsparse
#    PATHS
#    "${HIP_ROOT_DIR}"
#    ENV ROCM_PATH
#    ENV HIP_PATH
#    /opt/rocm
#    /opt/rocm/hipsparse
#    PATH_SUFFIXES lib
#    NO_DEFAULT_PATH)
#find_path(HIP_hipsparse_INCLUDE_DIR
#    NAMES hipsparse.h
#    PATHS
#    ${HIP_ROOT_DIR}/hipsparse/include
#    $ENV{ROCM_PATH}/hipsparse/include
#    $ENV{HIP_PATH}/hipsparse/include
#    /opt/rocm/include
#    /opt/rocm/hipsparse/include
#    NO_DEFAULT_PATH)
#mark_as_advanced(HIP_hipsparse_LIBRARY)
#list(APPEND REQUIRED_CUDA_LIB_VARS HIP_hipsparse_LIBRARY)
#list(APPEND _hipsparse_includes ${HIP_hipsparse_INCLUDE_DIR})

#if(HIP_hipsparse_LIBRARY AND NOT TARGET HIP::hipsparse)
#  add_library(HIP::hipsparse UNKNOWN IMPORTED)
#  set_target_properties(HIP::hipsparse PROPERTIES
#    IMPORTED_LOCATION "${HIP_hipsparse_LIBRARY}"
#    )
#endif()

#if (HIP_PLATFORM STREQUAL "hip-clang" OR HIP_PLATFORM STREQUAL "hcc")
#    find_library(HIP_rocsparse_LIBRARY rocsparse
#        PATHS
#        "${HIP_ROOT_DIR}"
#        ENV ROCM_PATH
#        ENV HIP_PATH
#        /opt/rocm
#        /opt/rocm/rocsparse
#        PATH_SUFFIXES lib
#        NO_DEFAULT_PATH)
#    find_path(HIP_rocsparse_INCLUDE_DIR
#        NAMES rocsparse.h
#        PATHS
#        ${HIP_ROOT_DIR}/rocsparse/include
#        $ENV{ROCM_PATH}/rocsparse/include
#        $ENV{HIP_PATH}/rocsparse/include
#        /opt/rocm/include
#        /opt/rocm/rocsparse/include
#        NO_DEFAULT_PATH)
#
#    list(APPEND _hipsparse_includes ${HIP_rocsparse_INCLUDE_DIR})
#    mark_as_advanced(HIP_rocsparse_LIBRARY)
#    set_target_properties(HIP::hipsparse PROPERTIES
#        INTERFACE_INCLUDE_DIRECTORIES "${_hipsparse_includes}"
#        INTERFACE_LINK_LIBRARIES "${HIP_rocsparse_LIBRARY}"
#        )
#    list(APPEND REQUIRED_CUDA_LIB_VARS HIP_rocsparse_LIBRARY)
#endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    # find cuda-memcheck
    find_program(CUDA_MEMCHECK_EXECUTABLE
      NAMES cuda-memcheck
      HINTS "${CUDA_BIN_PATH}"
      NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_MEMCHECK_EXECUTABLE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDALibs
  REQUIRED_CUDA_LIB_VARS
    ${REQUIRED_CUDA_LIB_VARS}
    ${REQUIRED_HIP_LIB_VARS}
)
