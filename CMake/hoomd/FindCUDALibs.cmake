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

if (HIP_PLATFORM STREQUAL "hcc")
    # find libraries that go with this compiler
    find_library(HIP_hip_hcc_LIBRARY hip_hcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)

    find_library(HIP_hip_hcc_LIBRARY hip_hcc HINTS ${HIP_LIB_PATH} NO_DEFAULT_PATH)
    mark_as_advanced(HIP_hip_hcc_LIBRARY)
    if(HIP_hip_hcc_LIBRARY AND NOT TARGET HIP::hip_hcc)
      add_library(HIP::hip_hcc UNKNOWN IMPORTED)
      set_target_properties(HIP::hip_hcc PROPERTIES
        IMPORTED_LOCATION "${HIP_hip_hcc_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    list(APPEND REQUIRED_HIP_LIB_VARS "HIP_hip_hcc_LIBRARY")
else()
    # define empty target
    add_library(HIP::hip_hcc UNKNOWN IMPORTED)
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
    list(add REQUIRED_CUDA_LIB_VARS CUDA_nvToolsExt_LIBRARY)
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
    list(add REQUIRED_CUDA_LIB_VARS CUDA_cusolver_LIBRARY)
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
    list(add REQUIRED_CUDA_LIB_VARS CUDA_cusparse_LIBRARY)
else()
    # cusparse not supported by HIP (?)
    add_library(CUDA::cusparse UNKNOWN IMPORTED)
endif()

find_library(HIP_hiprand_LIBRARY hiprand
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hiprand
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)
find_path(HIP_hiprand_INCLUDE_DIR
    NAMES hiprand.h
    PATHS
    ${HIP_ROOT_DIR}/hiprand/include
    $ENV{ROCM_PATH}/hiprand/include
    $ENV{HIP_PATH}/hiprand/include
    /opt/rocm/include
    /opt/rocm/hiprand/include
    NO_DEFAULT_PATH)
mark_as_advanced(HIP_hiprand_LIBRARY)
list(APPEND REQUIRED_CUDA_LIB_VARS HIP_hiprand_LIBRARY)
list(APPEND _hiprand_includes ${HIP_hiprand_INCLUDE_DIR})

if(HIP_hiprand_LIBRARY AND NOT TARGET HIP::hiprand)
  add_library(HIP::hiprand UNKNOWN IMPORTED)
  set_target_properties(HIP::hiprand PROPERTIES
    IMPORTED_LOCATION "${HIP_hiprand_LIBRARY}"
    )
endif()

if(HIP_PLATFORM STREQUAL "hcc")
    find_library(HIP_rocrand_LIBRARY rocrand
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/rocrand
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    find_path(HIP_rocrand_INCLUDE_DIR
        NAMES rocrand.h
        PATHS
        ${HIP_ROOT_DIR}/rocrand/include
        $ENV{ROCM_PATH}/rocrand/include
        $ENV{HIP_PATH}/rocrand/include
        /opt/rocm/include
        /opt/rocm/rocrand/include
        NO_DEFAULT_PATH)

    list(APPEND _hiprand_includes ${HIP_rocrand_INCLUDE_DIR})
    mark_as_advanced(HIP_rocrand_LIBRARY)
    set_target_properties(HIP::hiprand PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_hiprand_includes}"
        INTERFACE_LINK_LIBRARIES "${HIP_rocrand_LIBRARY}"
        )
    list(APPEND REQUIRED_CUDA_LIB_VARS HIP_rocrand_LIBRARY)
endif()

find_path(HIP_hipfft_INCLUDE_DIR
    NAMES hipfft.h
    PATHS
    ${HIP_ROOT_DIR}/hipfft/include
    $ENV{ROCM_PATH}/hipfft/include
    $ENV{HIP_PATH}/hipfft/include
    /opt/rocm/include
    /opt/rocm/hipfft/include
    NO_DEFAULT_PATH)

list(APPEND REQUIRED_CUDA_LIB_VARS HIP_hipfft_INCLUDE_DIR)

if(HIP_PLATFORM STREQUAL "hcc")
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
        ${HIP_ROOT_DIR}/rocfft/include
        $ENV{ROCM_PATH}/rocfft/include
        $ENV{HIP_PATH}/rocfft/include
        /opt/rocm/include
        /opt/rocm/rocfft/include
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

if (HIP_PLATFORM STREQUAL "nvcc")
    # find cuda-memcheck
    find_program(CUDA_MEMCHECK_EXECUTABLE
      NAMES cuda-memcheck
      HINTS "${CUDA_BIN_PATH}"
      NO_DEFAULT_PATH)
    mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)
    list(add REQUIRED_CUDA_LIB_VARS CUDA_MEMCHECK_EXECUTABLE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDALibs
  REQUIRED_CUDA_LIB_VARS
    ${REQUIRED_CUDA_LIB_VARS}
)
