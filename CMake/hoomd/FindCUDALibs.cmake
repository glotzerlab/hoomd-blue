# Find CUDA libraries and binaries used by HOOMD

set(REQUIRED_CUDA_LIB_VARS "")
if (HIP_PLATFORM STREQUAL "nvcc")
    # find CUDA library path
    get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_LIB_PATH "${CUDA_BIN_PATH}/../lib64/" ABSOLUTE)

    # find libraries that go with this compiler
    find_library(CUDA_cudart_LIBRARY cudart HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_cudart_LIBRARY)
    if(CUDA_cudart_LIBRARY AND NOT TARGET CUDA::cudart)
      add_library(CUDA::cudart UNKNOWN IMPORTED)
      set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "${CUDA_cudart_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsCUDART "Found cudart: ${CUDA_cudart_LIBRARY}" "[${CUDA_cudart_LIBRARY}]")
    list(APPEND REQUIRED_CUDA_LIB_VARS "CUDA_cudart_LIBRARY")
else()
    # define empty target
    add_library(CUDA::cudart UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cudadevrt_LIBRARY cudadevrt HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_cudadevrt_LIBRARY)
    if(CUDA_cudadevrt_LIBRARY AND NOT TARGET CUDA::cudadevrt)
      add_library(CUDA::cudadevrt UNKNOWN IMPORTED)
      set_target_properties(CUDA::cudadevrt PROPERTIES
        IMPORTED_LOCATION "${CUDA_cudadevrt_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
else()
    # separable compilation not supported with HIP
    add_library(CUDA::cudadevrt UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    # find libraries that go with this compiler
    find_library(CUDA_cuda_LIBRARY cuda HINTS ${CUDA_LIB_PATH}
                 /usr/local/cuda-10.1/compat
                 /usr/local/cuda-10.2/compat
                 /usr/local/cuda-11.1/compat
                 /usr/local/cuda-11.2/compat
                 /usr/local/cuda-11.3/compat
                 /usr/local/cuda-11.4/compat
                 /usr/local/cuda-11.5/compat
                 /usr/local/cuda-11.6/compat
                 /usr/local/cuda-11.7/compat
                 /usr/local/cuda-11.8/compat
                 /usr/local/cuda-11.9/compat
                 /usr/local/cuda-11.10/compat
                 /usr/local/cuda-12.0/compat
                 /usr/local/cuda-12.1/compat
                 /usr/local/cuda-12.2/compat
                 /usr/local/cuda-12.3/compat
                 /usr/local/cuda-12.4/compat
                 /usr/local/cuda-12.5/compat
                 /usr/local/cuda-12.6/compat
                 /usr/local/cuda-12.7/compat
                 /usr/local/cuda-12.8/compat
                 /usr/local/cuda-12.9/compat
                 /usr/local/cuda-12.10/compat)
    mark_as_advanced(CUDA_cuda_LIBRARY)
    if(CUDA_cuda_LIBRARY AND NOT TARGET CUDA::cuda)
      add_library(CUDA::cuda UNKNOWN IMPORTED)
      set_target_properties(CUDA::cuda PROPERTIES
        IMPORTED_LOCATION "${CUDA_cuda_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsCUDA "Found cuda: ${CUDA_cuda_LIBRARY}" "[${CUDA_cuda_LIBRARY}]")
    list(APPEND REQUIRED_CUDA_LIB_VARS "CUDA_cuda_LIBRARY")
else()
    # define empty target
    add_library(CUDA::cuda UNKNOWN IMPORTED)
endif()

endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_cufft_LIBRARY)
    if(CUDA_cufft_LIBRARY AND NOT TARGET CUDA::cufft)
      add_library(CUDA::cufft UNKNOWN IMPORTED)
      set_target_properties(CUDA::cufft PROPERTIES
        IMPORTED_LOCATION "${CUDA_cufft_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsCUFFT "Found cufft: ${CUDA_cufft_LIBRARY}" "[${CUDA_cufft_LIBRARY}]")
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cufft_LIBRARY)
else()
    # cufft API is supported natively by HIP
    add_library(CUDA::cufft UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc" AND ENABLE_NVTOOLS)
    find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
    if(CUDA_nvToolsExt_LIBRARY AND NOT TARGET CUDA::nvToolsExt)
      add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION "${CUDA_nvToolsExt_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsNVToolsExt "Found nvToolsExt: ${CUDA_nvToolsExt_LIBRARY}" "$[{CUDA_nvToolsExt_LIBRARY}]")
else()
    # nvtools not supported by HIP
    add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_cusolver_LIBRARY)
    if(CUDA_cusolver_LIBRARY AND NOT TARGET CUDA::cusolver)
      add_library(CUDA::cusolver UNKNOWN IMPORTED)
      set_target_properties(CUDA::cusolver PROPERTIES
        IMPORTED_LOCATION "${CUDA_cusolver_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsCUSolver "Found cusolver: ${CUDA_cusolver_LIBRARY}" "[${CUDA_cusolver_LIBRARY}]")
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cusolver_LIBRARY)
else()
    # cusolver not offered by HIP (?)
    add_library(CUDA::cusolver UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    find_library(CUDA_cusparse_LIBRARY cusparse HINTS ${CUDA_LIB_PATH})
    mark_as_advanced(CUDA_cusparse_LIBRARY)
    if(CUDA_cusparse_LIBRARY AND NOT TARGET CUDA::cusparse)
      add_library(CUDA::cusparse UNKNOWN IMPORTED)
      set_target_properties(CUDA::cusparse PROPERTIES
        IMPORTED_LOCATION "${CUDA_cusparse_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
      )
    endif()
    find_package_message(CUDALibsCUSparse "Found cusparse: ${CUDA_cusparse_LIBRARY}" "[${CUDA_cusparse_LIBRARY}]")
    list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cusparse_LIBRARY)
else()
    # cusparse not supported by HIP (?)
    add_library(CUDA::cusparse UNKNOWN IMPORTED)
endif()

if (HIP_PLATFORM STREQUAL "hip-clang")
    find_package(hipfft)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    # find compute-sanitizer / cuda-memcheck
    find_program(CUDA_MEMCHECK_EXECUTABLE
      NAMES compute-sanitizer
      HINTS "${CUDA_BIN_PATH}" "${CUDA_BIN_PATH}../../extras/compute-sanitizer"
      NO_DEFAULT_PATH)

    # Fall back on cuda-memcheck when compute-sanitizer is not available (CUDA 10)
    if (NOT CUDA_MEMCHECK_EXECUTABLE)
        find_program(CUDA_MEMCHECK_EXECUTABLE
          NAMES cuda-memcheck
          HINTS "${CUDA_BIN_PATH}"
          NO_DEFAULT_PATH)
    endif()

    find_package_message(CUDALibsMemcheck "Found compute-sanitizer: ${CUDA_MEMCHECK_EXECUTABLE}" "[${CUDA_MEMCHECK_EXECUTABLE}]")
    mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)
endif()

if (HIP_PLATFORM STREQUAL "nvcc")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(CUDALibs
      REQUIRED_VARS
        ${REQUIRED_CUDA_LIB_VARS}
    )
endif()
