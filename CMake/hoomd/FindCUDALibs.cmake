# Find CUDA libraries and binaries used by HOOMD

# find CUDA library path
get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_LIB_PATH "${CUDA_BIN_PATH}/../lib64/" ABSOLUTE)

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

find_library(CUDA_cudadevrt_LIBRARY cudadevrt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
mark_as_advanced(CUDA_cudadevrt_LIBRARY)
if(CUDA_cudadevrt_LIBRARY AND NOT TARGET CUDA::cudadevrt)
  add_library(CUDA::cudadevrt UNKNOWN IMPORTED)
  set_target_properties(CUDA::cudadevrt PROPERTIES
    IMPORTED_LOCATION "${CUDA_cudadevrt_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  )
endif()

find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
mark_as_advanced(CUDA_cufft_LIBRARY)
if(CUDA_cufft_LIBRARY AND NOT TARGET CUDA::cufft)
  add_library(CUDA::cufft UNKNOWN IMPORTED)
  set_target_properties(CUDA::cufft PROPERTIES
    IMPORTED_LOCATION "${CUDA_cufft_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  )
endif()

find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
mark_as_advanced(CUDA_nvToolsExt_LIBRARY)
if(CUDA_nvToolsExt_LIBRARY AND NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
  set_target_properties(CUDA::nvToolsExt PROPERTIES
    IMPORTED_LOCATION "${CUDA_nvToolsExt_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  )
endif()

find_library(CUDA_cusolver_LIBRARY cusolver HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
mark_as_advanced(CUDA_cusolver_LIBRARY)
if(CUDA_cusolver_LIBRARY AND NOT TARGET CUDA::cusolver)
  add_library(CUDA::cusolver UNKNOWN IMPORTED)
  set_target_properties(CUDA::cusolver PROPERTIES
    IMPORTED_LOCATION "${CUDA_cusolver_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  )
endif()

find_library(CUDA_cusparse_LIBRARY cusparse HINTS ${CUDA_LIB_PATH} NO_DEFAULT_PATH)
mark_as_advanced(CUDA_cusparse_LIBRARY)
if(CUDA_cusparse_LIBRARY AND NOT TARGET CUDA::cusparse)
  add_library(CUDA::cusparse UNKNOWN IMPORTED)
  set_target_properties(CUDA::cusparse PROPERTIES
    IMPORTED_LOCATION "${CUDA_cusparse_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  )
endif()

# find cuda-memcheck
find_program(CUDA_MEMCHECK_EXECUTABLE
  NAMES cuda-memcheck
  HINTS "${CUDA_BIN_PATH}"
  NO_DEFAULT_PATH)
mark_as_advanced(CUDA_MEMCHECK_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDALibs
  REQUIRED_VARS
    CUDA_cudart_LIBRARY
    CUDA_cudadevrt_LIBRARY
    CUDA_cufft_LIBRARY
    CUDA_nvToolsExt_LIBRARY
    CUDA_cusolver_LIBRARY
    CUDA_cusparse_LIBRARY
    CUDA_MEMCHECK_EXECUTABLE
)
