if (MKL_LIBRARIES)
  set(MKL_FIND_QUIETLY TRUE)
endif (MKL_LIBRARIES)

if(CMAKE_MINOR_VERSION GREATER 4)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")

find_library(MKL_LIBRARIES
  mkl_rt
  HINTS
  $ENV{MKLLIB}
  $ENV{MKLROOT}/lib/intel64
  $ENV{MKL_HOME}/lib/intel64
  /opt/mkl-*/lib/em64t
  /opt/intel/mkl/*/lib/em64t
  /opt/intel/Compiler/*/*/mkl/lib/em64t
  ${LIB_INSTALL_DIR}
)

if(MKL_LIBRARIES)
  set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_ilp64 mkl_intel_thread mkl_core pthread m)
endif()

else(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")

find_library(MKL_LIBRARIES
  mkl_core
  HINTS
  $ENV{MKLLIB}
  /opt/mkl-*/lib/32
  /opt/intel/mkl/*/lib/32
  /opt/intel/Compiler/*/*/mkl/lib/32
  ${LIB_INSTALL_DIR}
)

if(MKL_LIBRARIES)
  set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel mkl_intel_thread mkl_core pthread m)
  add_definitions(-DMKL_ILP64)
endif()

endif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")

find_path(MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS
  $ENV{MKLINCLUDE}
  $ENV{MKLROOT}/include
  $ENV{MKL_HOME}/include
  /opt/mkl-*/
  /opt/intel/mkl/*/
  /opt/intel/Compiler/*/*/mkl/
  ${CMAKE_INSTALL_PREFIX}/include
  PATH_SUFFIXES include
  )

endif(CMAKE_MINOR_VERSION GREATER 4)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIR)

mark_as_advanced(MKL_LIBRARIES MKL_INCLUDE_DIR)
