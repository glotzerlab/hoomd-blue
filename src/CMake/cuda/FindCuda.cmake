###############################################################################
#  For more information, please see: http://software.sci.utah.edu
#
#  The MIT License
#
#  Copyright (c) 2007-2008
#  Scientific Computing and Imaging Institute, University of Utah
#
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
# This script locates the Nvidia Compute Unified Driver Architecture (CUDA) 
# tools. It should on both linux and windows, and should be reasonably up to 
# date with cuda releases.
#
# The script will prompt the user to specify CUDA_INSTALL_PREFIX if the 
# prefix cannot be determined by the location of nvcc in the system path. To
# use a different installed version of the toolkit set the environment variable
# CUDA_BIN_PATH before running cmake (e.g. CUDA_BIN_PATH=/usr/local/cuda1.0 
# instead of the default /usr/local/cuda).
#
# Set CUDA_BUILD_TYPE to "Device" or "Emulation" mode.
# _DEVICEEMU is defined in "Emulation" mode.
#
# Set CUDA_BUILD_CUBIN to "ON" or "OFF" to enable and extra compilation pass
# with the -cubin option in Device mode. 
#
# The output is parsed and register, shared memory usage is printed during 
# build. Default ON.
# 
# The script creates the following macros:
# CUDA_INCLUDE_DIRECTORIES( path0 path1 ... )
# -- Sets the directories that should be passed to nvcc 
#    (e.g. nvcc -Ipath0 -Ipath1 ... ). These paths usually contain other .cu 
#    files.
# 
# CUDA_ADD_LIBRARY( cuda_target file0 file1 ... )
# -- Creates a shared library "cuda_target" which contains all of the source 
#    (*.c, *.cc, etc.) specified and all of the nvcc'ed .cu files specified.
#    All of the specified source files and generated .c files are compiled 
#    using the standard CMake compiler, so the normal INCLUDE_DIRECTORIES, 
#    LINK_DIRECTORIES, and TARGET_LINK_LIBRARIES can be used to affect their
#    build and link.
#
# CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ... )
# -- Same as CUDA_ADD_LIBRARY except that an exectuable is created.
#
# The script defines the following variables:
#
# ( Note CUDA_ADD_* macros setup cuda/cut library dependencies automatically. 
# These variables are only needed if a cuda API call must be made from code in 
# a outside library or executable. )
#
# CUDA_INCLUDE         -- Include directory for cuda headers.
# CUDA_TARGET_LINK     -- Cuda RT library. 
# CUDA_NVCC_FLAGS      -- Additional NVCC command line arguments. NOTE: 
#                         multiple arguments must be semi-colon delimited 
#                         e.g. --compiler-options;-Wall
#
# It might be necessary to set CUDA_INSTALL_PATH manually on certain platforms,
# or to use a cuda runtime not installed in the default location. In newer 
# versions of the toolkit the cuda library is included with the graphics 
# driver- be sure that the driver version matches what is needed by the cuda 
# runtime version.
# 
# -- Abe Stephens SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
###############################################################################

# FindCuda.cmake

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/CudaDependency.cmake)

###############################################################################
###############################################################################
# Locate CUDA, Set Build Type, etc.
###############################################################################
###############################################################################

# Parse CUDA build type.
IF (NOT CUDA_BUILD_TYPE)
  SET(CUDA_BUILD_TYPE "Device" CACHE STRING "Cuda build type: Emulation or Device")
ENDIF(NOT CUDA_BUILD_TYPE)

# Emulation if the card isn't present.
IF (CUDA_BUILD_TYPE MATCHES "Emulation")
  # Emulation.
  SET(nvcc_flags --device-emulation -D_DEVICEEMU -g --host-compilation C++)
ELSE(CUDA_BUILD_TYPE MATCHES "Emulation")
  # Device present.
  SET(nvcc_flags --host-compilation C++ -O3)
ENDIF(CUDA_BUILD_TYPE MATCHES "Emulation")

# support shared library builds
if (NOT ENABLE_STATIC)
set(nvcc_flags ${nvcc_flags} "-Xcompiler" ${CMAKE_SHARED_LIBRARY_C_FLAGS})
endif (NOT ENABLE_STATIC)

# nvcc 64-bit build workaround
if (CMAKE_CL_64)
  set(nvcc_flags ${nvcc_flags} -ccbin "C:\\Program Files (x86)\\Microsoft Visual Studio 8\\VC\\bin")
endif (CMAKE_CL_64)

###############
## CUDA ARCH settings
set(CUDA_ARCH 10 CACHE STRING "Target architecture to compile CUDA code for. Valid options are 10, 11, 12, or 13 (currently). They correspond to compute 1.0, 1.1, 1.2, and 1.3 GPU hardware")
# the arch is going to be passed on a command line: verify it so the user doesn't make any blunders
set(_cuda_arch_ok FALSE)
foreach(_valid_cuda_arch 10 11 12 13)
	if (CUDA_ARCH EQUAL ${_valid_cuda_arch})
		set(_cuda_arch_ok TRUE)
	endif (CUDA_ARCH EQUAL ${_valid_cuda_arch})
endforeach(_valid_cuda_arch)
if (NOT _cuda_arch_ok)
	message(FATAL_ERROR "Wrong CUDA_ARCH specified. Must be one of 10, 11, 12, or 13")
endif (NOT _cuda_arch_ok)

set(nvcc_flags ${nvcc_flags} -arch sm_${CUDA_ARCH} -DCUDA_ARCH=${CUDA_ARCH})
add_definitions(-DCUDA_ARCH=${CUDA_ARCH})

# user options
SET(CUDA_BUILD_CUBIN FALSE CACHE BOOL "Generate and parse .cubin files in Device mode.")
SET(CUDA_NVCC_FLAGS "" CACHE STRING "Semi-colon delimit multiple arguments.")

# Search for the cuda distribution.
IF(NOT CUDA_INSTALL_PREFIX)
  FIND_PATH(CUDA_INSTALL_PREFIX
    NAMES nvcc nvcc.exe
    PATHS /usr/local/cuda /opt/cuda ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin
    DOC "Toolkit location."
    )
    
  IF (CUDA_INSTALL_PREFIX) 
    STRING(REGEX REPLACE "[/\\\\]?bin[/\\\\]?$" "" CUDA_INSTALL_PREFIX ${CUDA_INSTALL_PREFIX})
  ENDIF(CUDA_INSTALL_PREFIX)
  IF (NOT EXISTS ${CUDA_INSTALL_PREFIX})
    MESSAGE(FATAL_ERROR "Specify CUDA_INSTALL_PREFIX")
  ENDIF (NOT EXISTS ${CUDA_INSTALL_PREFIX})
ENDIF (NOT CUDA_INSTALL_PREFIX)

# CUDA_NVCC
IF (NOT CUDA_NVCC)
  FIND_PROGRAM(CUDA_NVCC 
    nvcc
    PATHS ${CUDA_INSTALL_PREFIX}/bin $ENV{CUDA_BIN_PATH}
    )
  IF(NOT CUDA_NVCC)
    MESSAGE(FATAL_ERROR "Could not find nvcc")
  ELSE(NOT CUDA_NVCC)
    MARK_AS_ADVANCED(CUDA_NVCC)
  ENDIF(NOT CUDA_NVCC)
ENDIF(NOT CUDA_NVCC)

# CUDA_NVCC_INCLUDE_ARGS
# IF (NOT FOUND_CUDA_NVCC_INCLUDE)
  FIND_PATH(FOUND_CUDA_NVCC_INCLUDE
    device_functions.h # Header included in toolkit
    PATHS ${CUDA_INSTALL_PREFIX}/include 
          $ENV{CUDA_INC_PATH}
    )
  
  IF(NOT FOUND_CUDA_NVCC_INCLUDE)
    MESSAGE(FATAL_ERROR "Could not find Cuda headers")
  ELSE(NOT FOUND_CUDA_NVCC_INCLUDE)
    # Set the initial include dir.
    SET (CUDA_NVCC_INCLUDE_ARGS "-I"${FOUND_CUDA_NVCC_INCLUDE})
	SET (CUDA_INCLUDE ${FOUND_CUDA_NVCC_INCLUDE})

    MARK_AS_ADVANCED(
      FOUND_CUDA_NVCC_INCLUDE
      CUDA_NVCC_INCLUDE_ARGS
      )
  ENDIF(NOT FOUND_CUDA_NVCC_INCLUDE)

# ENDIF(NOT FOUND_CUDA_NVCC_INCLUDE)

# CUDA_TARGET_LINK
IF (NOT CUDA_TARGET_LINK)

  FIND_LIBRARY(FOUND_CUDART
    cudart
    PATHS ${CUDA_INSTALL_PREFIX}/lib $ENV{CUDA_LIB_PATH}
    DOC "\"cudart\" library"
    )
  
  # Check to see if cudart library was found.
  IF(NOT FOUND_CUDART)
    MESSAGE(FATAL_ERROR "Could not find cudart library (cudart)")
  ENDIF(NOT FOUND_CUDART)  

  # 1.1 toolkit on linux doesn't appear to have a separate library on 
  # some platforms.
  FIND_LIBRARY(FOUND_CUDA
    cuda
    PATHS ${CUDA_INSTALL_PREFIX}/lib
    DOC "\"cuda\" library (older versions only)."
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )

  # Add cuda library to the link line only if it is found.
  IF (FOUND_CUDA)
    SET(CUDA_TARGET_LINK ${FOUND_CUDA})
  ENDIF(FOUND_CUDA)

  # Always add cudart to the link line.
  IF(FOUND_CUDART)
    SET(CUDA_TARGET_LINK
      ${CUDA_TARGET_LINK} ${FOUND_CUDART}
      )
    MARK_AS_ADVANCED(
      CUDA_TARGET_LINK 
      CUDA_LIB
      FOUND_CUDA
      FOUND_CUDART
      CUDA_BUILD_CUBIN
      CUDA_NVCC_FLAGS
      CUDA_INSTALL_PREFIX
      )
  ELSE(FOUND_CUDART)
    MESSAGE(FATAL_ERROR "Could not find cuda libraries.")
  ENDIF(FOUND_CUDART)
  
ENDIF(NOT CUDA_TARGET_LINK)

###############################################################################
# Add include directories to pass to the nvcc command.
MACRO(CUDA_INCLUDE_DIRECTORIES)
  FOREACH(dir ${ARGN})
    SET(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS} -I${dir})
  ENDFOREACH(dir ${ARGN})
ENDMACRO(CUDA_INCLUDE_DIRECTORIES)


##############################################################################
##############################################################################
# This helper macro populates the following variables and setups up custom commands and targets to
# invoke the nvcc compiler. The compiler is invoked once with -M to generate a dependency file and
# a second time with -cuda to generate a .c file
# ${target_srcs}
# ${cuda_cu_sources}
##############################################################################
##############################################################################

MACRO(CUDA_add_custom_commands cuda_target)

  SET(target_srcs "")
  SET(cuda_cu_sources "")

  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  FOREACH(file ${ARGN})
    IF(${file} MATCHES ".*\\.cu$")
   
    # strip CMAKE_SOURCE_DIR from the head of ${file}
    string(LENGTH ${CMAKE_SOURCE_DIR} _cuda_source_dir_length)
    string(LENGTH ${file} _cuda_file_length)
    math(EXPR _begin "${_cuda_source_dir_length} + 1")
    math(EXPR _stripped_file_length "${_cuda_file_length} - ${_cuda_source_dir_length} - 1")
	STRING(SUBSTRING ${file} ${_begin} ${_stripped_file_length} stripped_file )

    # Add a custom target to generate a cpp file
	SET(generated_file  "${CMAKE_BINARY_DIR}/src/cuda/${stripped_file}_${cuda_target}_generated${CMAKE_CXX_OUTPUT_EXTENSION}")

    SET(generated_target "${stripped_file}_target")
    
    FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/cuda)

    SET(source_file ${file})

    # MESSAGE("${CUDA_NVCC} ${source_file} ${CUDA_NVCC_FLAGS} ${nvcc_flags} -cuda -o ${generated_file} ${CUDA_NVCC_INCLUDE_ARGS}")
    
    # Bring in the dependencies.  Creates a variable CUDA_NVCC_DEPEND
	SET(cmake_dependency_file "${generated_file}.depend")
	CUDA_INCLUDE_NVCC_DEPENDENCIES(${cmake_dependency_file})
	SET(NVCC_generated_dependency_file "${generated_file}.NVCC-depend")


	# Build the NVCC made dependency file
  IF (CUDA_BUILD_TYPE MATCHES "Device" AND CUDA_BUILD_CUBIN)
    SET(NVCC_generated_cubin_file "${generated_file}.NVCC-cubin.txt")
	  ADD_CUSTOM_COMMAND(

      # Generate the .cubin output.
      OUTPUT ${NVCC_generated_cubin_file}
      COMMAND ${CUDA_NVCC}
      ARGS ${source_file} 
      ${CUDA_NVCC_FLAGS}
      ${nvcc_flags}
      -DNVCC
      -cubin
      -o ${NVCC_generated_cubin_file} 
      ${CUDA_NVCC_INCLUDE_ARGS}

      # Execute the parser script.
      COMMAND  ${CMAKE_COMMAND}
      ARGS 
      -D input_file="${NVCC_generated_cubin_file}"
      -P "${CMAKE_SOURCE_DIR}/CMake/cuda/parse_cubin.cmake"


      # MAIN_DEPENDENCY ${source_file}
      DEPENDS ${source_file}
      DEPENDS ${CUDA_NVCC_DEPEND}



	    COMMENT "Building NVCC -cubin File: ${NVCC_generated_cubin_file}"
      )
  ELSE (CUDA_BUILD_TYPE MATCHES "Device" AND CUDA_BUILD_CUBIN)
    # Depend on something that will exist.
    SET(NVCC_generated_cubin_file "${source_file}")
  ENDIF (CUDA_BUILD_TYPE MATCHES "Device"AND CUDA_BUILD_CUBIN)

	# Build the NVCC made dependency file
	ADD_CUSTOM_COMMAND(
      OUTPUT ${NVCC_generated_dependency_file}
      COMMAND ${CUDA_NVCC}
      ARGS ${source_file} 
           ${CUDA_NVCC_FLAGS}
           ${nvcc_flags}
           -DNVCC
           -M
           -o ${NVCC_generated_dependency_file} 
           ${CUDA_NVCC_INCLUDE_ARGS}
      # MAIN_DEPENDENCY ${source_file}
      DEPENDS ${source_file}
      DEPENDS ${CUDA_NVCC_DEPEND}
	  COMMENT "Building NVCC Dependency File: ${NVCC_generated_dependency_file}"
    )
    
    # Build the CMake readible dependency file
	ADD_CUSTOM_COMMAND(
	  OUTPUT ${cmake_dependency_file}
      COMMAND ${CMAKE_COMMAND}
      ARGS 
      -D input_file="${NVCC_generated_dependency_file}"
      -D output_file="${cmake_dependency_file}"
      -P "${CMAKE_SOURCE_DIR}/CMake/cuda/make2cmake.cmake"
      MAIN_DEPENDENCY ${NVCC_generated_dependency_file}
      COMMENT "Converting NVCC dependency to CMake (${cmake_dependency_file})"
    )

    ADD_CUSTOM_COMMAND(
      OUTPUT ${generated_file}
      MAIN_DEPENDENCY ${source_file} 
      DEPENDS ${CUDA_NVCC_DEPEND}
      DEPENDS ${cmake_dependency_file}
      DEPENDS ${NVCC_generated_cubin_file}
      COMMAND ${CUDA_NVCC} 
      ARGS ${source_file} 
           ${CUDA_NVCC_FLAGS}
           ${nvcc_flags}
           -DNVCC
           --keep
           -c -o ${generated_file} 
           ${CUDA_NVCC_INCLUDE_ARGS}
       COMMENT "Building NVCC ${source_file}: ${generated_file}"
      )
    	
    SET(cuda_cu_sources ${cuda_cu_sources} ${source_file})

    # Add the generated file name to the source list.
    SET(target_srcs ${target_srcs} ${generated_file})
    
	SET_SOURCE_FILES_PROPERTIES(
		${generated_file}
		PROPERTIES
		EXTERNAL_OBJECT true # to say that "this is actually an object file, so it should not be compiled, only linked"
		GENERATED true       # to say that "it is OK that the obj-files do not exist before build time"
  		)

    ELSE(${file} MATCHES ".*\\.cu$")
  
    # Otherwise add the file name to the source list.
    SET(target_srcs ${target_srcs} ${file})
  
    ENDIF(${file} MATCHES ".*\\.cu$")
  ENDFOREACH(file)

ENDMACRO(CUDA_add_custom_commands)

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
MACRO(CUDA_ADD_LIBRARY cuda_target)

  # Create custom commands and targets for each file.
  CUDA_add_custom_commands( ${cuda_target} ${ARGN} )  
  
  # Add the library.
  ADD_LIBRARY(${cuda_target}
    ${target_srcs}
    ${cuda_cu_sources}
    )

  TARGET_LINK_LIBRARIES(${cuda_target}
    ${CUDA_TARGET_LINK}
    )

ENDMACRO(CUDA_ADD_LIBRARY cuda_target)


###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
MACRO(CUDA_ADD_EXECUTABLE cuda_target)
  
  # Create custom commands and targets for each file.
  CUDA_add_custom_commands( ${cuda_target} ${ARGN} )
  
  # Add the library.
  ADD_EXECUTABLE(${cuda_target}
    ${target_srcs}
    ${cuda_cu_sources}
    )

  TARGET_LINK_LIBRARIES(${cuda_target}
    ${CUDA_TARGET_LINK}
    )


ENDMACRO(CUDA_ADD_EXECUTABLE cuda_target)

