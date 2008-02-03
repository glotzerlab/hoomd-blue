#### 
# This file has been borrowed from the svn trunk of cmake-modules, since it is much improved over the 
# one in the currently released version in CMake itself
# http://code.google.com/p/cmake-modules/source
# it has been modified slightly

#CMake-Modules Project New BSD License

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.

#* Neither the name of the CMake-Modules Project nor the names of its
#  contributors may be used to endorse or promote products derived from this
#  software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# - Try to find Boost
# Once done this will define
#
#  BOOST_FOUND - System has Boost
#  BOOST_INCLUDE_DIRS - Boost include directory
#  BOOST_LIBRARIES - Link these to use Boost
#  BOOST_LIBRARY_DIRS - The path to where the Boost library files are.
#  BOOST_DEFINITIONS - Compiler switches required for using Boost
#  BOOST_LIBRARIES_SUFFIX - Boost libraries name suffix (e.g -vc71-mt-gd-1_34, -gcc41-mt...)
#
#  BOOST_DATE_TIME_FOUND               True if Boost Date Time was found.
#  BOOST_FILESYSTEM_FOUND              True if Boost Filesystem was found.
#  BOOST_IOSTREAMS_FOUND               True if Boost Iostream was found.
#  BOOST_PRG_EXEC_MONITOR_FOUND        True if Boost Program Exec Monitor was found.
#  BOOST_PROGRAM_OPTIONS_FOUND         True if Boost Program Options was found.
#  BOOST_PYTHON_FOUND                  True if Boost Python was found.
#  BOOST_REGEX_FOUND                   True if Boost Regex was found.
#  BOOST_SERIALIZATION_FOUND           True if Boost Serialization was found.
#  BOOST_SIGNALS_FOUND                 True if Boost Signals was found.
#  BOOST_TEST_EXEC_MONITOR_FOUND       True if Boost Test Exec Monitor was found.
#  BOOST_THREAD-MT_FOUND               True if Boost Thread was found.
#  BOOST_UNIT_TEST_FRAMEWORK_FOUND     True if Boost Unit Test Framework was found.
#  BOOST_WSERIALIZATION_FOUND          True if Boost WSerialization was found.
#
#  BOOST_DATE_TIME_LIBRARY             The Boost Date Time libary.
#  BOOST_FILESYSTEM_LIBRARY            The Boost Filesystem libary.
#  BOOST_IOSTREAMS_LIBRARY             The Boost Iostream libary.
#  BOOST_PRG_EXEC_MONITOR_LIBRARY      The Boost Program libary.
#  BOOST_PROGRAM_OPTIONS_LIBRARY       The Boost Program libary.
#  BOOST_PYTHON_LIBRARY                The Boost Python libary.
#  BOOST_REGEX_LIBRARY                 The Boost Regex libary.
#  BOOST_SERIALIZATION_LIBRARY         The Boost Serialization libary.
#  BOOST_SIGNALS_LIBRARY               The Boost Signals libary.
#  BOOST_TEST_EXEC_MONITOR_LIBRARY     The Boost Test Exec Monitor libary.
#  BOOST_THREAD_LIBRARY                The Boost Thread libary.
#  BOOST_UNIT_TEST_FRAMEWORK_LIBRARY   The Boost Unit Test Framework libary.
#  BOOST_WSERIALIZATION_LIBRARY        The Boost WSerialization libary.
#
#  Copyright (c) 2006 Andreas Schneider <mail@cynapses.org>
#  Copyright (c) 2007 Wengo
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#


if (BOOST_LIBRARIES AND BOOST_INCLUDE_DIRS)
  # in cache already
  set(BOOST_FOUND TRUE)
else (BOOST_LIBRARIES AND BOOST_INCLUDE_DIRS)
  # Add in some path suffixes. These will have to be updated whenever
  # a new Boost version comes out.
  set(BOOST_PATH_SUFFIX
    boost-1_34_1
    boost-1_34_0
    boost-1_34
    boost-1_33_1
    boost-1_33_0
    boost-1_33
  )

  if (WIN32)
    # In windows, automatic linking is performed, so you do not have to specify the libraries.
    # If you are linking to a dynamic runtime, then you can choose to link to either a static or a
    # dynamic Boost library, the default is to do a static link.  You can alter this for a specific
    # library "whatever" by defining BOOST_WHATEVER_DYN_LINK to force Boost library "whatever" to
    # be linked dynamically.  Alternatively you can force all Boost libraries to dynamic link by
    # defining BOOST_ALL_DYN_LINK.

    # This feature can be disabled for Boost library "whatever" by defining BOOST_WHATEVER_NO_LIB,
    # or for all of Boost by defining BOOST_ALL_NO_LIB.

    # If you want to observe which libraries are being linked against then defining
    # BOOST_LIB_DIAGNOSTIC will cause the auto-linking code to emit a #pragma message each time
    # a library is selected for linking.
    set(BOOST_LIB_DIAGNOSTIC_DEFINITIONS "-DBOOST_LIB_DIAGNOSTIC")

    set(BOOST_INCLUDE_SEARCH_DIRS
      $ENV{BOOSTINCLUDEDIR}
      $ENV{BOOST_ROOT}/include
      $ENV{BOOST_ROOT}
      C:/boost/include
      $ENV{PROGRAMFILES}/boost/boost_1_34_1
      $ENV{PROGRAMFILES}/boost/boost_1_34_0
      $ENV{PROGRAMFILES}/boost/boost_1_33_1
      # D: is very often the cdrom drive, if you don't have a
      # cdrom inserted it will popup a very annoying dialog
      #D:/boost/include
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
    )

    set(BOOST_LIBRARIES_SEARCH_DIRS
      $ENV{BOOSTLIBDIR}
      $ENV{BOOST_ROOT}/lib
      C:/boost/lib
      $ENV{PROGRAMFILES}/boost/boost_1_34_1/lib
      $ENV{PROGRAMFILES}/boost/boost_1_34_0/lib
      $ENV{PROGRAMFILES}/boost/boost_1_33_1/lib
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
    )

    if (MSVC71)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -vc71-mt-gd-1_34_1
          -vc71-mt-gd-1_34
          -vc71-mt-gd-1_33_1
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -vc71-mt-1_34_1
          -vc71-mt-1_34
          -vc71-mt-1_33_1
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (MSVC71)

    if (MSVC80)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -vc80-mt-gd-1_34_1
          -vc80-mt-gd-1_34
          -vc80-mt-gd-1_33_1
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -vc80-mt-1_34_1
          -vc80-mt-1_34
          -vc80-mt-1_33_1
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (MSVC80)

    if (MINGW)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -mgw-mt-d
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -mgw-mt
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (MINGW)

    if (CYGWIN)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -gcc-mt-d
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(BOOST_LIBRARIES_SUFFIXES
          -gcc-mt
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (CYGWIN)

  else (WIN32)
    set(BOOST_LIBRARIES_SUFFIXES
	-mt
	-mt-s
	-mt-1_34_1
	-mt-1_34_0
	-mt-1_33_1
	-gcc-mt
	-gcc-mt-s
	-gcc41-mt
	-gcc41-mt-s
    )
  endif (WIN32)

  find_path(BOOST_INCLUDE_DIR
    NAMES
      boost/config.hpp
    PATHS
      ${BOOST_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES
      ${BOOST_PATH_SUFFIX}
  )

  foreach (TMP_BOOST_LIBRARIES_SUFFIX ${BOOST_LIBRARIES_SUFFIXES} "")

    if (NOT BOOST_DATE_TIME_LIBRARY)
      find_library(BOOST_DATE_TIME_LIBRARY
        NAMES
          boost_date_time${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )

      if (BOOST_DATE_TIME_LIBRARY)
        # BOOST_DATE_TIME_LIBRARY was found
        # sets the libraries suffix, this code is ugly
        # but CMake does not allow to break a loop :/
        set(BOOST_LIBRARIES_SUFFIX
          ${TMP_BOOST_LIBRARIES_SUFFIX}
          CACHE INTERNAL "" FORCE
        )
      endif (BOOST_DATE_TIME_LIBRARY)

    endif (NOT BOOST_DATE_TIME_LIBRARY)

    if (NOT BOOST_FILESYSTEM_LIBRARY)
      find_library(BOOST_FILESYSTEM_LIBRARY
        NAMES
          boost_filesystem${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_FILESYSTEM_LIBRARY)

    if (NOT BOOST_IOSTREAMS_LIBRARY)
      find_library(BOOST_IOSTREAMS_LIBRARY
        NAMES
          boost_iostreams${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_IOSTREAMS_LIBRARY)

    if (NOT BOOST_PRG_EXEC_MONITOR_LIBRARY)
      find_library(BOOST_PRG_EXEC_MONITOR_LIBRARY
        NAMES
          boost_prg_exec_monitor${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_PRG_EXEC_MONITOR_LIBRARY)

    if (NOT BOOST_PROGRAM_OPTIONS_LIBRARY)
      find_library(BOOST_PROGRAM_OPTIONS_LIBRARY
        NAMES
          boost_program_options${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_PROGRAM_OPTIONS_LIBRARY)

    if (NOT BOOST_PYTHON_LIBRARY)
      find_library(BOOST_PYTHON_LIBRARY
        NAMES
          boost_python${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_PYTHON_LIBRARY)

    if (NOT BOOST_REGEX_LIBRARY)
      find_library(BOOST_REGEX_LIBRARY
        NAMES
          boost_regex${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_REGEX_LIBRARY)

    if (NOT BOOST_SERIALIZATION_LIBRARY)
      find_library(BOOST_SERIALIZATION_LIBRARY
        NAMES
          boost_serialization${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_SERIALIZATION_LIBRARY)

    if (NOT BOOST_SIGNALS_LIBRARY)
      find_library(BOOST_SIGNALS_LIBRARY
        NAMES
          boost_signals${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_SIGNALS_LIBRARY)

    if (NOT BOOST_TEST_EXEC_MONITOR_LIBRARY)
      if (WIN32)
        set (_name libboost_test_exec_monitor${TMP_BOOST_LIBRARIES_SUFFIX})
      else (WIN32)
        set (_name boost_test_exec_monitor${TMP_BOOST_LIBRARIES_SUFFIX})
      endif (WIN32)
      find_library(BOOST_TEST_EXEC_MONITOR_LIBRARY
        NAMES
          ${_name}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_TEST_EXEC_MONITOR_LIBRARY)

    if (NOT BOOST_THREAD_LIBRARY)
      find_library(BOOST_THREAD_LIBRARY
        NAMES
          boost_thread${TMP_BOOST_LIBRARIES_SUFFIX}
          boost_thread-mt
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_THREAD_LIBRARY)

    if (NOT BOOST_UNIT_TEST_FRAMEWORK_LIBRARY)
      set (_boost_unit_test_lib_name "")
      if (WIN32)
        set (_boost_unit_test_lib_name libboost_unit_test_framework${TMP_BOOST_LIBRARIES_SUFFIX})
      else (WIN32)
        set (_boost_unit_test_lib_name boost_unit_test_framework${TMP_BOOST_LIBRARIES_SUFFIX})
      endif (WIN32)
      find_library(BOOST_UNIT_TEST_FRAMEWORK_LIBRARY
        NAMES
          ${_boost_unit_test_lib_name}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_UNIT_TEST_FRAMEWORK_LIBRARY)

    if (NOT BOOST_WSERIALIZATION_LIBRARY)
      find_library(BOOST_WSERIALIZATION_LIBRARY
        NAMES
          boost_wserialization${TMP_BOOST_LIBRARIES_SUFFIX}
        PATHS
          ${BOOST_LIBRARIES_SEARCH_DIRS}
      )
    endif (NOT BOOST_WSERIALIZATION_LIBRARY)

    if (BOOST_DATE_TIME_LIBRARY)
      set(BOOST_DATE_TIME_FOUND TRUE)
    endif (BOOST_DATE_TIME_LIBRARY)
    if (BOOST_FILESYSTEM_LIBRARY)
      set(BOOST_FILESYSTEM_FOUND TRUE)
    endif (BOOST_FILESYSTEM_LIBRARY)
    if (BOOST_IOSTREAMS_LIBRARY)
      set(BOOST_IOSTREAMS_FOUND TRUE)
    endif (BOOST_IOSTREAMS_LIBRARY)
    if (BOOST_PRG_EXEC_MONITOR_LIBRARY)
      set(BOOST_PRG_EXEC_MONITOR_FOUND TRUE)
    endif (BOOST_PRG_EXEC_MONITOR_LIBRARY)
    if (BOOST_PROGRAM_OPTIONS_LIBRARY)
      set(BOOST_PROGRAM_OPTIONS_FOUND TRUE)
    endif (BOOST_PROGRAM_OPTIONS_LIBRARY)
    if (BOOST_PYTHON_LIBRARY)
      set(BOOST_PYTHON_FOUND TRUE)
    endif (BOOST_PYTHON_LIBRARY)
    if (BOOST_REGEX_LIBRARY)
      set(BOOST_REGEX_FOUND TRUE)
    endif (BOOST_REGEX_LIBRARY)
    if (BOOST_SERIALIZATION_LIBRARY)
      set(BOOST_SERIALIZATION_FOUND TRUE)
    endif (BOOST_SERIALIZATION_LIBRARY)
    if (BOOST_SIGNALS_LIBRARY)
      set(BOOST_SIGNALS_FOUND TRUE)
    endif (BOOST_SIGNALS_LIBRARY)
    if (BOOST_TEST_EXEC_MONITOR_LIBRARY)
      set(BOOST_TEST_EXEC_MONITOR_FOUND TRUE)
    endif (BOOST_TEST_EXEC_MONITOR_LIBRARY)
    if (BOOST_THREAD_LIBRARY)
      set(BOOST_THREAD-MT_FOUND TRUE)
    endif (BOOST_THREAD_LIBRARY)
    if (BOOST_UNIT_TEST_FRAMEWORK_LIBRARY)
      set(BOOST_UNIT_TEST_FRAMEWORK_FOUND TRUE)
    endif (BOOST_UNIT_TEST_FRAMEWORK_LIBRARY)
    if (BOOST_WSERIALIZATION_LIBRARY)
      set(BOOST_WSERIALIZATION_FOUND TRUE)
    endif (BOOST_WSERIALIZATION_LIBRARY)

  endforeach (TMP_BOOST_LIBRARIES_SUFFIX)

  set(BOOST_INCLUDE_DIRS
    ${BOOST_INCLUDE_DIR}
  )

  if (BOOST_DATE_TIME_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_DATE_TIME_LIBRARY}
    )
  endif (BOOST_DATE_TIME_FOUND)
  if (BOOST_FILESYSTEM_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_FILESYSTEM_LIBRARY}
    )
  endif (BOOST_FILESYSTEM_FOUND)
  if (BOOST_IOSTREAMS_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_IOSTREAMS_LIBRARY}
    )
  endif (BOOST_IOSTREAMS_FOUND)
  if (BOOST_PRG_EXEC_MONITOR_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_PRG_EXEC_MONITOR_LIBRARY}
    )
  endif (BOOST_PRG_EXEC_MONITOR_FOUND)
  if (BOOST_PROGRAM_OPTIONS_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_PROGRAM_OPTIONS_LIBRARY}
    )
  endif (BOOST_PROGRAM_OPTIONS_FOUND)
  if (BOOST_PYTHON_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_PYTHON_LIBRARY}
    )
  endif (BOOST_PYTHON_FOUND)
  if (BOOST_REGEX_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_REGEX_LIBRARY}
    )
  endif (BOOST_REGEX_FOUND)
  if (BOOST_SERIALIZATION_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_SERIALIZATION_LIBRARY}
    )
  endif (BOOST_SERIALIZATION_FOUND)
  if (BOOST_SIGNALS_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_SIGNALS_LIBRARY}
    )
  endif (BOOST_SIGNALS_FOUND)
  if (BOOST_TEST_EXEC_MONITOR_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_TEST_EXEC_MONITOR_LIBRARY}
    )
  endif (BOOST_TEST_EXEC_MONITOR_FOUND)
  if (BOOST_THREAD-MT_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_THREAD_LIBRARY}
    )
  endif (BOOST_THREAD-MT_FOUND)
  if (BOOST_UNIT_TEST_FRAMEWORK_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
  endif (BOOST_UNIT_TEST_FRAMEWORK_FOUND)
  if (BOOST_WSERIALIZATION_FOUND)
    set(BOOST_LIBRARIES
      ${BOOST_LIBRARIES}
      ${BOOST_WSERIALIZATION_LIBRARY}
    )
  endif (BOOST_WSERIALIZATION_FOUND)

  if (BOOST_INCLUDE_DIRS AND BOOST_LIBRARIES)
    set(BOOST_FOUND TRUE)
  endif (BOOST_INCLUDE_DIRS AND BOOST_LIBRARIES)

  if (BOOST_FOUND)
    if (NOT Boost_FIND_QUIETLY)
      message(STATUS "Found Boost: ${BOOST_LIBRARIES}")
    endif (NOT Boost_FIND_QUIETLY)
  else (BOOST_FOUND)
    if (Boost_FIND_REQUIRED)
      message(FATAL_ERROR "Please install the Boost libraries and development packages")
    endif (Boost_FIND_REQUIRED)
  endif (BOOST_FOUND)

  foreach (BOOST_LIBDIR ${BOOST_LIBRARIES})
    get_filename_component(BOOST_LIBRARY_DIRS ${BOOST_LIBDIR} PATH)
  endforeach (BOOST_LIBDIR ${BOOST_LIBRARIES})

  # Under Windows, automatic linking is performed, so no need to specify the libraries.
  if (WIN32)
    set(BOOST_LIBRARIES "")
  endif (WIN32)

  # show the BOOST_INCLUDE_DIRS and BOOST_LIBRARIES variables only in the advanced view
  mark_as_advanced(BOOST_LIBRARIES BOOST_DEFINITIONS BOOST_LIBRARIES_SUFFIX)
  # hide all of the boost libraries
  mark_as_advanced(BOOST_DATE_TIME_LIBRARY BOOST_FILESYSTEM_LIBRARY BOOST_IOSTREAMS_LIBRARY BOOST_PRG_EXEC_MONITOR_LIBRARY BOOST_PROGRAM_OPTIONS_LIBRARY BOOST_PYTHON_LIBRARY BOOST_REGEX_LIBRARY BOOST_SERIALIZATION_LIBRARY BOOST_SIGNALS_LIBRARY BOOST_TEST_EXEC_MONITOR_LIBRARY BOOST_THREAD_LIBRARY BOOST_UNIT_TEST_FRAMEWORK_LIBRARY BOOST_WSERIALIZATION_LIBRARY)
endif (BOOST_LIBRARIES AND BOOST_INCLUDE_DIRS)
