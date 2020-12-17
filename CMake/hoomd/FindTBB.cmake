find_path(TBB_INCLUDE_DIR tbb/tbb.h)

find_library(TBB_LIBRARY tbb
             HINTS ${TBB_INCLUDE_DIR}/../lib )

if(TBB_INCLUDE_DIR AND EXISTS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h")
    file(STRINGS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h" TBB_H REGEX "^#define TBB_VERSION_.*$")

    string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*$" "\\1" TBB_VERSION_MAJOR "${TBB_H}")
    string(REGEX REPLACE "^.*TBB_VERSION_MINOR ([0-9]+).*$" "\\1" TBB_VERSION_MINOR  "${TBB_H}")
    set(TBB_VERSION_STRING "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")
endif()

# handle the QUIETLY and REQUIRED arguments and set TBB_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB
                                  REQUIRED_VARS TBB_LIBRARY TBB_INCLUDE_DIR
                                  VERSION_VAR TBB_VERSION_STRING)

# Detect clang and fix incompatibility with TBB
# https://github.com/wjakob/tbb/blob/master/CMakeLists.txt
if (NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
  endif()
endif()

if(TBB_LIBRARY AND NOT TARGET TBB::tbb)
    add_library(TBB::tbb UNKNOWN IMPORTED)
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION_RELEASE "${TBB_LIBRARY}"
        IMPORTED_LOCATION "${TBB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}")

    if (TBB_USE_GLIBCXX_VERSION)
        set_target_properties(TBB::tbb PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "TBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION}")
    endif()
endif()
