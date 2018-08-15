find_library(TBB_LIBRARY tbb
             HINTS ENV TBB_LINK)

get_filename_component(_tbb_lib_dir ${TBB_LIBRARY} DIRECTORY)

find_path(TBB_INCLUDE_DIR tbb/tbb.h
          HINTS ENV TBB_INC
          HINTS ${_tbb_lib_dir}/../include)

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

if(TBB_FOUND)
  set(TBB_LIBRARIES ${TBB_LIBRARY})
endif()
