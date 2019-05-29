option(ENABLE_TBB "Enable support for Threading Building Blocks (TBB)" off)

if(ENABLE_TBB)
    find_package(TBB 4.3)
    list(APPEND HOOMD_COMMON_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIR})

    # Detect clang and fix incompatibility with TBB
    # https://github.com/wjakob/tbb/blob/master/CMakeLists.txt
    if (NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # using Clang
        string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
      endif()
    endif()

    list(APPEND HOOMD_COMMON_LIBS ${TBB_LIBRARY})
endif()

if (TBB_USE_GLIBCXX_VERSION)
   add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
endif()
