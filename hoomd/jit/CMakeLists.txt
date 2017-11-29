set(PACKAGE_NAME jit)

# find and configure LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM Definitions: ${LLVM_DEFINITIONS}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# grab all available libs
set(_llvm_libs ${LLVM_AVAILABLE_LIBS})

# add library directory as necessary
# https://stackoverflow.com/questions/36757842/cmake-use-foreach-and-find-library-to-return-full-path-of-libraries
foreach(LIB ${_llvm_libs})
    find_library(FOUND_LIB_${LIB} ${LIB} HINTS ${LLVM_LIBRARY_DIRS})
    list(APPEND llvm_libs ${FOUND_LIB_${LIB}})
endforeach()

# set files to compile
set(_${PACKAGE_NAME}_sources
     module-${PACKAGE_NAME}.cc
     PatchEnergyJIT.cc
   )

# we compile a separate package just for the LLVM-interfacing part,
# so that can be compiled with and without RTTI
set(_${PACKAGE_NAME}_llvm_sources
    EvalFactory.cc
    OrcLazyJIT.cc)

set(_${PACKAGE_NAME}_headers PatchEnergyJIT.h
                             EvalFactory.h
                             OrcLazyJIT.h
   )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_${PACKAGE_NAME}_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)
set_source_files_properties(${_${PACKAGE_NAME}_llvm_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

add_library (_${PACKAGE_NAME} SHARED ${_${PACKAGE_NAME}_sources})
add_library (_${PACKAGE_NAME}_llvm SHARED ${_${PACKAGE_NAME}_llvm_sources})

# set the appropriate compiler flags on the _llvm target
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
llvm_update_compile_flags(_${PACKAGE_NAME}_llvm)

# work around missing LLVM link information
if(LLVM_ENABLE_TERMINFO)
    find_library(TERMINFO NAMES tinfo ncurses)
    if (${TERMINFO} STREQUAL TERMINFO-NOTFOUND)
        message(FATAL_ERROR "no libtinfo or libncurses is found in system")
    else (${TERMINFO} STREQUAL TERMINFO-NOTFOUND)
        target_link_libraries(_${PACKAGE_NAME}_llvm ${TERMINFO})
    endif (${TERMINFO} STREQUAL TERMINFO-NOTFOUND)
endif()

# link the libraries to their dependencies
target_link_libraries(_${PACKAGE_NAME}_llvm ${llvm_libs})

# need to link llvm_libs here, too, otherwise module import fails
target_link_libraries(_${PACKAGE_NAME} _hoomd _${PACKAGE_NAME}_llvm ${HOOMD_COMMON_LIBS} ${llvm_libs})

# set installation RPATH
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}/)
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_${PACKAGE_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_${PACKAGE_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_${PACKAGE_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_${PACKAGE_NAME}")
set_target_properties(_${PACKAGE_NAME}_llvm PROPERTIES PREFIX "" OUTPUT_NAME "_${PACKAGE_NAME}_llvm")


# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_${PACKAGE_NAME} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_${PACKAGE_NAME})
fix_cudart_rpath(_${PACKAGE_NAME}_llvm)

# install the library
install(TARGETS _${PACKAGE_NAME} _${PACKAGE_NAME}_llvm
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${PACKAGE_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          patch.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PACKAGE_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME} ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/${PACKAGE_NAME}/${file}
        COMMENT    "Copy hoomd/include/hoomd/${PACKAGE_NAME}/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_${PACKAGE_NAME}_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/${PACKAGE_NAME}
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_${PACKAGE_NAME}_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME}_headers ALL DEPENDS ${_${PACKAGE_NAME}_headers})
endif()

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    # add_subdirectory(test)
endif()
