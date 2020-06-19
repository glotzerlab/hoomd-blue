set(PACKAGE_NAME jit)

# find and configure LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM Definitions: ${LLVM_DEFINITIONS}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs executionengine support core irreader native runtimedyld orcjit object transformutils lto)

# set files to compile
set(_${PACKAGE_NAME}_sources
     module-${PACKAGE_NAME}.cc
     PatchEnergyJIT.cc
     GPUEvalFactory.cc
     PatchEnergyJITGPU.cc
     PatchEnergyJITUnion.cc
     PatchEnergyJITUnionGPU.cc
   )

# we compile a separate package just for the LLVM-interfacing part,
# so that can be compiled with and without RTTI
set(_${PACKAGE_NAME}_llvm_sources EvalFactory.cc ExternalFieldEvalFactory.cc)

set(_${PACKAGE_NAME}_headers PatchEnergyJIT.h
                             PatchEnergyJITUnion.h
                             PatchEnergyJITGPU.h
                             PatchEnergyJITUnionGPU.h
                             ExternalFieldJIT.h
                             EvalFactory.h
                             Evaluator.cuh
                             EvaluatorUnionGPU.cuh
                             ExternalFieldEvalFactory.h
                             GPUEvalFactory.h
                             KaleidoscopeJIT.h
                             jitify.hpp
   )

pybind11_add_module(_${PACKAGE_NAME} SHARED ${_${PACKAGE_NAME}_sources} ${_${PACKAGE_NAME}_cu_sources} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_${PACKAGE_NAME} ALIAS _${PACKAGE_NAME})

add_library (_${PACKAGE_NAME}_llvm SHARED ${_${PACKAGE_NAME}_llvm_sources})

if (ENABLE_HIP)
    # no device linking
    target_compile_definitions(_${PACKAGE_NAME} PUBLIC CUDA_DEVRT_LIBRARY_PATH=\"${CUDA_cudadevrt_LIBRARY}\")
    target_compile_definitions(_${PACKAGE_NAME} PUBLIC CUDA_INCLUDE_PATH=\"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}\")
    string(REPLACE ";" "_" CUDA_COMPUTE_ARCHS "${CUDA_ARCH_LIST}")
    target_compile_definitions(_${PACKAGE_NAME} PUBLIC CUDA_COMPUTE_ARCHS=\"${CUDA_COMPUTE_ARCHS}\")
    target_link_libraries(_${PACKAGE_NAME} PUBLIC CUDA::cuda CUDA::nvrtc)
endif (ENABLE_HIP)

target_include_directories(_${PACKAGE_NAME}_llvm PUBLIC
                           $<BUILD_INTERFACE:${HOOMD_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${PYTHON_SITE_INSTALL_DIR}/include>)

# set the appropriate compiler flags on the _llvm target
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
llvm_update_compile_flags(_${PACKAGE_NAME}_llvm)

# bug: llvm_update_compile_flags removes the c++11 compile flag that hoomd sets for old cmake versions
# cmake 3.1 and newer are not affected becuase we use CMAKE_CXX_STANDARD
if (CMAKE_VERSION VERSION_LESS 3.1.0)
    message(FATAL_ERROR "CMake 3.1.0 or newer is required to buid the JIT module")
endif()

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
target_link_libraries(_${PACKAGE_NAME} PUBLIC _hoomd PRIVATE _${PACKAGE_NAME}_llvm ${llvm_libs})

# set installation RPATH
if(APPLE)
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${PACKAGE_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN")
endif()

fix_cudart_rpath(_${PACKAGE_NAME})
fix_cudart_rpath(_${PACKAGE_NAME}_llvm)

# install the library
install(TARGETS _${PACKAGE_NAME} _${PACKAGE_NAME}_llvm
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${PACKAGE_NAME}
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
          external.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${PACKAGE_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${PACKAGE_NAME} ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_${PACKAGE_NAME}_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/${PACKAGE_NAME}
       )

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    # add_subdirectory(test)
endif()
