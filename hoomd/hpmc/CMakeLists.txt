# Maintainer: joaander

option(ENABLE_HPMC_SPHINX_GPU "Enable sphinx on the GPU" OFF)
if (ENABLE_HPMC_SPHINX_GPU)
    add_definitions(-DENABLE_SPHINX_GPU)
endif()

set(_hpmc_sources   module.cc
                    module_external_field.cc
                    module_free_volume.cc
                    module_hpmc.cc
                    module_hpmc_gpu.cc
                    module_muvt.cc
                    module_sdf.cc
                    module_sphere.cc
                    module_convex_polygon.cc
                    UpdaterBoxMC.cc
                    IntegratorHPMC.cc
                    )

# if (ENABLE_CUDA)
# list(APPEND _hpmc_sources
#                            )
# endif()

set(_hpmc_cu_sources ComputeFreeVolumeGPU.cu
                     IntegratorHPMCMonoGPU.cu
                     IntegratorHPMCMonoImplicitGPU.cu
                     )

# quiet some warnings locally on files we can't modify
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(IntegratorHPMCMonoImplicitGPU.cu PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_hpmc_sources} ${_hpmc_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_hpmc_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_hpmc SHARED ${_hpmc_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_hpmc _hoomd ${HOOMD_COMMON_LIBS})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_hpmc PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_hpmc PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_hpmc PROPERTIES PREFIX "" OUTPUT_NAME "_hpmc")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_hpmc PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_hpmc)

# install the library
install(TARGETS _hpmc
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/hpmc
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
        COMMENT    "Copy hoomd/hpmc/${file}"
    )
ENDMACRO(copy_file)

set(files   analyze.py
            compute.py
            data.py
            __init__.py
            integrate.py
            update.py
            util.py
            field.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/hpmc
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hpmc ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
