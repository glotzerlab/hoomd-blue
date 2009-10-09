# $Id$
# $URL$
# Maintainer: joaander

# enable testing on this directory
if (BUILD_TESTING)
    enable_testing()
endif (BUILD_TESTING)

# here is where we actually compile the python module version of the library
if (ENABLE_CUDA)
cuda_add_library (hoomd_python_module ${HOOMD_CPU_SRCS}
${HOOMD_GPU_SRCS} ${PYTHON_SRCS} OPTIONS ${CUDA_ADDITIONAL_OPTIONS})
else (ENABLE_CUDA)
add_library (hoomd_python_module ${HOOMD_CPU_SRCS}
${PYTHON_SRCS})
endif (ENABLE_CUDA)

##################################
# change the name of the library if it is dynamic
if (NOT ENABLE_STATIC)
    # link the library to its dependancies
    target_link_libraries(hoomd_python_module ${HOOMD_COMMON_LIBS})

    # tweak the properties of the output to make a more useful python module
    set_target_properties(hoomd_python_module PROPERTIES PREFIX "" OUTPUT_NAME "hoomd")

    # .dylib is not regognized as a python module by python on Mac OS X
    if(APPLE)
        set_target_properties(hoomd_python_module PROPERTIES SUFFIX ".so")
    endif(APPLE)
    
    # also: only install the library if it is dynamic  and not in windows
    if (NOT WIN32)
        install(TARGETS hoomd_python_module
            LIBRARY DESTINATION ${LIB_INSTALL_DIR}/python-module
            CONFIGURATIONS Release
            )
            
        # installs the hoomd runner script
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_dynamic.in 
                        ${CMAKE_CURRENT_BINARY_DIR}/hoomd_dynamic 
                        @ONLY
                        )
        install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_dynamic
            DESTINATION ${BIN_INSTALL_DIR}
            CONFIGURATIONS Release
            RENAME hoomd
            )
    endif (NOT WIN32)
endif (NOT ENABLE_STATIC)

########################
## now setup the hoomd executable
    
# setup the python runner executable
# it won't link dynamically with the changed name above, but there really isn't a need to create a dynamically
# linked version as it can be loaded into python directly
if (ENABLE_STATIC)
    add_executable(hoomd hoomd.cc)
    target_link_libraries(hoomd hoomd_python_module ${HOOMD_COMMON_LIBS})
    set_target_properties(hoomd
        PROPERTIES PREFIX ""
        OUTPUT_NAME "hoomd"
        )
    
    # install the executable
    install(TARGETS hoomd
        RUNTIME DESTINATION ${LIB_INSTALL_DIR}
        CONFIGURATIONS Release
        )
        
    # installs the hoomd runner script
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_static.in 
                    ${CMAKE_CURRENT_BINARY_DIR}/hoomd_static
                    @ONLY
                    )	
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_static
            DESTINATION ${BIN_INSTALL_DIR}
            CONFIGURATIONS Release
            RENAME hoomd
            )
            
    # add the hoomd_script tests to the unit test list
    get_target_property(CUR_TEST_EXE hoomd LOCATION)

    # unfortunately, this needs to be handled differently in visual studio
    if (MSVC)
        add_test(hoomd_script_tests_cpu ${EXECUTABLE_OUTPUT_PATH}/hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=cpu)
        if (ENABLE_CUDA)
            add_test(hoomd_script_tests_gpu ${EXECUTABLE_OUTPUT_PATH}/hoomd ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=gpu)
        endif (ENABLE_CUDA)
    else (MSVC)
        add_test(hoomd_script_tests_cpu ${CUR_TEST_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=cpu)
        
        if (ENABLE_CUDA)
            add_test(hoomd_script_tests_gpu ${CUR_TEST_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_script_tests.py --mode=gpu)
        endif (ENABLE_CUDA)
    endif (MSVC)
    
endif (ENABLE_STATIC)

if (ENABLE_PLUGINAPI)
install(TARGETS hoomd_python_module         
        RUNTIME DESTINATION ${BIN_INSTALL_DIR}
        LIBRARY DESTINATION ${LIB_BASE_INSTALL_DIR}
        ARCHIVE DESTINATION ${LIB_BASE_INSTALL_DIR})    
endif (ENABLE_PLUGINAPI)


#################################
## Create source file groups
source_group(utils FILES ${UTILS_SRCS})
source_group(data_structures FILES ${DATA_STRUCT_SRCS})
source_group(computes FILES ${COMPUTES_SRCS})
source_group(computes_gpu FILES ${COMPUTES_GPU_SRCS})
source_group(analyzers FILES ${ANALYZER_SRCS})
source_group(extern FILES ${EXTERN_SRCS})
source_group(updaters FILES ${UPDATER_SRCS})
source_group(updaters_gpu FILES ${UPDATER_GPU_SRCS})
source_group(cuda FILES ${CUDA_SRCS})
source_group(system FILES ${SYSTEM_SRCS})
source_group(python FILES ${PYTHON_SRCS})
