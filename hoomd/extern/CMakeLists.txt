file(GLOB_RECURSE _extern_headers
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     *.h *.cuh *.hpp *.cmake)

foreach(file ${_extern_headers})
    # install into installation target
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/${dir})
endforeach()

# setup hipper target, as it will be installed by HOOMD
add_library(hipper INTERFACE)
add_library(HOOMD::hipper ALIAS hipper)
target_include_directories(hipper INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hipper/include>
                                            $<INSTALL_INTERFACE:${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/hipper/include>)
if (HIP_PLATFORM STREQUAL "hcc" OR HIP_PLATFORM STREQUAL "hip-clang")
    target_compile_definitions(hipper INTERFACE HIPPER_HIP)
elseif(HIP_PLATFORM STREQUAL "nvcc")
    target_compile_definitions(hipper INTERFACE HIPPER_CUDA)
endif()
install(TARGETS hipper EXPORT HOOMDTargets)

# setup neighbor target, as it will be installed by HOOMD
add_library(neighbor INTERFACE)
add_library(HOOMD::neighbor ALIAS neighbor)
target_include_directories(neighbor INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/neighbor/include>
                                              $<INSTALL_INTERFACE:${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/neighbor/include>)
target_link_libraries(neighbor INTERFACE HOOMD::hipper)
install(TARGETS neighbor EXPORT HOOMDTargets)
