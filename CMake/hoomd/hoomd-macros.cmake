###############################
# Helper macros
macro(fix_cudart_rpath target)
if (ENABLE_HIP AND APPLE)
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcudart.dylib ${CUDA_CUDART_LIBRARY} ${_target_exe})
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcufft.dylib ${CUDA_cufft_LIBRARY} ${_target_exe})
endif (ENABLE_HIP AND APPLE)
endmacro(fix_cudart_rpath)

# find a package by config first
macro(find_package_config_first package version)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.15)
    set(_old_prefer_config ${CMAKE_FIND_PACKAGE_PREFER_CONFIG})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
    find_package(${package} ${version} ${ARGN})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ${_old_prefer_config})
else()
    find_package(${package} ${version} QUIET CONFIG ${ARGN})
    if (NOT ${${package}_FOUND})
        find_package(${package} ${version} MODULE ${ARGN} REQUIRED)
    endif()
endif()

endmacro()
