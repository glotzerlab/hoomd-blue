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

# copy all given files from the current source directory to the current build directory
# files must be specified by relative path
#
# @param files: list of files to copy
# @param target: name of copy target
# @param validate_pattern: Check ${CMAKE_CURRENT_BINARY_DIR}/${validate_pattern} for files
#                          that are not in ${files} and issue a warning.
macro(copy_files_to_build files target validate_pattern)
    file(RELATIVE_PATH relative_dir ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    foreach(file ${files})
        add_custom_command (
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
            POST_BUILD
            COMMAND    ${CMAKE_COMMAND}
            ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
            COMMENT    "Copy ${relative_dir}/${file}"
        )
    endforeach()

    add_custom_target(copy_${target} ALL DEPENDS ${files})

    file(GLOB _matching_files "${CMAKE_CURRENT_BINARY_DIR}/${validate_pattern}")
    foreach(file ${_matching_files})
        file(RELATIVE_PATH relative_file ${CMAKE_CURRENT_BINARY_DIR} ${file})
        list(FIND files ${relative_file} found)
        if (found EQUAL -1)
            message(WARNING "${file} exists but is not provided by the source. "
                            "Remove this file to prevent unexpected errors.")
        endif()
    endforeach()
endmacro()

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
