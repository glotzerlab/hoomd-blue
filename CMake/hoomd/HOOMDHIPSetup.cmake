OPTION(ENABLE_HIP "True if we are compiling with HIP as a target" FALSE)

find_package(HIP)
if(HIP_FOUND)
    set(ENABLE_HIP TRUE)
    OPTION(HIP_NVCC_FLAGS "Flags used by HIP for compiling with nvcc")
    MARK_AS_ADVANCED(HIP_NVCC_FLAGS)

    # call hipcc to tell us about the nvcc options
    set(ENV{HIPCC_VERBOSE} 1)
    EXECUTE_PROCESS(COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/CMake/hoomd/test.cc OUTPUT_VARIABLE _hipcc_verbose_out)

    string(REPLACE " " ";" _hipcc_verbose_options ${_hipcc_verbose_out})

    # drop the compiler exeuctable and the "hipcc-cmd"
    LIST(REMOVE_AT _hipcc_verbose_options 0 1)

    # drop the -x cu option to not duplicate it with CMake's options
    LIST(FIND _hipcc_verbose_options "-x" _idx)
    if (NOT ${_idx} EQUAL "-1")
    math(EXPR _idx_plus_one "${_idx} + 1")
    LIST(REMOVE_AT _hipcc_verbose_options ${_idx} ${_idx_plus_one})
    endif()

    # finally drop the test file
    LIST(FILTER _hipcc_verbose_options EXCLUDE REGEX test.cc)
    SET(HIP_NVCC_FLAGS ${_hipcc_options_str})

    #search for HIP include directory
    find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
            PATHS
           "${HIP_ROOT_DIR}"
            ENV ROCM_PATH
            ENV HIP_PATH
            PATH_SUFFIXES include)

    if(NOT TARGET HIP::hip)
        add_library(HIP::hip INTERFACE IMPORTED)
        set_target_properties(HIP::hip PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}")
        target_compile_options(HIP::hip INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${HIP_NVCC_FLAGS}>)
    endif()

endif()


