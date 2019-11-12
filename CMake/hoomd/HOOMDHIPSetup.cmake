OPTION(ENABLE_HIP "True if we are compiling with HIP as a target" FALSE)

if(ENABLE_HIP)
    find_package(HIP)

    if (HIP_FOUND)
        OPTION(HIP_NVCC_FLAGS "Flags used by HIP for compiling with nvcc")
        MARK_AS_ADVANCED(HIP_NVCC_FLAGS)

        # call hipcc to tell us about the nvcc options
        set(ENV{HIPCC_VERBOSE} 1)
        EXECUTE_PROCESS(COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/CMake/hoomd/test.cc OUTPUT_VARIABLE _hipcc_verbose_out)

        string(REPLACE " " ";" _hipcc_verbose_options ${_hipcc_verbose_out})

        # get the compiler executable for device code
        LIST(GET _hipcc_verbose_options 1 _hip_compiler)

        # set it as the compiler
        if (${_hip_compiler} MATCHES nvcc)
            set(HIP_PLATFORM nvcc)
        elseif(${_hip_compiler} MATCHES hcc)
            message(ERROR "Deprecaterd hcc backend for HIP is unsupported" ${_hip_compiler})
        elseif(${_hip_compiler} MATCHES clang)
            # fixme
            set(HIP_PLATFORM hip-clang)
        else()
            message(ERROR "Unknown HIP backend " ${_hip_compiler})
        endif()

        # use hipcc as C++ linker for shared libraries
        SET(CMAKE_CUDA_COMPILER ${HIP_HIPCC_EXECUTABLE})
        string(REPLACE "<CMAKE_CXX_COMPILER>" "${HIP_HIPCC_EXECUTABLE}" _link_exec ${CMAKE_CXX_CREATE_SHARED_LIBRARY})
        SET(CMAKE_CXX_CREATE_SHARED_LIBRARY ${_link_exec})

        # use hipcc as C++ linker for executables
        SET(CMAKE_CUDA_COMPILER ${HIP_HIPCC_EXECUTABLE})
        string(REPLACE "<CMAKE_CXX_COMPILER>" "${HIP_HIPCC_EXECUTABLE}" _link_exec ${CMAKE_CXX_LINK_EXECUTABLE})
        SET(CMAKE_CXX_LINK_EXECUTABLE ${_link_exec})


        # this is hack to set the right options on hipcc, may not be portable
        include(hipcc)

	    # don't let CMake examine the compiler, because it will fail
        SET(CMAKE_CUDA_COMPILER_FORCED TRUE)
        ENABLE_LANGUAGE(CUDA)

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

    else()
        # here we go if hipcc is not available, fall back on internal HIP->CUDA headers

        set(HIP_INCLUDE_DIR "")

        #pbly need to extract from header
        set(HIP_VERSION_MAJOR "")
        set(HIP_VERSION_MINOR "")
        set(HIP_VERSION_PATCH "")
        set(HIP_NVCC_FLAGS "")
        message(ERROR "hipcc not found")
    endif()

    if(NOT TARGET HIP::hip)
        add_library(HIP::hip INTERFACE IMPORTED)
        set_target_properties(HIP::hip PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}")
#        target_compile_options(HIP::hip INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${HIP_NVCC_FLAGS}>)
        target_compile_definitions(HIP::hip INTERFACE ENABLE_HIP)

        if(HIP_PLATFORM STREQUAL "hip-clang")
            # needed with hip-clang
            target_compile_definitions(HIP::hip INTERFACE __HIP_PLATFORM_HCC__)
        endif()

        target_compile_definitions(HIP::hip INTERFACE HIP_PLATFORM=${HIP_PLATFORM})

        # set HIP_VERSION_* on non-CUDA targets (the version is already defined on CUDA targets through HIP_NVCC_FLAGS)
        target_compile_definitions(HIP::hip INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_MAJOR=${HIP_VERSION_MAJOR}>)
        target_compile_definitions(HIP::hip INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_MINOR=${HIP_VERSION_MINOR}>)
        target_compile_definitions(HIP::hip INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:HIP_VERSION_PATCH=${HIP_VERSION_PATCH}>)

        # branch upon HCC or NVCC target
        if(${HIP_PLATFORM} STREQUAL "nvcc")
            target_compile_definitions(HIP::hip INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:__HIP_PLATFORM_NVCC__>)
        elseif(${HIP_PLATFORM} STREQUAL "hip-clang")
            target_compile_definitions(HIP::hip INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:__HIP_PLATFORM_HCC__>)
        endif()
    endif()

    # CMake doesn't know HIP as a language, compile through CUDA
	if (NOT ENABLE_HIP OR (ENABLE_HIP AND HIP_PLATFORM STREQUAL "nvcc"))
		if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.0)
			message(SEND_ERROR "HOOMD-blue requires CUDA 9.0 or newer")
		endif()
	endif()
	find_package(CUDALibs REQUIRED)
endif()


