# setup CUDA compile options
if (ENABLE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # ignore warnings about unused results
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -diag-suppress 2810")

    execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" --version
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
    )

    string(
        FIND
        "${_stdout}"
        "SCALE compiler"
        _scale_compiler_position
    )

    if(_result EQUAL 0 AND NOT _position EQUAL -1)
        set(_HOOMD_FOUND_SCALE TRUE)
        find_package_message(_hoomd_scale "Found SCALE: ${CMAKE_CUDA_COMPILER} (version ${CMAKE_CUDA_COMPILER_VERSION})" "[${CMAKE_CUDA_COMPILER}][${CMAKE_CUDA_COMPILER_VERSION}]")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-cuda-implicit-real-arch")
    else()
        set(_HOOMD_FOUND_SCALE FALSE)
    endif()
endif (ENABLE_GPU)
