# setup CUDA compile options
if (ENABLE_GPU)
    ENABLE_LANGUAGE(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # ignore warnings about unused results
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -diag-suppress 2810")
endif (ENABLE_GPU)
