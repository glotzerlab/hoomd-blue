include_directories(${MPI_INCLUDE_PATH} ${CUDA_TOOLKIT_INCLUDE})

if(ENABLE_CUDA)
    # CUFFT is default for CUDA
    set(CUDA_SOURCES cufft_single_interface.c)

    set(CUDA_SOURCES dfft_cuda.c ${CUDA_SOURCES})
    cuda_compile(DFFT_CUDA_O dfft_cuda.cu SHARED)
    set(CUDA_OBJECTS ${DFFT_CUDA_O})
    set(CUDA_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
endif()

if(ENABLE_HOST)
    if(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_MKL")
        set(HOST_SOURCES mkl_single_interface.c)
    elseif(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_ACML")
        set(HOST_SOURCES acml_single_interface.c)
    elseif(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_BARE")
        set(HOST_SOURCES bare_fft_interface.c bare_fft.c)
    endif()

    set(HOST_SOURCES dfft_host.c ${HOST_SOURCES})
endif()

add_library (dfft SHARED dfft_common.c ${HOST_SOURCES} ${CUDA_SOURCES} ${CUDA_OBJECTS})
target_link_libraries(dfft ${MPI_C_LIBRARIES} ${LOCAL_FFT_LIBRARIES} ${CUDA_LIBS})

if(MPI_COMPILE_FLAGS)
  set_target_properties(dfft PROPERTIES
    COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dfft PROPERTIES
    LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()


install(TARGETS dfft DESTINATION ${PYTHON_MODULE_BASE_DIR})
