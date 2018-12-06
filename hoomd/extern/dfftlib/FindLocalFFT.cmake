find_package(MKL QUIET)
find_package(ACML QUIET)

option(ENABLE_HOST "CPU FFT support" ON)
if (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
    set(LOCAL_FFT_LIB LOCAL_LIB_MKL)
    set(LOCAL_FFT_LIBRARIES "${MKL_LIBRARIES}")
    include_directories(${MKL_INCLUDE_DIR})
    set(ENABLE_OPENMP 1)
elseif(ACML_LIBRARIES)
    set(LOCAL_FFT_LIB LOCAL_LIB_ACML)
    set(LOCAL_FFT_LIBRARIES "${ACML_LIBRARIES}")
    include_directories(${ACML_INCLUDES})
endif()

if (NOT LOCAL_FFT_LIB)
    # fallback on bare FFT
    set(LOCAL_FFT_LIB LOCAL_LIB_BARE)
endif()
