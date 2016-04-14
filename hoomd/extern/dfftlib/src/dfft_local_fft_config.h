/*
 * Include interface definitions for local FFT libraries
 */

#include <dfft_lib_config.h>
#ifndef NVCC
#ifdef ENABLE_HOST
/* Local FFT library for host DFFT */

#if (LOCAL_FFT_LIB == LOCAL_LIB_MKL)
/* MKL, single precision is the default library*/
#include "mkl_single_interface.h"

#elif (LOCAL_FFT_LIB == LOCAL_LIB_ACML)
/* ACML, single precision */
#include "acml_single_interface.h"

#elif (LOCAL_FFT_LIB == LOCAL_LIB_BARE)
/* fall back on bare FFT */
#include "bare_fft_interface.h"
#endif
#endif /* ENABLE_HOST */
#endif /* NVCC */

#ifdef ENABLE_CUDA
/* CUFFT is the default library */
#include "cufft_single_interface.h"
#endif /* ENABLE_CUDA */

