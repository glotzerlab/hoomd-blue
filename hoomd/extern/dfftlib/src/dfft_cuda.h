/*
 * Distributed FFT on the host
 */

#ifndef __DFFT_CUDA_H__
#define __DFFT_CUDA_H__

#include <dfft_lib_config.h>

#ifdef ENABLE_CUDA
#include "dfft_common.h"

#ifndef NVCC

#ifdef __cplusplus
#define EXTERN_DFFT extern "C"
#else
#define EXTERN_DFFT
#endif

/*
 * Create a device plan for distributed FFT
 */
EXTERN_DFFT int dfft_cuda_create_plan(dfft_plan *p,
    int ndim, int *gdim, int *inembed, int *oembed,
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map);

/*
 * Destroy a device plan
 */
EXTERN_DFFT void dfft_cuda_destroy_plan(dfft_plan plan);

/*
 * Set error checking on a plan
 */
EXTERN_DFFT void dfft_cuda_check_errors(dfft_plan *plan, int check_err);

/*
 * Execute the parallel FFT on the device
 */
EXTERN_DFFT int dfft_cuda_execute(cuda_cpx_t *id_in, cuda_cpx_t *d_out, int dir, dfft_plan *p);

#undef EXTERN_DFFT
#endif
#endif // ENABLE_CUDA
#endif
