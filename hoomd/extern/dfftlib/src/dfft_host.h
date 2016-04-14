/*
 * Distributed FFT on the host
 */

#ifndef __DFFT_HOST_H__
#define __DFFT_HOST_H__

#include <dfft_lib_config.h>
#ifdef ENABLE_HOST

#include "dfft_common.h"

#ifdef __cplusplus
#define EXTERN_DFFT extern "C"
#else
#define EXTERN_DFFT
#endif

/* 
 *
 * Create a plan for distributed FFT on the host
 */
EXTERN_DFFT int dfft_create_plan(dfft_plan *p,
    int ndim, int *gdim, int *inembed, int *oembed,
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map);

/*
 * Destroy a plan
 */
EXTERN_DFFT void dfft_destroy_plan(dfft_plan plan);

/*
 * Execute the parallel FFT
 */
EXTERN_DFFT int dfft_execute(cpx_t *h_in, cpx_t *h_out, int dir, dfft_plan p);

#undef EXTERN_DFFT

#endif // ENABLE_HOST
#endif
