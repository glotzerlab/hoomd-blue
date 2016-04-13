/*
 * Distributed FFT global defines
 */

#ifndef __DFFT_COMMON_H__
#define __DFFT_COMMON_H__

#include <dfft_lib_config.h>
#include "dfft_local_fft_config.h"

#include <mpi.h>

/*
 * Data structure for a distributed FFT
 */
typedef struct
    {
    int ndim;            /* dimensionality */
    int *gdim;           /* global input array dimensions */
    int *inembed;        /* embedding, per dimension, of input array */
    int *oembed;         /* embedding, per dimension, of output array */

    #ifdef ENABLE_HOST
    plan_t *plans_short_forward;/* short distance butterflies, forward dir */
    plan_t *plans_long_forward;  /* long distance butterflies, inverse dir */
    plan_t *plans_short_inverse; /* short distance butterflies, inverse dir */
    plan_t *plans_long_inverse;  /* long distance butterflies, inverse dir */
    #endif

    int **rho_L;        /* bit reversal lookup length L, per dimension */
    int **rho_pk0;      /* bit reversal lookup length p/k0, per dimension */
    int **rho_Lk0;      /* bit reversal lookup length L/k0, per dimension */

    int *pdim;            /* Dimensions of processor grid */
    int *pidx;            /* Processor index, per dimension */
    MPI_Comm comm;        /* MPI communicator */

    int *offset_recv; /* temporary arrays */
    int *offset_send;
    int *nrecv;
    int *nsend;

    #ifdef ENABLE_HOST
    cpx_t *scratch;       /* Scratch array */
    cpx_t *scratch_2;
    cpx_t *scratch_3;
    #endif

    #ifdef ENABLE_CUDA
    cuda_cpx_t *d_scratch;   /* Scratch array */
    cuda_cpx_t *d_scratch_2;
    cuda_cpx_t *d_scratch_3;
    cuda_cpx_t *h_stage_in;  /* Staging array for MPI calls */
    cuda_cpx_t *h_stage_out; /* Staging array for MPI calls */
    #endif

    int scratch_size;     /* Size of scratch array */

    int np;               /* size of problem (number of elements per proc) */
    int size_in;          /* size including embedding */
    int size_out;         /* size including embedding */
    int *k0;              /* Last stage of butterflies (per dimension */

    int input_cyclic;     /* ==1 if input for the forward transform is cyclic */
    int output_cyclic;    /* ==1 if output for the backward transform is cyclic */

    int device;           /* ==1 if this is a device plan */
    #ifdef ENABLE_CUDA
    int check_cuda_errors; /* == 1 if we are checking errors */
    #endif

    int row_m;            /* ==1 If we are using row-major procesor id mapping */

    int **c0;             /* variables for redistribution, per stage and dimension */
    int **c1;

    #ifdef ENABLE_CUDA
    int **d_c0;           /* Device memory for passing variables for redistribution kernels */
    int **d_c1;
    int *d_pidx;
    int *d_pdim;
    int *d_iembed;
    int *d_oembed;
    int *d_length;
    #endif

    /* variables for multidimensional plans */
    int dfft_multi; /* == 1 if multidimensional local FFTs are used */
    int max_depth;  /* maximal number of redistributions per dimension */
    int *depth;     /* depth of multi-dim dFFT per dimension */
    int *n_fft;     /* number of FFTs at every level */
    int final_multi; /* If the final stage is a multidimensional transform */
    int **rev_j1, **rev_partial, **rev_global; /* flags to indicate bit reversal per dimension */
    #ifdef ENABLE_CUDA
    cuda_plan_t **cuda_plans_multi_fw; /* Multidimensional plan configuration (forward)*/
    cuda_plan_t **cuda_plans_multi_bw; /* backward plans */
    cuda_plan_t *cuda_plans_final_fw; /* Level-0 plans, forward */
    cuda_plan_t *cuda_plans_final_bw; /* backward plans */
    cuda_scalar_t **d_alpha;          /* alpha variables for twiddle factors per dim */
    cuda_scalar_t **h_alpha;          /* host variable */
    int **d_rev_j1, **d_rev_partial, **d_rev_global;  /* bit reversal flags */
    #endif

    #ifdef ENABLE_HOST
    plan_t **plans_multi;
    #endif

    int *proc_map;                    /* Map of cartesian index onto processor ranks */

    int init;                         /* ==1 if initialization run is requested */
    } dfft_plan;

/*
 * Create a plan for distributed FFT (internal interface)
 */
int dfft_create_plan_common(dfft_plan *plan,
    int ndim, int *gdim, int *inembed, int *ombed,
    int *pdim, int *pidx, int row_m, int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map,
    int device);

/*
 * Destroy a plan (internal interface)
 */
void dfft_destroy_plan_common(dfft_plan plan, int device);

#endif
