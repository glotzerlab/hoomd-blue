/*
 * CUFFT (single precision) backend for distributed FFT
 */

#ifndef __DFFT_CUFFT_SINGLE_INTERFACE_H__
#define __DFFT_CUFFT_SINGLE_INTERFACE_H__

#include <cufft.h>
#include <cuda_runtime.h>

typedef float cuda_scalar_t;
typedef cufftComplex cuda_cpx_t;
typedef cufftHandle cuda_plan_t;

#define CUDA_RE(X) X.x
#define CUDA_IM(X) X.y

/* this library supports a multidimensional transform */
#define CUDA_FFT_SUPPORTS_MULTI
/* maximum dimensionality the library can transform locally */
#define CUDA_FFT_MAX_N 3

#ifndef NVCC

/* Initialize the library
 */
int dfft_cuda_init_local_fft();

/* De-initialize the library
 */
void dfft_cuda_teardown_local_fft();

/* Create a one-dimensional CUFFT plan
 */
int dfft_cuda_create_1d_plan(
    cuda_plan_t *plan,
    int dim,
    int howmany,
    int istride,
    int idist,
    int ostride,
    int odist,
    int dir);

/* Create a n-dimensional CUFFT plan
 * Input is in column-major
 */
int dfft_cuda_create_nd_plan(
    cuda_plan_t *plan,
    int ndim,
    int *dim,
    int howmany,
    int *iembed,
    int istride,
    int idist,
    int *oembed,
    int ostride,
    int odist,
    int dir);

int dfft_cuda_allocate_aligned_memory(cuda_cpx_t **ptr, size_t size);

void dfft_cuda_free_aligned_memory(cuda_cpx_t *ptr);

/* Destroy a plan */
int dfft_cuda_destroy_local_plan(cuda_plan_t *p);

/*
 * Excecute a local 1D FFT
 */
int dfft_cuda_local_fft( cuda_cpx_t *in, cuda_cpx_t *out, cuda_plan_t p, int dir);

#endif /* NVCC */
#endif
