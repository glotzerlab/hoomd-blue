/*
 * hipfft (single precision) backend for distributed FFT, implementation
 */

#include "cufft_single_interface.h"

#include <stdio.h>

/* Initialize the library
 */
int dfft_cuda_init_local_fft()
    {
    return 0;
    }

/* De-initialize the library
 */
void dfft_cuda_teardown_local_fft()
    {
    }

/* Create a 1d CUFFT plan
 *
 * sign = 0 (forward) or 1 (inverse)
 */
int dfft_cuda_create_1d_plan(
    cuda_plan_t *plan,
    int dim,
    int howmany,
    int istride,
    int idist,
    int ostride,
    int odist,
    int dir)
    {
    int dims[1];
    dims[0] = dim;

    #ifdef __HIP_PLATFORM_HCC__
    hipfftResult res;
    res = hipfftPlanMany(plan, 1, dims, dims, istride, idist, dims,
        ostride, odist, HIPFFT_C2C, howmany);
    if (res != HIPFFT_SUCCESS)
    #else
    cufftResult res;
    res = cufftPlanMany(plan, 1, dims, dims, istride, idist, dims,
        ostride, odist, CUFFT_C2C, howmany);
    if (res != CUFFT_SUCCESS)
    #endif
        {
        printf("CUFFT Error: %d\n", res);
        return 1;
        }
    return 0;
    }

/* Create a n-d CUFFT plan
 *
 * sign = 0 (forward) or 1 (inverse)
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
    int dir)
    {
    #ifdef __HIP_PLATFORM_HCC__
    hipfftResult res;
    res = hipfftPlanMany(plan, ndim, dim, iembed, istride, idist, oembed,
        ostride, odist, HIPFFT_C2C, howmany);
    if (res != HIPFFT_SUCCESS)
    #else
    cufftResult res;
    res = cufftPlanMany(plan, ndim, dim, iembed, istride, idist, oembed,
        ostride, odist, CUFFT_C2C, howmany);
    if (res != CUFFT_SUCCESS)
    #endif
        {
        printf("CUFFT Error: %d\n", res);
        return 1;
        }
    return 0;
    }


int dfft_cuda_allocate_aligned_memory(cuda_cpx_t **ptr, size_t size)
    {
    hipMalloc((void **) ptr,size);
    return 0;
    }

void dfft_cuda_free_aligned_memory(cuda_cpx_t *ptr)
    {
    hipFree(ptr);
    }

/* Destroy a 1d plan */
int dfft_cuda_destroy_local_plan(cuda_plan_t *p)
    {
    #ifdef __HIP_PLATFORM_HCC__
    hipfftResult res = hipfftDestroy(*p);
    if (res != HIPFFT_SUCCESS)
    #else
    cufftResult res = cufftDestroy(*p);
    if (res != CUFFT_SUCCESS)
    #endif
        {
        printf("hipfftDestroy error: %d\n", res);
        return res;
        }
    return 0;
    }

/*
 * Excecute a local 1D FFT
 */
int dfft_cuda_local_fft(
    cuda_cpx_t *in,
    cuda_cpx_t *out,
    cuda_plan_t p,
    int dir)
    {
    #ifdef __HIP_PLATFORM_HCC__
    hipfftResult res;
    res = hipfftExecC2C(p, in, out, dir ? HIPFFT_BACKWARD : HIPFFT_FORWARD);
    #else
    cufftResult res;
    res = cufftExecC2C(p, in, out, dir ? CUFFT_INVERSE : CUFFT_FORWARD);
    #endif
    return res;
    }

