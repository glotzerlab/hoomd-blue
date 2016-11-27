/* MKL (single precision) backend for distributed FFT, implementation
 */

#include "mkl_single_interface.h"

/* Initialize the library
 */
int dfft_init_local_fft()
    {
    return 0;
    }

/* De-initialize the library
 */
void dfft_teardown_local_fft()
    {
    }

/* Create a FFTW plan
 *
 * sign = 0 (forward) or 1 (inverse)
 */
int dfft_create_1d_plan(
    plan_t *plan,
    int dim,
    int howmany,
    int istride,
    int idist,
    int ostride,
    int odist,
    int dir)
    {
    MKL_LONG istrides[2];
    istrides[0] = (MKL_LONG) 0;
    istrides[1] = (MKL_LONG) istride;
    MKL_LONG ostrides[2];
    ostrides[0] = (MKL_LONG) 0;
    ostrides[1] = (MKL_LONG) ostride;

    DftiCreateDescriptor(plan, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)dim );
    DftiSetValue(*plan, DFTI_INPUT_STRIDES, istrides);
    DftiSetValue(*plan, DFTI_OUTPUT_STRIDES, ostrides);
    DftiSetValue(*plan, DFTI_INPUT_DISTANCE, (MKL_LONG) idist);
    DftiSetValue(*plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) odist);
    DftiSetValue(*plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(*plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) howmany);
    DftiCommitDescriptor(*plan);
    return 0;
    }

int dfft_allocate_aligned_memory(cpx_t **ptr, size_t size)
    {
    *ptr = (cpx_t *) malloc(size);
    return 0;
    }

void dfft_free_aligned_memory(cpx_t *ptr)
    {
    free(ptr);
    }

/* Destroy a 1d plan */
void dfft_destroy_1d_plan(plan_t *p)
    {
    DftiFreeDescriptor(p);
    }

/* Excecute a local 1D FFT
 */
void dfft_local_1dfft(
    cpx_t *in,
    cpx_t *out,
    plan_t p,
    int dir)
    {
    if (!dir)
        DftiComputeForward(p, (MKL_Complex8 *) in, (MKL_Complex8 *) out);
    else
        DftiComputeBackward(p, (MKL_Complex8 *) in, (MKL_Complex8 *) out);
    }
