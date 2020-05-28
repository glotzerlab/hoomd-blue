/* Bare radix-2 backend for distributed FFT, implementation
 */

#include "bare_fft_interface.h"

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
    plan->n = dim;
    plan->istride = istride;
    plan->ostride = ostride;
    plan->idist = idist;
    plan->odist = odist;
    plan->howmany = howmany;
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
    }

/*
 * Excecute a local 1D FFT
 */
void dfft_local_1dfft(
    cpx_t *in,
    cpx_t *out,
    plan_t p,
    int dir)
    {
    if (!dir)
        radix2_fft(in, out, p. n, -1, p);
    else
        radix2_fft(in, out, p. n, 1, p);
    }
