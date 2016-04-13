/* Bare radix-2 FFT backend for distributed FFT
 */

#ifndef __DFFT_BARE_FFT_INTERFACE_H__
#define __DFFT_BARE_FFT_INTERFACE_H__

#include "bare_fft.h"
#include <stdlib.h>

typedef cpxfloat cpx_t;
typedef bare_fft_plan plan_t;

#define RE(X) X.x
#define IM(X) X.y

/* Initialize the library
 */
int dfft_init_local_fft();

/* De-initialize the library
 */
void dfft_teardown_local_fft();

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
    int dir);

int dfft_allocate_aligned_memory(cpx_t **ptr, size_t size);

void dfft_free_aligned_memory(cpx_t *ptr);

/* Destroy a 1d plan */
void dfft_destroy_1d_plan(plan_t *p);

/* Excecute a local 1D FFT
 */
void dfft_local_1dfft(
    cpx_t *in,
    cpx_t *out,
    plan_t p,
    int dir);
#endif
