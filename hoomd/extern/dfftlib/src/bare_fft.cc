#include "bare_fft.h"
#include <math.h>

/* adapted from Numerical Recipes C code r3.04 */
/* this is probably the simplest possible implementation
 * of a radix-2 FFT, and it is probably slow
 */

/* out-of-place transform */
void four1(cpxfloat *in, cpxfloat *out, const int n, const int isign, const int istride, const int ostride) {
	int nn,mmax,m,j,istep,i;
	float wtemp,wr,wpr,wpi,wi,theta,tempr,tempi;

    /* memcpy, using an out-of-place bit reversal would be more efficient */
    for (i = 0; i < n; ++i)
        out[i*ostride] = in[i*istride];

    /* bit reversal (in-place) */
	nn = n << 1;
	j = 1;
	for (i=1;i<nn;i+=2)
        {
        if (j > i)
            {
            cpxfloat tmp = out[(i/2)*ostride];
            out[(i/2)*ostride] = out[(j/2)*ostride];
            out[(j/2)*ostride] = tmp;
            }
		m=n;
		while (m >= 2 && j > m)
            {
            j -= m;
            m >>= 1;
            }
        j += m;
    }
    
    /* Radix-2 butterflies */
    mmax=2;

    float *data = (float *)out;
    int stride = 2*ostride;
    while (nn > mmax) {
        istep=mmax << 1;
        theta=isign*(6.28318530717959/mmax);
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0;
        wi=0.0;

        for (m=1;m<mmax;m+=2)
            {
            for (i=m;i<=nn;i+=istep)
                {
                j=i+mmax;
                tempr=wr*data[((j-1)/2)*stride]-wi*data[((j-1)/2)*stride+1];
                tempi=wr*data[((j-1)/2)*stride+1]+wi*data[((j-1)/2)*stride];
                data[((j-1)/2)*stride]=data[((i-1)/2)*stride]-tempr;
                data[((j-1)/2)*stride+1]=data[((i-1)/2)*stride+1]-tempi;
                data[((i-1)/2)*stride] += tempr;
                data[((i-1)/2)*stride+1] += tempi;
                }
            wtemp=wr;
            wr=wr*wpr-wi*wpi+wr;
            wi=wi*wpr+wtemp*wpi+wi;
            }   
        mmax=istep;
        }
    }

void radix2_fft(cpxfloat *in, cpxfloat *out, const int n, const int isign, bare_fft_plan plan)
    {
    int i;
    for (i = 0; i < plan.howmany; ++i)
        four1(in+i*plan.idist,out+i*plan.odist, plan.n, isign, plan.istride, plan.ostride);
    }
