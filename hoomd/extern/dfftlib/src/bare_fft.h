/* A simple radix-2 FFT */
typedef struct { float x,y;} cpxfloat;

typedef struct
    {
    int n;
    int istride;
    int ostride;
    int idist;
    int odist;
    int howmany;
    } bare_fft_plan;

void radix2_fft(cpxfloat *in, cpxfloat *out, const int n, const int isign, bare_fft_plan plan);
