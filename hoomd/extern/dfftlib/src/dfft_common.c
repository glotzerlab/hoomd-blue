#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "dfft_common.h"
#include "dfft_host.h"
#include "dfft_cuda.h"

#include <stdio.h>

#define CHECK_PLAN_CREATE(res) \
    {                                                                          \
    if (res != 0)                                                              \
        {                                                                      \
        printf("Plan creation failed, file %s, line %d.\n",__FILE__,__LINE__); \
        assert(!res);                                                          \
        exit(1);                                                               \
        }                                                                      \
    }

/*****************************************************************************
 * Implementation of common functions for device and host distributed FFT
 *****************************************************************************/
void bitrev_init(int n, int *rho)
    {
    int j;
    int n1, rem, val, k, lastbit, one=1;

    if (n==1)
        {
        rho[0]= 0;
        return;
        }
    n1= n;
    for(j=0; j<n; j++)
        {
        rem= j;
        val= 0;
        for (k=1; k<n1; k <<= 1)
            {
            lastbit= rem & one;
            rem >>= 1;
            val <<= 1;
            val |= lastbit;
            }
        rho[j]= (int)val;
        }
   }

/*****************************************************************************
 * Plan management
 *****************************************************************************/

/* create a multidimensional execution plan */
int dfft_create_execution_flow(dfft_plan *plan)
    {
    /* compute depth of parallel fft (number of redistributions/dimension) */
    plan->depth = malloc(sizeof(int)*plan->ndim);
    plan->max_depth = 0;
    int i,d,j;
    for (i=0; i< plan->ndim; ++i)
        {
        d = 0;
        int p = plan->pdim[i];
        int l = plan->gdim[i]/p;
        if (l > 1)
            {
            for (j = plan->k0[i]; j <= p; j*=l)
                d++;
            }
        plan->depth[i] = d;
        if (d > plan->max_depth) plan->max_depth = d;
        }

    if (plan->max_depth)
        plan->n_fft = (int *)malloc(sizeof(int)*plan->max_depth);

    /* we always decompose into 1d FFTs when the backend doesn't support
     * multidimensional */
    int decompose_1d = 1 ;
    #ifdef ENABLE_CUDA
    if (plan->device)
        {
        #ifdef CUDA_FFT_SUPPORTS_MULTI
        decompose_1d = ((plan->ndim > CUDA_FFT_MAX_N) ? 1 : 0);
        #else
        decompose_1d = 1;
        #endif

        }
    else
    #endif
        {
        #ifdef HOST_FFT_SUPPORTS_MULTI
        decompose_1d = 0;
        #else
        decompose_1d = 1;
        #endif
        }

    /* allocate plans */
    #ifdef ENABLE_CUDA
    if (plan->device)
        {
        if (plan->max_depth)
            {
            plan->cuda_plans_multi_fw =
                (cuda_plan_t **)malloc(sizeof(cuda_plan_t *)*plan->max_depth);
            plan->cuda_plans_multi_bw =
                (cuda_plan_t **)malloc(sizeof(cuda_plan_t *)*plan->max_depth);
           }
        }
    #endif

    /* now create plans for every level, except the lowest */
    /* this is a general 'vector-radix' implementation */
    for (d = plan->max_depth-1; d >= 0; d--)
        {
        /* count number of ffts at this level in all dimensions */
        int n = 0;
        for (i = 0; i < plan->ndim; ++i)
            if (plan->depth[i] > d) n++;

        if (n < plan->ndim || decompose_1d)
            {
            /* 1D FFT for all 'active' dimensions, transposing as we go */
            plan->n_fft[d] = plan->ndim;

            /* allocate ndim plans */
            #ifdef ENABLE_CUDA
            if (plan->device)
                {
                plan->cuda_plans_multi_fw[d] =
                    (cuda_plan_t *)malloc(sizeof(cuda_plan_t)*plan->ndim);
                plan->cuda_plans_multi_bw[d] =
                    (cuda_plan_t *)malloc(sizeof(cuda_plan_t)*plan->ndim);
                }
            #endif

            for (i = 0; i< plan->ndim; ++i)
                {
                /* if at this level, there is no planned FFT along the current
                 * dimension, just proceed  */
                if (plan->depth[i] > d)
                    {
                    #ifdef ENABLE_CUDA
                    int dim;
                    int istride = plan->size_in/plan->inembed[i];
                    int ostride = 1;
                    int idist = 1;
                    int odist = plan->inembed[i];
                    int howmany = plan->size_in/plan->inembed[i];

                    dim = plan->gdim[i]/plan->pdim[i];

                    if (plan->device)
                        {
                        int res;
                        res = dfft_cuda_create_1d_plan(&plan->cuda_plans_multi_fw[d][i],
                            dim, howmany, istride, idist, ostride, odist, 0);
                        CHECK_PLAN_CREATE(res);
                        res = dfft_cuda_create_1d_plan(&plan->cuda_plans_multi_bw[d][i],
                            dim, howmany, istride, idist, ostride, odist, 1);
                        CHECK_PLAN_CREATE(res);
                        }
                    #endif
                    }
                }
            }
        else
            {
            /* all dimensions have equal length, use multidimensinoal FFT */
            plan->n_fft[d] = 1;

            /* allocate plan */
            #ifdef ENABLE_CUDA
            if (plan->device)
                {
                plan->cuda_plans_multi_fw[d] =
                    (cuda_plan_t *)malloc(sizeof(cuda_plan_t));
                plan->cuda_plans_multi_bw[d] =
                    (cuda_plan_t *)malloc(sizeof(cuda_plan_t));
                }
            #endif

            int *dim = malloc(sizeof(int)*plan->ndim);
            for (i = 0; i < plan->ndim; ++i)
                dim[i] = plan->gdim[i]/plan->pdim[i];

            /* create multidimensional forward and backward plans */
            #ifdef ENABLE_CUDA
            int howmany = 1;
                        
            if (plan->device)
                {
                int res;
                res = dfft_cuda_create_nd_plan(&plan->cuda_plans_multi_fw[d][0],
                    plan->ndim, dim, howmany,
                    plan->inembed, 1, 1, plan->inembed, 1, 1, 0);
                CHECK_PLAN_CREATE(res);
                res = dfft_cuda_create_nd_plan(&plan->cuda_plans_multi_bw[d][0],
                    plan->ndim, dim, howmany,
                    plan->inembed, 1, 1, plan->inembed, 1, 1, 1);
                CHECK_PLAN_CREATE(res);
                }
            #endif
            free(dim);
            }
        } /* end loop over levels */

    /* create plans for final stage */
    plan->final_multi = 1;
    if (decompose_1d)
        {
        plan->final_multi = 0;
        }
    else
        {
        /* find out if we are doing a full FFT in every dimension */
        for (i = 0; i < plan->ndim; ++i)
            if (! (plan->k0[i] == plan->gdim[i]/plan->pdim[i]))
                {
                plan->final_multi = 0;
                break;
                }
        }

    if (!plan->final_multi)
        {
        /* again, we proceed by doing partial 1d FFTs dimension-wise,
         * in between we have to transpose */

        /* allocate plans */
        #ifdef ENABLE_CUDA
        if (plan->device)
            {
            plan->cuda_plans_final_fw =
                (cuda_plan_t *)malloc(sizeof(cuda_plan_t)*plan->ndim);
            plan->cuda_plans_final_bw =
                (cuda_plan_t *)malloc(sizeof(cuda_plan_t)*plan->ndim);
            }
        #endif

        int size = plan->size_in;
        for (i = 0; i < plan->ndim; ++i)
            {
            #ifdef ENABLE_CUDA
            int s = size/plan->inembed[i] *(plan->gdim[i]/plan->pdim[i]);
            int dim = plan->k0[i];
            int howmany = s/(plan->k0[i]);
            int istride = s/(plan->k0[i]);
            int idist = 1;
            int ostride = s/(plan->k0[i]);
            int odist = 1;

            int res;
            res = dfft_cuda_create_1d_plan(&plan->cuda_plans_final_fw[i],
                dim, howmany, istride, idist, ostride, odist, 0);
            CHECK_PLAN_CREATE(res);
            res = dfft_cuda_create_1d_plan(&plan->cuda_plans_final_bw[i],
                dim, howmany, istride, idist, ostride, odist, 1);
            CHECK_PLAN_CREATE(res);
            #endif

            /* this time, we change to the output embedding */
            size /= plan->inembed[i];
            size *= plan->oembed[i];
            }
        }
    else
        {
        /* the final stage is a multidimensional transform */
        /* allocate plan */
        #ifdef ENABLE_CUDA
        if (plan->device)
            {
            plan->cuda_plans_final_fw =
                (cuda_plan_t *)malloc(sizeof(cuda_plan_t));
            plan->cuda_plans_final_bw =
                (cuda_plan_t *)malloc(sizeof(cuda_plan_t));
            }
        #endif

        int *dim = malloc(sizeof(int)*plan->ndim);
        for (i = 0; i < plan->ndim; ++i)
            dim[i] = plan->gdim[i]/plan->pdim[i];

        /* create multidimensional forward and backward plans */
        #ifdef ENABLE_CUDA
        int howmany = 1;
                
        if (plan->device)
            {
            int res;
            res = dfft_cuda_create_nd_plan(&plan->cuda_plans_final_fw[0],
                plan->ndim, dim, howmany,
                plan->inembed, 1, 1, plan->oembed, 1, 1, 0);
            CHECK_PLAN_CREATE(res);
            res = dfft_cuda_create_nd_plan(&plan->cuda_plans_final_bw[0],
                plan->ndim, dim, howmany,
                plan->inembed, 1, 1, plan->oembed, 1, 1, 1);
            CHECK_PLAN_CREATE(res);
            }
        #endif
        free(dim);
        }
    return 0;
    }

int dfft_create_plan_common(dfft_plan *p,
    int ndim, int *gdim,
    int *inembed, int *oembed,
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map,
    int device)
    {
    int nump;

    p->comm = comm;

    MPI_Comm_size(comm,&nump);

    /* number of processor must be power of two */
    if (nump & (nump-1)) return 4;

    /* Allocate memory for processor map and copy over */
    p->proc_map = malloc(sizeof(int)*nump);
    memcpy(p->proc_map, proc_map, sizeof(int)*nump);

    p->pdim = malloc(ndim*sizeof(int));
    p->gdim = malloc(ndim*sizeof(int));
    p->pidx = malloc(ndim*sizeof(int));

    p->inembed = malloc(ndim*sizeof(int));
    p->oembed = malloc(ndim*sizeof(int));

    p->ndim = ndim;

    int i;
    for (i = 0; i < ndim; i++)
        {
        p->gdim[i] = gdim[i];

        /* Every dimension must be a power of two */
        if (gdim[i] & (gdim[i]-1)) return 5;

        p->pdim[i] = pdim[i];
        }

    if (inembed != NULL)
        {
        for (i = 0; i < ndim; i++)
            p->inembed[i] = inembed[i];
        }
    else
        {
        for (i = 0; i < ndim; i++)
            p->inembed[i] = p->gdim[i]/p->pdim[i];
        }

    if (oembed != NULL)
        {
        for (i = 0; i < ndim; i++)
            p->oembed[i] = oembed[i];
        }
    else
        {
        for (i = 0; i < ndim; i++)
            p->oembed[i] = p->gdim[i]/p->pdim[i];
        }

    p->offset_send = (int *)malloc(sizeof(int)*nump);
    p->offset_recv = (int *)malloc(sizeof(int)*nump);
    p->nsend = (int *)malloc(sizeof(int)*nump);
    p->nrecv = (int *)malloc(sizeof(int)*nump);

    if (!device)
        {
        #ifdef ENABLE_HOST
        p->plans_short_forward = malloc(sizeof(plan_t)*ndim);
        p->plans_long_forward = malloc(sizeof(plan_t)*ndim);
        p->plans_short_inverse = malloc(sizeof(plan_t)*ndim);
        p->plans_long_inverse = malloc(sizeof(plan_t)*ndim);
        #else
        return 3;
        #endif
        }

    /* local problem size */
    int size_in = 1;
    int size_out = 1;

    /* since we expect column-major input, the leading dimension
      has no embedding */
    p->inembed[0] = gdim[0]/pdim[0];
    p->oembed[0] = gdim[0]/pdim[0];

    for (i = 0; i < ndim ; ++i)
        {
        size_in *= p->inembed[i];
        size_out *= p->oembed[i];
        }

    p->size_in = size_in;
    p->size_out = size_out;

    /* find length k0 of last stage of butterflies */
    p->k0 = malloc(sizeof(int)*ndim);

    for (i = 0; i< ndim; ++i)
        {
        int length = gdim[i]/pdim[i];
        if (length > 1)
            {
            int c;
            for (c=gdim[i]; c>length; c /= length)
                ;
            p->k0[i] = c;
            }
        else
            {
            p->k0[i] = 1;
            }
        }

    p->rho_L = (int **)malloc(ndim*sizeof(int *));
    p->rho_pk0= (int **)malloc(ndim*sizeof(int *));
    p->rho_Lk0 = (int **)malloc(ndim*sizeof(int *));

    for (i = 0; i < ndim; ++i)
        {
        int length = gdim[i]/pdim[i];
        p->rho_L[i] = (int *) malloc(sizeof(int)*length);
        p->rho_pk0[i] = (int *) malloc(sizeof(int)*pdim[i]/(p->k0[i]));
        p->rho_Lk0[i] = (int *) malloc(sizeof(int)*length/(p->k0[i]));
        bitrev_init(length, p->rho_L[i]);
        bitrev_init(pdim[i]/(p->k0[i]),p->rho_pk0[i]);
        bitrev_init(length/(p->k0[i]),p->rho_Lk0[i]);
        }

    /* processor coordinates */
    for (i = 0; i < ndim; ++i)
        {
        p->pidx[i] = pidx[i];
        }

    /* init local FFT library */
    int res;
    if (!device)
        {
        #ifdef ENABLE_HOST
        res = dfft_init_local_fft();
        #else
        return 3;
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        res = dfft_cuda_init_local_fft();
        #else
        return 2;
        #endif
        }

    if (res) return 1;

    int size = size_in;

    p->dfft_multi = 0;

    p->device = device;

    if (device)
        {
        /* use multidimensional local transforms */
        dfft_create_execution_flow(p);

        /* allocate storage for variables */
        int dmax = p->max_depth + 2;
        p->rev_j1 = (int **) malloc(sizeof(int *)*dmax);
        p->rev_global = (int **) malloc(sizeof(int *)*dmax);
        p->rev_partial = (int **) malloc(sizeof(int *)*dmax);
        p->c0 = (int **) malloc(sizeof(int *)*dmax);
        p->c1 = (int **) malloc(sizeof(int *)*dmax);
        int d;
        for (d = 0; d < dmax; ++d)
            {
            p->rev_j1[d] = (int *) malloc(sizeof(int)*ndim);
            p->rev_global[d] = (int *) malloc(sizeof(int)*ndim);
            p->rev_partial[d] = (int *) malloc(sizeof(int)*ndim);
            p->c0[d] = (int *) malloc(sizeof(int)*ndim);
            p->c1[d] = (int *) malloc(sizeof(int)*ndim);
            }

        p->dfft_multi = 1;
        }
    else
        {
        for (i = 0; i < ndim; ++i)
            {
            /* plan for short-distance butterflies */
            int st = size/p->inembed[i]*(gdim[i]/pdim[i]);

            #ifdef ENABLE_HOST
            int howmany = 1;
            #ifdef FFT1D_SUPPORTS_THREADS
            howmany = st/(p->k0[i]);
            #endif
            dfft_create_1d_plan(&(p->plans_short_forward[i]),p->k0[i],
                howmany, st/(p->k0[i]), 1, st/(p->k0[i]), 1, 0);
            dfft_create_1d_plan(&(p->plans_short_inverse[i]),p->k0[i],
                howmany, st/(p->k0[i]), 1, st/(p->k0[i]), 1, 1);

            /* plan for long-distance butterflies */
            int length = gdim[i]/pdim[i];
            #ifdef FFT1D_SUPPORTS_THREADS
            howmany = st/length;
            #endif
            dfft_create_1d_plan(&(p->plans_long_forward[i]), length,
                howmany, st/length,1, st/length,1, 0);
            dfft_create_1d_plan(&(p->plans_long_inverse[i]), length,
                howmany, st/length,1, st/length,1, 1);
            #else
            return 3;
            #endif

            size /= p->inembed[i];
            size *= p->oembed[i];
            }
        }

    /* Allocate scratch space */
    int scratch_size = 1;
    for (i = 0; i < ndim; ++i)
        scratch_size *= ((p->inembed[i] > p->oembed[i]) ? p->inembed[i]  : p->oembed[i]);
    p->scratch_size = scratch_size;

    if (!device)
        {
        #ifdef ENABLE_HOST
        dfft_allocate_aligned_memory(&(p->scratch),sizeof(cpx_t)*scratch_size);
        dfft_allocate_aligned_memory(&(p->scratch_2),sizeof(cpx_t)*scratch_size);
        dfft_allocate_aligned_memory(&(p->scratch_3),sizeof(cpx_t)*scratch_size);
        #else
        return 3;
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        dfft_cuda_allocate_aligned_memory(&(p->d_scratch),sizeof(cuda_cpx_t)*scratch_size);
        dfft_cuda_allocate_aligned_memory(&(p->d_scratch_2),sizeof(cuda_cpx_t)*scratch_size);
        dfft_cuda_allocate_aligned_memory(&(p->d_scratch_3),sizeof(cuda_cpx_t)*scratch_size);
        #else
        return 2;
        #endif
        }

    p->input_cyclic = input_cyclic;
    p->output_cyclic = output_cyclic;

    #ifdef ENABLE_CUDA
    #ifndef NDEBUG
    p->check_cuda_errors = 1;
    #else
    p->check_cuda_errors = 0;
    #endif
    #endif

    p->row_m = row_m;

    /* before plan creation is complete, an initialization run will
     * be performed */
    p->init = 1;

    return 0;
    }

void dfft_destroy_plan_common(dfft_plan p, int device)
    {
    if (device != p.device) return;

    /* Clean-up */
    free(p.proc_map);

    int i;
    int ndim = p.ndim;
    for (i = 0; i < ndim; ++i)
        {
        if (!device)
            {
            #ifdef ENABLE_HOST
            dfft_destroy_1d_plan(&p.plans_short_forward[i]);
            dfft_destroy_1d_plan(&p.plans_short_inverse[i]);
            dfft_destroy_1d_plan(&p.plans_long_forward[i]);
            dfft_destroy_1d_plan(&p.plans_long_inverse[i]);
            #endif
            }
        }

    for (i = 0; i < ndim; ++i)
        {
        free(p.rho_L[i]);
        free(p.rho_pk0[i]);
        free(p.rho_Lk0[i]);
        }

    free(p.rho_L);
    free(p.rho_pk0);
    free(p.rho_Lk0);

    free(p.offset_recv);
    free(p.offset_send);
    free(p.nrecv);
    free(p.nsend);

    free(p.pidx);
    free(p.pdim);
    free(p.gdim);

    if (!device)
        {
        #ifdef ENABLE_HOST
        dfft_free_aligned_memory(p.scratch);
        dfft_free_aligned_memory(p.scratch_2);
        dfft_free_aligned_memory(p.scratch_3);
        dfft_teardown_local_fft();
        #endif
        }
    else
        {
        #ifdef ENABLE_CUDA
        dfft_cuda_free_aligned_memory(p.d_scratch);
        dfft_cuda_free_aligned_memory(p.d_scratch_2);
        dfft_cuda_free_aligned_memory(p.d_scratch_3);
        dfft_cuda_teardown_local_fft();
        #endif
        }

    if (p.dfft_multi)
        {
        int d;
        for (d = 0; d < p.max_depth; ++d)
            {
            int j;
            for (j = 0; j < p.n_fft[d]; ++j)
                {
                if (p.device && d < p.depth[j])
                    {
                    #ifdef ENABLE_CUDA
                    dfft_cuda_destroy_local_plan(&p.cuda_plans_multi_fw[d][j]);
                    dfft_cuda_destroy_local_plan(&p.cuda_plans_multi_bw[d][j]);
                    #endif
                    }
                }

           }

        for (d = 0; d < p.max_depth+2; ++d)
            {
            free(p.rev_j1[d]);
            free(p.rev_partial[d]);
            free(p.rev_global[d]);
            free(p.c0[d]);
            free(p.c1[d]);
            }
        free(p.rev_j1);
        free(p.rev_partial);
        free(p.rev_global);
        free(p.c0);
        free(p.c1);

        int n = (p.final_multi ? 1 : p.ndim);
        for (i = 0; i < n; ++i)
            {
            if (p.device)
                {
                #ifdef ENABLE_CUDA
                dfft_cuda_destroy_local_plan(&p.cuda_plans_final_fw[i]);
                dfft_cuda_destroy_local_plan(&p.cuda_plans_final_bw[i]);
                #endif
                }
            }

        if (p.max_depth) free(p.n_fft);
        free(p.depth);
        }
    }
