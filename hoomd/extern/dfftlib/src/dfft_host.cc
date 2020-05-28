#include <stdlib.h>
#include <string.h>

#include "dfft_host.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <math.h>

/*****************************************************************************
 * Implementation of the distributed FFT
 *****************************************************************************/

/*
 * Redistribute from group-cyclic with cycle c0 to cycle c1>=c0
 */
void dfft_redistribute_block_to_cyclic_1d(
                  int *dim,
                  int *pdim,
                  int ndim,
                  int current_dim,
                  int c0,
                  int c1,
                  int* pidx,
                  int size_in,
                  int *embed,
                  cpx_t *work,
                  cpx_t *scratch,
                  int *dfft_nsend,
                  int *dfft_nrecv,
                  int *dfft_offset_send,
                  int *dfft_offset_recv,
                  MPI_Comm comm,
                  int *proc_map,
                  int row_m)
    {
    /* exit early if nothing needs to be done */
    if (c0 == c1) return;

    int length = dim[current_dim]/pdim[current_dim];

    /* compute stride for column major matrix storage */
    int stride = size_in/embed[current_dim];

    /* processor index along current dimension */
    int s = pidx[current_dim];

    int ratio = c1/c0;
    int size = ((length/ratio > 1) ? (length/ratio) : 1);
    int npackets = length/size;
    size *= stride;

    int pdim_tot=1;
    int k;
    for (k = 0; k < ndim; ++k)
        pdim_tot *= pdim[k];

    int t;
    for (t = 0; t<pdim_tot; ++t)
        {
        dfft_nsend[t] = 0;
        dfft_nrecv[t] = 0;
        dfft_offset_send[t] = 0;
        dfft_offset_recv[t] = 0;
        }

    int j0;
    int j2;

    j0 = s % c0;
    j2 = s / c0;

    /* initialize send offsets and pack data */
    int j;
    #pragma omp parallel for private(j,k)
    for (j = 0; j < npackets; ++j)
        {
        int offset = j*size;
        int jglob = j2*c0*length + j * c0 + j0;
        int desti = (jglob/(c1*length))*c1+ jglob%c1;
        int destproc = 0;
        if (row_m)
            {
            for (k = ndim-1; k >=0 ;--k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? desti : pidx[k]);
                }
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? desti : pidx[k]);
                }
            }

        int rank = proc_map[destproc];
        dfft_nsend[rank] = size*sizeof(cpx_t);
        dfft_offset_send[rank] = offset*sizeof(cpx_t);
        int r;
        for(r=0; r< (size/stride); r++)
            for (k=0; k < stride; k++)
               scratch[offset + r*stride+k]=  work[(j+r*ratio)*stride+k];
        }

    /* initialize recv offsets */
    int offset = 0;
    j0 = s % c1;
    j2 = s/c1;

    int r;
    for (r = 0; r < npackets; ++r)
        {
        offset = r*size;
        j = r*size/stride;
        int jglob = j2*c1*length+ j * c1 + j0;
        int srci = (jglob/(c0*length))*c0+jglob%c0;
        int srcproc = 0;
        int k;
        if (row_m)
            {
            for (k = ndim-1; k >= 0; --k)
                {
                srcproc *= pdim[k];
                srcproc += ((current_dim == k) ? srci : pidx[k]);
                }
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                srcproc *= pdim[k];
                srcproc += ((current_dim == k) ? srci : pidx[k]);
                }
            }

        int rank = proc_map[srcproc];

        dfft_nrecv[rank] = size*sizeof(cpx_t);
        dfft_offset_recv[rank] = offset*sizeof(cpx_t);
        }

    /* synchronize */
    MPI_Barrier(comm);

    /* communicate */
    MPI_Alltoallv(scratch,dfft_nsend, dfft_offset_send, MPI_BYTE,
                  work, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                  comm);
    }

/* Redistribute from group-cyclic with cycle c0 to cycle c0>=c1
 * rev=1 if local order is reversed
 *
 * if rev = 1 and np >= c0 (last stage) it really transforms
 * into a hybrid-distribution, which after the last local ordered
 * DFT becomes the cyclic distribution
 */
void dfft_redistribute_cyclic_to_block_1d(int *dim,
                     int *pdim,
                     int ndim,
                     int current_dim,
                     int c0,
                     int c1,
                     int* pidx,
                     int rev,
                     int size_in,
                     int *embed,
                     cpx_t *work,
                     cpx_t *scratch,
                     int *rho_L,
                     int *rho_pk0,
                     int *dfft_nsend,
                     int *dfft_nrecv,
                     int *dfft_offset_send,
                     int *dfft_offset_recv,
                     MPI_Comm comm,
                     int *proc_map,
                     int row_m)
    {
    if (c1 == c0) return;

    /* length along current dimension */
    int length = dim[current_dim]/pdim[current_dim];
    int size = length*c1/c0;
    size = (size ? size : 1);
    int npackets = length/size;

    int stride = size_in/embed[current_dim];

    /* processor index along current dimension */
    int s=pidx[current_dim];
    /* number of procs along current dimension */
    int p=pdim[current_dim];

    size *= stride;

    int offset = 0;
    int recv_size,send_size;
    int j0_local = s%c0;
    int j2_local = s/c0;
    int j0_new_local = s%c1;
    int j2_new_local = s/c1;

    int pdim_tot=1;
    int k;
    for (k = 0; k < ndim; ++k)
        pdim_tot *= pdim[k];

    int i;
    for (i = 0; i < pdim_tot; ++i)
        {
        dfft_nsend[i] = 0;
        dfft_nrecv[i] = 0;
        dfft_offset_send[i] = 0;
        dfft_offset_recv[i] = 0;
        }

    for (i = 0; i < p; ++i)
        {
        int j0_remote = i%c0;
        int j2_remote = i/c0;

        int j0_new_remote = i % c1;
        int j2_new_remote = i/c1;

        /* decision to send and/or receive */
        int send = 0;
        int recv = 0;
        if (rev && (length >= c0))
            {
            /* redistribute into block with reversed processor id
               and swapped-partially reversed local order (the c0 LSB
               of the local index are MSB, and the n/p/c0 MSB
               are LSB and are reversed */
            send = (((j2_new_remote % (p/c0)) == (rho_pk0[j2_local])) ? 1 : 0);
            recv = (((j2_new_local % (p/c0)) == (rho_pk0[j2_remote])) ? 1 : 0);
            }
        else
            {
            send = (((j2_new_remote / (c0/c1)) == j2_local) && ((j0_local % c1)==j0_new_remote) ? 1 : 0);
            recv = (((j2_new_local / (c0/c1)) == j2_remote) &&  ((j0_remote % c1)==j0_new_local) ? 1 : 0);

            if (length*c1 < c0)
                {
                send &= (j0_local/(length*c1) == j2_new_remote % (c0/(length*c1)));
                recv &= (j0_remote/(length*c1) == j2_new_local % (c0/(length*c1)));
                }
            }

        /* offset of first element sent */
        int j1;
        if (length*c1 >= c0)
            {
            j1 = (j2_new_remote % (c0/c1))*length*c1/c0;
            }
        else
            {
            j1 = (j2_new_remote / (c0/(length*c1))) % length;
            }

        if (rev)
            {
            if (length >= c0)
                {
                j1 = j2_new_remote/(p/c0);
                }
            else
                j1 = rho_L[j1];
            }

        /* mirror remote decision to send */
        send_size = (send ? size : 0);
        recv_size = (recv ? size : 0);

        int destproc = 0;
        int k;
        if (row_m)
            {
            for (k = ndim-1; k >=0 ;--k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? i : pidx[k]);
                }
            }
        else
            {
            for (k = 0; k < ndim; ++k)
                {
                destproc *= pdim[k];
                destproc += ((current_dim == k) ? i : pidx[k]);
                }
            }

        int rank = proc_map[destproc];

        dfft_offset_send[rank] = (send ? (stride*j1*sizeof(cpx_t)) : 0);
        if (rev && (length > c0/c1))
            {
            /* we are directly receving into the work buf */
            dfft_offset_recv[rank] = stride*j0_remote*length/c0*sizeof(cpx_t);
            }
        else
            {
            dfft_offset_recv[rank] = offset*sizeof(cpx_t);
            }

        dfft_nsend[rank] = send_size*sizeof(cpx_t);
        dfft_nrecv[rank] = recv_size*sizeof(cpx_t);
        offset += (recv ? size : 0);
        }

    /* we need to pack data if the local input buffer is reversed
       and we are sending more than one element */
    if (rev && (size > stride))
        {
        offset = 0;
        /*#pragma omp ... */
        int i;
        for (i = 0; i <p; ++i)
            {
            int destproc = 0;
            int k;
            if (row_m)
                {
                for (k = ndim-1; k >=0 ;--k)
                    {
                    destproc *= pdim[k];
                    destproc += ((current_dim == k) ? i : pidx[k]);
                    }
                }
            else
                {
                for (k = 0; k < ndim; ++k)
                    {
                    destproc *= pdim[k];
                    destproc += ((current_dim == k) ? i : pidx[k]);
                    }
                }

            int rank = proc_map[destproc];

            int j1_offset = dfft_offset_send[rank]/sizeof(cpx_t)/stride;

            /* we are sending from a tmp buffer/stride */
            dfft_offset_send[rank] = offset*sizeof(cpx_t)*stride;
            int n = dfft_nsend[rank]/stride/sizeof(cpx_t);
            int j;
            for (j = 0; j < n; j++)
                for (k = 0; k < stride; ++ k)
                    scratch[(offset+j)*stride+k] = work[(j1_offset+j*c0)*stride+k];

            offset += n;
            }

        /* perform communication */
        MPI_Barrier(comm);
        MPI_Alltoallv(scratch,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      work, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);
        }
    else
        {
        /* perform communication */
        MPI_Barrier(comm);
        MPI_Alltoallv(work,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      scratch, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);

        /* unpack */
        int r;
        #pragma omp parallel for private(r)
        for (r = 0; r < npackets; ++r)
            {
            int j1, j1_offset, del;
            int j0_remote = j0_new_local + r*c1;
            if (rev && (length >= c0))
                {
                j1_offset = j0_remote*length/c0;
                del = 1;
                }
            else
                {
                j1_offset = j0_remote/c1;
                del = c0/c1;
                }
            int j;
            for (j = 0; j < (size/stride); ++j)
                {
                j1 = j1_offset + j*del;
                int k;
                for (k = 0; k < stride; ++k)
                    work[j1*stride+k] = scratch[r*size+j*stride+k];
                }
            }
        }
    }

/* plan_long: complete local FFT
   plan_short: partial local FFT
   input and output are M-cyclic (M=pdim[current_dim])
   (out-of-place version, overwrites input)
   */
void mpifft1d_dif(int *dim,
            int *pdim,
            int ndim,
            int current_dim,
            int* pidx,
            int inverse,
            int size,
            int *embed,
            cpx_t *in,
            cpx_t *out,
            plan_t plan_short,
            plan_t plan_long,
            int *rho_L,
            int *rho_pk0,
            int *rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int *proc_map,
            int row_m)
    {
    int p = pdim[current_dim];
    int length = dim[current_dim]/pdim[current_dim];
    int st = size/embed[current_dim]*(dim[current_dim]/pdim[current_dim]);

    /* compute stride for column major matrix storage */
    int stride = size/embed[current_dim];

    int c;
    int k0 = length;
    for (c = p; c >1; c /= length)
        {
#if 1
        /* do local out-of-place place FFT (long-distance butterflies) */
        #ifdef FFT1D_SUPPORTS_THREADS
        dfft_local_1dfft(in, out, plan_long, inverse);
        #else
        int i;
        #pragma omp parallel for
        for (i = 0; i < st/length; ++i)
            dfft_local_1dfft(in+i, out+i, plan_long, inverse);
        #endif
        /* apply twiddle factors */
        double alpha = ((double)(pidx[current_dim] %c))/(double)c;
        int j;
        #pragma omp parallel for private(j)
        for (j = 0; j < length; j++)
            {
            double theta = -(double)2.0 * (double)M_PI * alpha/(double) length;
            cpx_t w;
            RE(w) = cos((double)j*theta);
            IM(w) = sin((double)j*theta);

            double sign = ((inverse) ? (-1.0) : 1.0);
            IM(w) *=sign;

            int r;
            for (r = 0; r < stride; ++r)
                {
                cpx_t x = out[j*stride+r];
                cpx_t y;
                RE(y) = RE(x) * RE(w) - IM(x) * IM(w);
                IM(y) = RE(x) * IM(w) + IM(x) * RE(w);

                in[j*stride+r] = y;
                }
            }
        int rev = 1;
#else
        int rev = 0;
#endif

        /* in-place redistribute from group-cyclic c -> c1 */
        int c1 = ((c > length) ? (c/length) : 1);
        k0 = c;
        dfft_redistribute_cyclic_to_block_1d(dim,pdim,ndim,current_dim, c, c1,
            pidx, rev, size, embed, in,out,rho_L,rho_pk0,
            dfft_nsend,dfft_nrecv,dfft_offset_send,dfft_offset_recv,
            comm, proc_map, row_m);
        }

    /* perform remaining short-distance butterflies,
     * out-of-place 1d FFT */
    #ifdef FFT1D_SUPPORTS_THREADS
    dfft_local_1dfft(in, out, plan_short,inverse);
    #else
    int i;
    #pragma omp parallel for
    for (i = 0; i < st/k0; ++i)
        dfft_local_1dfft(in+i, out+i, plan_short, inverse);
    #endif
    }

/* n-dimensional fft routine (in-place)
 */
void mpifftnd_dif(int *dim,
            int *pdim,
            int ndim,
            int* pidx,
            int inv,
            int size_in,
            int *inembed,
            int *oembed,
            cpx_t *work,
            cpx_t *scratch,
            plan_t *plans_short,
            plan_t *plans_long,
            int **rho_L,
            int **rho_pk0,
            int **rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int *proc_map,
            int row_m)
    {
    int size = size_in;
    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* assume input in local column major */
        mpifft1d_dif(dim, pdim,ndim,current_dim,pidx, inv,
            size, inembed, work, scratch, plans_short[current_dim],
            plans_long[current_dim], rho_L[current_dim],
            rho_pk0[current_dim],rho_Lk0[current_dim],
            dfft_nsend,dfft_nrecv,dfft_offset_send,dfft_offset_recv,
            comm,proc_map, row_m);

        int l = dim[current_dim]/pdim[current_dim];
        int stride = size/inembed[current_dim];

        /* transpose local matrix */
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < l; ++i)
            {
            int j;
            for (j = 0; j < stride; ++j)
                {
                int gidx = j+i*stride;
                int new_idx = j*oembed[current_dim]+i;
                work[new_idx] = scratch[gidx];
                }
            }

        /* update size */
        size *= oembed[current_dim];
        size /= inembed[current_dim];
        }
    }

void redistribute_nd(int *dim,
            int *pdim,
            int ndim,
            int* pidx,
            int size,
            int *embed,
            cpx_t *work,
            cpx_t *scratch,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            int c2b,
            MPI_Comm comm,
            int *proc_map,
            int row_m)
    {
    cpx_t *cur_work =work;
    cpx_t *cur_scratch =scratch;

    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* redistribute along one dimension (in-place) */
        if (!c2b)
            dfft_redistribute_block_to_cyclic_1d(dim, pdim, ndim, current_dim,
                1, pdim[current_dim], pidx, size, embed,
                cur_work, cur_scratch, dfft_nsend,dfft_nrecv,
                dfft_offset_send, dfft_offset_recv, comm, proc_map, row_m);
        else
            dfft_redistribute_cyclic_to_block_1d(dim, pdim, ndim, current_dim,
                pdim[current_dim], 1, pidx, 0, size, embed, cur_work,
                cur_scratch, NULL, NULL, dfft_nsend,
                dfft_nrecv, dfft_offset_send, dfft_offset_recv, comm, proc_map, row_m);

        int l = dim[current_dim]/pdim[current_dim];
        int stride = size/embed[current_dim];

        /* transpose local matrix from column major to row major */
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < l; ++i)
            {
            int j;
            for (j = 0; j < stride; ++j)
                {
                int gidx = j+i*stride;
                int new_idx = j*embed[current_dim]+i;
                cur_scratch[new_idx] =cur_work[gidx];
                }
            }

        /* swap buffers */
        cpx_t *tmp;
        tmp = cur_scratch;
        cur_scratch = cur_work;
        cur_work = tmp;
        }

    if (ndim % 2)
        {
        memcpy(work, scratch, sizeof(cpx_t)*size);
        }
    }


/*****************************************************************************
 * Distributed FFT interface
 *****************************************************************************/
int dfft_execute(cpx_t *h_in, cpx_t *h_out, int dir, dfft_plan p)
    {
    /* only works on host plans */
    if (p.device) return 2;

    int out_of_place = (h_in == h_out) ? 0 : 1;

    cpx_t *scratch, *work;

    if (out_of_place)
        {
        work = p.scratch;
        scratch = p.scratch_2;
        memcpy(work, h_in, p.size_in*sizeof(cpx_t));
        }
    else
        {
        scratch = p.scratch;
        /*! FIXME need to ensure in buf size >= scratch_size */
        work = h_in;
        }

    if ((!dir && !p.input_cyclic) || (dir && !p.output_cyclic))
        {
        /* redistribution of input */
        redistribute_nd(p.gdim, p.pdim, p.ndim, p.pidx,
            p.size_in, p.inembed, work, scratch, p.nsend,p.nrecv,
            p.offset_send,p.offset_recv, 0, p.comm, p.proc_map, p.row_m);
        }

    /* multi-dimensional FFT */
    mpifftnd_dif(p.gdim, p.pdim, p.ndim, p.pidx, dir,
        p.size_in,p.inembed,p.oembed, work, scratch,
        dir ? p.plans_short_inverse : p.plans_short_forward,
        dir ? p.plans_long_inverse : p.plans_long_forward,
        p.rho_L, p.rho_pk0, p.rho_Lk0, p.nsend,p.nrecv,
        p.offset_send,p.offset_recv, p.comm, p.proc_map, p.row_m);

    if ((dir && !p.input_cyclic) || (!dir && !p.output_cyclic))
        {
        /* redistribution of output */
        redistribute_nd(p.gdim, p.pdim, p.ndim, p.pidx,
            p.size_out,p.oembed, work, scratch, p.nsend,p.nrecv,
            p.offset_send,p.offset_recv, 1, p.comm, p.proc_map, p.row_m);
        }

    if (out_of_place)
        {
        memcpy(h_out, work, sizeof(cpx_t)*p.size_out);
        }

    return 0;
    }

int dfft_create_plan(dfft_plan *p,
    int ndim, int *gdim, int *inembed, int *oembed,
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map)
    {
    return dfft_create_plan_common(p, ndim, gdim, inembed,
        oembed, pdim, pidx, row_m,
        input_cyclic, output_cyclic, comm, proc_map, 0);
    }

void dfft_destroy_plan(dfft_plan plan)
    {
    dfft_destroy_plan_common(plan, 0);
    }
