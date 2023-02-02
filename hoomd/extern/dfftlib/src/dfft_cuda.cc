#include "dfft_cuda.h"

#include <hip/hip_runtime.h>

#include <mpi.h>

#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include "dfft_common.h"
#include "dfft_cuda.h"
#include "dfft_cuda.cuh"

#include <stdio.h>
#define CHECK_CUDA() \
    {                                                                       \
    hipDeviceSynchronize();                                                \
    hipError_t err = hipGetLastError();                                   \
    if (err != hipSuccess)                                                 \
        {                                                                   \
        printf("CUDA Error in file %s, line %d: %s\n", __FILE__,__LINE__,   \
            hipGetErrorString(err));                                       \
        exit(1);                                                            \
        }                                                                   \
    }                                                                       \

#ifdef __HIP_PLATFORM_HCC__
#define CHECK_LOCAL_FFT(res) \
    {                                                                          \
    if (res != HIPFFT_SUCCESS)                                                 \
        {                                                                      \
        printf("Local FFT failed, error code %d, file %s, line %d.\n",res, __FILE__,__LINE__); \
        assert(!res);                                                          \
        exit(1);                                                               \
        }                                                                      \
    }
#else
#define CHECK_LOCAL_FFT(res) \
    {                                                                          \
    if (res != CUFFT_SUCCESS)                                                 \
        {                                                                      \
        printf("Local FFT failed, error code %d, file %s, line %d.\n",res, __FILE__,__LINE__); \
        assert(!res);                                                          \
        exit(1);                                                               \
        }                                                                      \
    }
#endif


/*****************************************************************************
 * Implementation of the distributed FFT
 *****************************************************************************/

/*
 * n-dimensional redistribute from group-cyclic with cycle c0 to cycle c1
 * 1 <=c0,c1 <= pdim[i]
 */
void dfft_cuda_redistribute_nd( dfft_plan *plan,int stage, int size_in, int *embed, int *d_embed, int dir,
                  cuda_cpx_t *d_work, int **rho_c0, int **rho_c1, int **rho_plc0c1, int **rho_pc0,
                  int **rho_pc1, int **rho_c0c1pl)
    {
    /* exit early if nothing needs to be done */
    int res = 0;
    int i;
    for (i = 0; i < plan->ndim; ++i)
        if (!(plan->c0[stage][i] == plan->c1[stage][i])
            || plan->rev_global[stage][i]) res = 1;
    if (!res) return;

    int pdim_tot=1;
    int k,t;
    for (k = 0; k < plan->ndim; ++k)
        pdim_tot *= plan->pdim[k];
    for (t = 0; t<pdim_tot; ++t)
        {
        plan->nsend[t] = 0;
        plan->nrecv[t] = 0;
        plan->offset_send[t] = 0;
        plan->offset_recv[t] = 0;
        }

    int roffs = 0;
    int soffs = 0;
    for (t = 0; t < pdim_tot; ++t)
        {
        int send_size = 1;
        int recv_size = 1;

        /* send and recv flags */
        int send =1;
        int recv=1;
        int current_dim;
        int tmp = t;
        int tmp_pidx = pdim_tot;
        for (current_dim = 0; current_dim < plan->ndim; ++current_dim)
            {
            /* find coordinate of remote procesor along current dimension */
            if (!plan->row_m)
                {
                tmp_pidx /= plan->pdim[current_dim];
                i = tmp / tmp_pidx;
                tmp %= tmp_pidx;
                }
            else
                {
                i = (tmp % plan->pdim[current_dim]);
                tmp/=plan->pdim[current_dim];
                }
            int length = plan->gdim[current_dim]/plan->pdim[current_dim];

            /* processor index along current dimension */
            int c0 = plan->c0[stage][current_dim];
            int c1 = plan->c1[stage][current_dim];

            int s = plan->pidx[current_dim];
            int j0_local = s % c0;
            int j2_local = s / c0;
            int j0_new_local = s % c1;
            int j2_new_local = s / c1;
            int p = plan->pdim[current_dim];

            int ratio;
            /* dir == 1: block to cyclic,
               dir == 0: cyclic to block */
            if (dir)
                ratio = c1/c0;
            else
                ratio = c0/c1;

            int size;


            /* initialize send offsets */
            int j0_remote = i % c0;
            int j2_remote = i / c0;
            int j0_new_remote = i % c1;
            int j2_new_remote = i / c1;

            if (dir)
                {
                send &= ((j0_local == (j0_new_remote % c0))
                    && (j2_new_remote == j2_local / ratio));
                recv &= ((j0_remote == (j0_new_local % c0))
                    && (j2_new_local == j2_remote / ratio));
                }
            else
                {
                /* assume dir == 0 */
                if (!plan->rev_global[stage][current_dim])
                    {
                    send &= (((j0_local % c1) == j0_new_remote)
                        && (j2_local == (j2_new_remote/ratio)));
                    recv &= (((j0_remote % c1) == j0_new_local)
                        && (j2_remote == (j2_new_local/ratio)));
                    }
                else
                    {
                    /* global bitreversed output */
                    if (p/c1 > c0)
                        {
                        /* this section is usually not called during a DFFT */
                        k = c0*c1/plan->pdim[i];
                        send &= ((j0_local == rho_c0[current_dim][j2_new_remote/k]) &&
                            (rho_c1[current_dim][j0_new_remote] == j2_local/k));
                        recv &= ((j0_remote == rho_c0[current_dim][j2_new_local/k]) &&
                            (rho_c1[current_dim][j0_new_local] == j2_remote/k));

                        if (p/c1 > length*c0)
                            {
                            k = p/(length*c0*c1);
                            send &= (rho_plc0c1[current_dim][j2_new_remote%k]
                                == (j2_local % k));
                            recv &= (rho_plc0c1[current_dim][j2_new_local%k]
                                == (j2_remote % k));
                            }
                        }
                    else
                        {
                        k = c0*c1/p;
                        if (p/c1 > 1)
                            {
                            send &= (((rho_pc1[current_dim][j2_new_remote] == j0_local%(p/c1))
                                &&(rho_pc0[current_dim][j0_new_remote % (p/c0)] == j2_local)));
                            recv &= (((rho_pc1[current_dim][j2_new_local] == j0_remote%(p/c1))
                                && (rho_pc0[current_dim][j0_new_local % (p/c0)] == j2_remote)));
                            }
                        else
                            {
                            send &= (((j2_new_remote == j0_local%(p/c1))
                                &&(rho_pc0[current_dim][j0_new_remote % (p/c0)] == j2_local)));
                            recv &= (((j2_new_local == j0_remote%(p/c1))
                                && (rho_pc0[current_dim][j0_new_local % (p/c0)] == j2_remote)));
                            }

                        if (p*length/c1 < c0)
                            {
                            /* this section is usually not called during a DFFT */
                            k = c0*c1/p/length;
                            send &= (rho_c0c1pl[current_dim][j0_new_remote/(c1/k)] ==
                                j0_local/(c0/k));
                            recv &= (rho_c0c1pl[current_dim][j0_new_local/(c1/k)] ==
                                j0_remote/(c0/k));
                            }
                        }
                    } /* rev_global */
                } /* dir */
            if (!plan->rev_global[stage][current_dim] && (ratio >= length))
                {
                if (dir)
                    {
                    send &= ((j0_new_remote / (length*c0))
                        == (j2_local % (ratio/length)));
                    recv &= ((j0_new_local / (length*c0))
                        == (j2_remote % (ratio/length)));
                    }
                else
                    {
                    send &= ((j0_local / (length*c1))
                        == (j2_new_remote % (ratio/length)));
                    recv &= ((j0_remote / (length*c1))
                        == (j2_new_local % (ratio/length)));
                    }
                }

            /* determine packet length for current dimension */
            if (! plan->rev_global[stage][current_dim])
                {
                if (ratio >= length)
                    size = 1;
                else
                    size = length/ratio;
                }
            else
                {
                if (p/c1 >= c0)
                    {
                    // usually not entered
                    size = ((p/c1 <= length*c0) ? (length*c0*c1/p) : 1);
                    }
                else
                    {
                    size = ((length*p/c1 >= c0) ? (length*p/c1/c0) : 1);
                    }
                }
            recv_size *= (recv ? size : 0);
            send_size *= (send ? size : 0);
            } /* end loop over dimensions */

        int rank = plan->proc_map[t];
        plan->nsend[rank] = (unsigned int)(send_size*sizeof(cuda_cpx_t));
        plan->nrecv[rank] = (unsigned int)(recv_size*sizeof(cuda_cpx_t));
        plan->offset_send[rank] = (unsigned int)(soffs*sizeof(cuda_cpx_t));
        plan->offset_recv[rank] = (unsigned int)(roffs*sizeof(cuda_cpx_t));
        roffs += recv_size;
        soffs += send_size;
        } /* end loop over processors */

    /* pack data */
    if (dir)
        {
        gpu_b2c_pack_nd(size_in, plan->d_c0[stage], plan->d_c1[stage], plan->ndim, d_embed,
            plan->d_length, plan->row_m, d_work, plan->d_scratch);
        if (plan->check_cuda_errors) CHECK_CUDA();
        }
    else
        {
        gpu_c2b_pack_nd(size_in, plan->d_c0[stage], plan->d_c1[stage], plan->ndim, d_embed,
            plan->d_length, plan->row_m, plan->d_pdim, plan->d_rev_j1[stage], plan->d_rev_global[stage],
            d_work, plan->d_scratch);
        if (plan->check_cuda_errors) CHECK_CUDA();
        }

    /* synchronize */
    MPI_Barrier(plan->comm);

    /* communicate */
    // stage into host buf
    hipMemcpy(plan->h_stage_in, plan->d_scratch, sizeof(cuda_cpx_t)*size_in,hipMemcpyDefault);
    if (plan->check_cuda_errors) CHECK_CUDA();

    MPI_Alltoallv(plan->h_stage_in,plan->nsend, plan->offset_send, MPI_BYTE,
                  plan->h_stage_out,plan->nrecv, plan->offset_recv, MPI_BYTE,
                  plan->comm);

    // copy back received data
    hipMemcpy(plan->d_scratch_2,plan->h_stage_out, sizeof(cuda_cpx_t)*size_in,hipMemcpyDefault);
    if (plan->check_cuda_errors) CHECK_CUDA();

    /* unpack data */
    if (dir)
        {
        gpu_b2c_unpack_nd(size_in, plan->d_c0[stage], plan->d_c1[stage], plan->ndim, d_embed,
            plan->d_length, plan->row_m, plan->d_scratch_2, d_work);
        if (plan->check_cuda_errors) CHECK_CUDA();
        }
    else
        {
        gpu_c2b_unpack_nd(size_in, plan->d_c0[stage], plan->d_c1[stage], plan->ndim, d_embed,
            plan->d_length, plan->row_m, plan->d_pdim, plan->d_rev_global[stage],
            plan->d_rev_partial[stage], plan->d_scratch_2, d_work);
        if (plan->check_cuda_errors) CHECK_CUDA();
        }
    }


/*
 * Redistribute from group-cyclic with cycle c0 to cycle c1>=c0
 */
void dfft_cuda_redistribute_block_to_cyclic_1d(
                  int *dim,
                  int *pdim,
                  int ndim,
                  int current_dim,
                  int c0,
                  int c1,
                  int* pidx,
                  int size_in,
                  int *embed,
                  cuda_cpx_t *d_work,
                  cuda_cpx_t *d_scratch,
                  cuda_cpx_t *h_stage_in,
                  cuda_cpx_t *h_stage_out,
                  int *dfft_nsend,
                  int *dfft_nrecv,
                  int *dfft_offset_send,
                  int *dfft_offset_recv,
                  MPI_Comm comm,
                  int check_err,
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
        dfft_nsend[destproc] = (unsigned int)(size*sizeof(cuda_cpx_t));
        dfft_offset_send[destproc] = (unsigned int)(offset*sizeof(cuda_cpx_t));
        }

    /* pack data */
    gpu_b2c_pack(npackets*size, ratio, size, npackets, stride, d_work, d_scratch);
    if (check_err) CHECK_CUDA();

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
        dfft_nrecv[srcproc] = (unsigned int)(size*sizeof(cuda_cpx_t));
        dfft_offset_recv[srcproc] = (unsigned int)(offset*sizeof(cuda_cpx_t));
        }

    /* synchronize */
    MPI_Barrier(comm);

    /* communicate */
    // stage into host buf
    hipMemcpy(h_stage_in, d_scratch, sizeof(cuda_cpx_t)*npackets*size,hipMemcpyDefault);
    if (check_err) CHECK_CUDA();

    MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                  h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                  comm);

    // copy back received data
    hipMemcpy(d_work,h_stage_out, sizeof(cuda_cpx_t)*size_in,hipMemcpyDefault);
    if (check_err) CHECK_CUDA();
    }

/* Redistribute from group-cyclic with cycle c0 to cycle c0>=c1
 * rev=1 if local order is reversed
 *
 * if rev = 1 and np >= c0 (last stage) it really transforms
 * into a hybrid-distribution, which after the last local ordered
 * DFT becomes the cyclic distribution
 */
void dfft_cuda_redistribute_cyclic_to_block_1d(int *dim,
                     int *pdim,
                     int ndim,
                     int current_dim,
                     int c0,
                     int c1,
                     int* pidx,
                     int rev,
                     int size_in,
                     int *embed,
                     cuda_cpx_t *d_work,
                     cuda_cpx_t *d_scratch,
                     cuda_cpx_t *h_stage_in,
                     cuda_cpx_t *h_stage_out,
                     int *rho_L,
                     int *rho_pk0,
                     int *dfft_nsend,
                     int *dfft_nrecv,
                     int *dfft_offset_send,
                     int *dfft_offset_recv,
                     MPI_Comm comm,
                     int check_err,
                     int row_m
                     )
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

        dfft_offset_send[destproc] = (unsigned int)((send ? (stride*j1*sizeof(cuda_cpx_t)) : 0));
        if (rev && (length > c0/c1))
            {
            /* we are directly receving into the work buf */
            dfft_offset_recv[destproc] = (unsigned int)(stride*j0_remote*length/c0*sizeof(cuda_cpx_t));
            }
        else
            {
            dfft_offset_recv[destproc] = (unsigned int)(offset*sizeof(cuda_cpx_t));
            }

        dfft_nsend[destproc] = (unsigned int)(send_size*sizeof(cuda_cpx_t));
        dfft_nrecv[destproc] = (unsigned int)(recv_size*sizeof(cuda_cpx_t));
        offset+=(recv ? size : 0);
        }

    /* we need to pack data if the local input buffer is reversed
       and we are sending more than one element */
    if (rev && (size > stride))
        {
        offset = 0;
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

            /* we are sending from a tmp buffer/stride */
            dfft_offset_send[destproc] = (unsigned int)(offset*sizeof(cuda_cpx_t)*stride);
            int n = (unsigned int)(dfft_nsend[destproc]/stride/sizeof(cuda_cpx_t));
            offset += n;
            }

        /* pack data */
        gpu_b2c_pack(size_in, c0, size, c0, stride, d_work, d_scratch);
        if (check_err) CHECK_CUDA();

        /* perform communication */
        MPI_Barrier(comm);
        // stage into host buf
        hipMemcpy(h_stage_in, d_scratch, sizeof(cuda_cpx_t)*length*stride,hipMemcpyDefault);
        if (check_err) CHECK_CUDA();

        MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);

        // copy back received data
        hipMemcpy(d_work,h_stage_out, sizeof(cuda_cpx_t)*npackets*size,hipMemcpyDefault);
        if (check_err) CHECK_CUDA();
        }
    else
        {
        /* perform communication */
        MPI_Barrier(comm);
        // stage into host buf
        hipMemcpy(h_stage_in, d_work, sizeof(cuda_cpx_t)*size_in,hipMemcpyDefault);
        if (check_err) CHECK_CUDA();

        MPI_Alltoallv(h_stage_in,dfft_nsend, dfft_offset_send, MPI_BYTE,
                      h_stage_out, dfft_nrecv, dfft_offset_recv, MPI_BYTE,
                      comm);

        // copy back received data
        hipMemcpy(d_scratch,h_stage_out, sizeof(cuda_cpx_t)*npackets*size,hipMemcpyDefault);
        if (check_err) CHECK_CUDA();

        /* unpack */
        gpu_c2b_unpack(npackets*size, length, c0, c1, size, j0_new_local, stride, rev, d_work, d_scratch);
        if (check_err) CHECK_CUDA();
        }
    }

/* plan_long: complete local FFT
   plan_short: partial local FFT
   input and output are M-cyclic (M=pdim[current_dim])
   (out-of-place version, overwrites input)
   */
void cuda_mpifft1d_dif(int *dim,
            int *pdim,
            int ndim,
            int current_dim,
            int* pidx,
            int inverse,
            int size,
            int *embed,
            cuda_cpx_t *d_in,
            cuda_cpx_t *d_out,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            cuda_plan_t plan_short,
            cuda_plan_t plan_long,
            int *rho_L,
            int *rho_pk0,
            int *rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int check_err,
            int row_m)
    {
    int p = pdim[current_dim];
    int length = dim[current_dim]/pdim[current_dim];

    /* compute stride for column major matrix storage */
    int stride = size/embed[current_dim];

    int c;
    for (c = p; c >1; c /= length)
        {
        /* do local out-of-place place FFT (long-distance butterflies) */
        dfft_cuda_local_fft(d_in, d_out, plan_long, inverse);

        /* apply twiddle factors */
        double alpha = ((double)(pidx[current_dim] %c))/(double)c;

        gpu_twiddle(size, length, stride, float(alpha), d_out, d_in, inverse);
        if (check_err) CHECK_CUDA();

        /* in-place redistribute from group-cyclic c -> c1 */
        int rev = 1;
        int c1 = ((c > length) ? (c/length) : 1);
        dfft_cuda_redistribute_cyclic_to_block_1d(dim,pdim,ndim,current_dim,
            c, c1, pidx, rev, size, embed, d_in,d_out,h_stage_in, h_stage_out,
            rho_L,rho_pk0, dfft_nsend,dfft_nrecv,dfft_offset_send,
            dfft_offset_recv, comm, check_err, row_m);
        }

    /* perform remaining short-distance butterflies,
     * out-of-place 1d FFT */
    dfft_cuda_local_fft(d_in, d_out, plan_short,inverse);
    }

/*
 * n-dimensional fft routine, based on 1-d transforms (in-place)
 */
void cuda_mpifftnd_dif(int *dim,
            int *pdim,
            int ndim,
            int* pidx,
            int inv,
            int size_in,
            int *inembed,
            int *oembed,
            cuda_cpx_t *d_work,
            cuda_cpx_t *d_scratch,
            cuda_cpx_t *h_stage_in,
            cuda_cpx_t *h_stage_out,
            cuda_plan_t *plans_short,
            cuda_plan_t *plans_long,
            int **rho_L,
            int **rho_pk0,
            int **rho_Lk0,
            int *dfft_nsend,
            int *dfft_nrecv,
            int *dfft_offset_send,
            int *dfft_offset_recv,
            MPI_Comm comm,
            int check_err,
            int row_m)
    {
    int size = size_in;
    int current_dim;
    for (current_dim = 0; current_dim < ndim; ++current_dim)
        {
        /* assume input in local column major */
        cuda_mpifft1d_dif(dim, pdim,ndim,current_dim,pidx, inv,
            size, inembed, d_work, d_scratch,h_stage_in, h_stage_out,
            plans_short[current_dim],
            plans_long[current_dim], rho_L[current_dim],
            rho_pk0[current_dim],rho_Lk0[current_dim],
            dfft_nsend,dfft_nrecv,dfft_offset_send,dfft_offset_recv,
            comm,check_err,row_m);

        int l = dim[current_dim]/pdim[current_dim];
        int stride = size/inembed[current_dim];

        /* transpose local matrix */
        gpu_transpose(size,l,stride, oembed[current_dim],d_scratch, d_work);
        if (check_err) CHECK_CUDA();

        /* update size */
        size *= oembed[current_dim];
        size /= inembed[current_dim];
        }
    }

/* n-dimensional FFT using local multidimensional FFTs
 * and n-dimensional redistributions
 */
void cuda_fftnd_multi(dfft_plan *p,
                      cuda_cpx_t *d_in,
                      cuda_cpx_t *d_out,
                      cuda_plan_t **cuda_plans_multi,
                      cuda_plan_t *cuda_plans_final,
                      int inv)
    {
    int d,i,j;
    /* initialize current stage */
    if (p->init && p->max_depth > 0)
        {
        for (i = 0; i < p->ndim; ++i)
            p->c0[p->max_depth-1][i] = p->pdim[i];
        }

    int res;
    for (d = p->max_depth-1; d>=0; d--)
        {
        cuda_cpx_t *cur_in = d_in;
        cuda_cpx_t *cur_out = p->d_scratch;
        if (!p->init)
            {
            for (j =0; j < p->n_fft[d]; ++j)
                {
                if (p->depth[j] > d)
                    {
                    /* do local FFT */
                    res = dfft_cuda_local_fft(cur_in, cur_out, cuda_plans_multi[d][j], inv);
                    CHECK_LOCAL_FFT(res);
                    if (p->check_cuda_errors) CHECK_CUDA();
                    }
                else
                    {
                    /* transpose only */
                    int l = p->gdim[j]/p->pdim[j];
                    int stride = p->size_in/p->inembed[j];

                    gpu_transpose(p->size_in,l,stride, p->inembed[j],cur_in,cur_out);
                    if (p->check_cuda_errors) CHECK_CUDA();
                    }

                /* swap pointers */
                cuda_cpx_t *tmp;
                tmp = cur_in;
                cur_in = cur_out;
                cur_out = tmp;
                }
            }
        else
            {
            /* initialize twiddle factors */
            for (i =0; i < p->ndim; ++i)
                {
                if (p->depth[i] > d)
                    p->h_alpha[d][i] = float(((double)(p->pidx[i] % p->c0[d][i]))/(double)p->c0[d][i]);
                else
                    p->h_alpha[d][i] = 0.0;
                }

            /* copy to device */
            hipMemcpy(p->d_alpha[d], p->h_alpha[d], sizeof(cuda_scalar_t)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            }

        if (!p->init)
            {
            /* twiddle */
            gpu_twiddle_nd(p->size_in, p->ndim, p->d_iembed, p->d_length,
                p->d_alpha[d], cur_in, d_in, inv);
            if (p->check_cuda_errors) CHECK_CUDA();
            }

        if (p->init)
            {
            /* update cycle */
            for (i = 0; i< p->ndim; ++i)
                {
                int length = p->gdim[i] / p->pdim[i];
                /* only update if necessary */
                if (p->depth[i] > d)
                    {
                    if (d >0)
                        {
                        /* decimate in steps of 'length' */
                        p->c1[d][i] = p->c0[d][i]/length;

                        /* the previous FFT produced bit-reversed output compared
                         * to an unordered FFT */
                        p->rev_j1[d][i] = 1;
                        p->rev_global[d][i] = 0;
                        p->rev_partial[d][i] = 0;
                        }
                    else
                        {
                        /* in the last stage, we go back to cyclic, after a bit reversal */
                        p->rev_j1[d][i] = 1;
                        p->rev_global[d][i] = 1;
                        p->rev_partial[d][i] = 1;
                        p->c1[d][i] = p->pdim[i];
                        }
                    }
                else
                    {
                    p->c1[d][i] = p->c0[d][i];
                    p->rev_global[d][i] = 0;
                    p->rev_partial[d][i] = 0;
                    p->rev_j1[d][i] = 0;
                    }
                }

            /* copy to device */
            hipMemcpy(p->d_c0[d], p->c0[d], sizeof(int)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            hipMemcpy(p->d_c1[d], p->c1[d], sizeof(int)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            hipMemcpy(p->d_rev_global[d], p->rev_global[d], sizeof(int)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            hipMemcpy(p->d_rev_partial[d], p->rev_partial[d], sizeof(int)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            hipMemcpy(p->d_rev_j1[d], p->rev_j1[d], sizeof(int)*p->ndim,hipMemcpyDefault);
            CHECK_CUDA();
            }

        if (!p->init)
            {
            /* redistribute */
            dfft_cuda_redistribute_nd(p, d, p->size_in, p->inembed, p->d_iembed,
                      0, d_in, NULL, NULL, NULL, p->rho_pk0, NULL, NULL);
            }

        /* old cycle == new cycle */
        if (p->init && d>0)
            {
            for (i = 0; i < p->ndim; ++i)
                p->c0[d-1][i] = p->c1[d][i];
            }
        }

    /* final stage */
    if (!p->init)
        {
        if (!p->final_multi)
            {
            int size = p->size_in;
            for (i = 0; i < p->ndim; ++i)
                {
                /* do 1d FFT */
                cuda_cpx_t *d_in_ptr = ((i == 0) ? d_in : p->d_scratch);
                res = dfft_cuda_local_fft(d_in_ptr, p->d_scratch_2, cuda_plans_final[i] , inv);
                CHECK_LOCAL_FFT(res);
                if (p->check_cuda_errors) CHECK_CUDA();

                /* transpose */
                int l = p->gdim[i]/p->pdim[i];
                int stride = size/p->inembed[i];

                /* transpose local matrix */
                cuda_cpx_t *d_out_ptr = ((i == p->ndim-1) ? d_out : p->d_scratch);
                gpu_transpose(size,l,stride, p->oembed[i],p->d_scratch_2, d_out_ptr);
                if (p->check_cuda_errors) CHECK_CUDA();

                /* update size */
                size *= p->oembed[i];
                size /= p->inembed[i];
                }
            }
        else
            {
            /* do multidimensional fft */
            int res;
            res = dfft_cuda_local_fft(d_in, d_out, cuda_plans_final[0] , inv);
            CHECK_LOCAL_FFT(res);
            if (p->check_cuda_errors) CHECK_CUDA();
            }
        }
    }

void dfft_cuda_redistribute(dfft_plan *plan, int size, int *embed, int *d_embed,
            cuda_cpx_t *d_work, int c2b)
    {
    int i;

    int d = ((c2b) ? (plan->max_depth + 1) : plan->max_depth);

    if (plan->init)
        {
        for (i = 0; i < plan->ndim; ++i)
            {
            /* no bit reversal */
            plan->rev_global[d][i] = 0;
            plan->rev_partial[d][i] = 0;
            plan->rev_j1[d][i] = 0;
            }

        if (!c2b)
            {
            for (i = 0; i < plan->ndim; ++i)
                {
                /* block to cyclic */
                plan->c0[d][i] = 1;
                plan->c1[d][i] = plan->pdim[i];
                }
            }
        else
            {
            for (i = 0; i < plan->ndim; ++i)
                {
                /* cyclic to block */
                plan->c0[d][i] = plan->pdim[i];
                plan->c1[d][i] = 1;
                }
            }

        hipMemcpy(plan->d_c0[d], plan->c0[d], sizeof(int)*plan->ndim,hipMemcpyDefault);
        CHECK_CUDA();
        hipMemcpy(plan->d_c1[d], plan->c1[d], sizeof(int)*plan->ndim,hipMemcpyDefault);
        CHECK_CUDA();
        hipMemcpy(plan->d_rev_global[d], plan->rev_global[d], sizeof(int)*plan->ndim,hipMemcpyDefault);
        CHECK_CUDA();
        hipMemcpy(plan->d_rev_partial[d], plan->rev_partial[d], sizeof(int)*plan->ndim,hipMemcpyDefault);
        CHECK_CUDA();
        hipMemcpy(plan->d_rev_j1[d], plan->rev_j1[d], sizeof(int)*plan->ndim,hipMemcpyDefault);
        CHECK_CUDA();
        }
    else
        {
        int dir = (c2b ? 0 : 1 );
        dfft_cuda_redistribute_nd(plan, d, size, embed, d_embed, dir,
              d_work,  NULL, NULL, NULL, NULL, NULL, NULL);
        }
    }


/*****************************************************************************
 * Distributed FFT interface
 *****************************************************************************/
int dfft_cuda_execute(cuda_cpx_t *d_in, cuda_cpx_t *d_out, int dir, dfft_plan *p)
    {
    int out_of_place = (d_in == d_out) ? 0 : 1;

    int check_err = p->check_cuda_errors;
    cuda_cpx_t *d_work = NULL;

    if (!p->init)
        {
        if (out_of_place)
            {
            d_work = p->d_scratch_3;
            hipMemcpy(d_work, d_in, p->size_in*sizeof(cuda_cpx_t),hipMemcpyDefault);
            if (check_err) CHECK_CUDA();
            }
        else
            {
            d_work = d_in;
            }
        }

    if (p->init || (!dir && !p->input_cyclic) || (dir && !p->output_cyclic))
        {
        /* redistribution of input */
        dfft_cuda_redistribute(p,p->size_in, p->inembed, p->d_iembed, d_work, 0);
        }

    /* multi-dimensional FFT */
    /*cuda_mpifftnd_dif(p.gdim, p.pdim, p.ndim, p.pidx, dir,
        p.size_in,p.inembed,p.oembed, d_work, d_scratch,
        p.h_stage_in, p.h_stage_out,
        dir ? p.cuda_plans_short_inverse : p.cuda_plans_short_forward,
        dir ? p.cuda_plans_long_inverse : p.cuda_plans_long_forward,
        p.rho_L, p.rho_pk0, p.rho_Lk0, p.nsend,p.nrecv,
        p.offset_send,p.offset_recv, p.comm,check_err,p.row_m); */

    cuda_fftnd_multi(p, d_work, d_out,
                     dir ? p->cuda_plans_multi_bw : p->cuda_plans_multi_fw,
                     dir ? p->cuda_plans_final_bw : p->cuda_plans_final_fw,
                     dir);

    if (p->init || (dir && !p->input_cyclic) || (!dir && !p->output_cyclic))
        {
        /* redistribution of output */
        dfft_cuda_redistribute(p,p->size_out, p->oembed, p->d_oembed, d_out, 1);
        }

    return 0;
    }

int dfft_cuda_create_plan(dfft_plan *p,
    int ndim, int *gdim,
    int *inembed, int *oembed,
    int *pdim, int *pidx, int row_m,
    int input_cyclic, int output_cyclic,
    MPI_Comm comm,
    int *proc_map)
    {
    int res = dfft_create_plan_common(p, ndim, gdim, inembed, oembed,
        pdim, pidx, row_m, input_cyclic, output_cyclic, comm, proc_map, 1);

    /* allocate staging bufs */
    /* we need to use posix_memalign/hipHostRegister instead
     * of hipHostMalloc, because hipHostMalloc doesn't have hooks
     * in the MPI library, and using it would lead to data corruption
     */
    int size = (unsigned int)(p->scratch_size*sizeof(cuda_cpx_t));
    int page_size = getpagesize();
    size = ((size + page_size - 1) / page_size) * page_size;
    int retval = posix_memalign((void **)&(p->h_stage_in),page_size,size);
    if (retval != 0)
        return 1;

    retval = posix_memalign((void **)&(p->h_stage_out),page_size,size);
    if (retval != 0)
        return 1;

    hipHostRegister(p->h_stage_in, size, hipHostMallocDefault);
    CHECK_CUDA();
    hipHostRegister(p->h_stage_out, size, hipHostMallocDefault);
    CHECK_CUDA();

    /* allocate memory for passing variables */
   hipMalloc((void **)&(p->d_pidx), sizeof(int)*ndim);
    CHECK_CUDA();
    hipMalloc((void **)&(p->d_pdim), sizeof(int)*ndim);
    CHECK_CUDA();
    hipMalloc((void **)&(p->d_iembed), sizeof(int)*ndim);
    CHECK_CUDA();
    hipMalloc((void **)&(p->d_oembed), sizeof(int)*ndim);
    CHECK_CUDA();
    hipMalloc((void **)&(p->d_length), sizeof(int)*ndim);
    CHECK_CUDA();

    /* initialize cuda buffers */
    int *h_length = (int *)malloc(sizeof(int)*ndim);
    int i;
    for (i = 0; i < ndim; ++i)
        h_length[i] = gdim[i]/pdim[i];
    hipMemcpy(p->d_pidx, pidx, sizeof(int)*ndim, hipMemcpyDefault);
    CHECK_CUDA();
    hipMemcpy(p->d_pdim, pdim, sizeof(int)*ndim, hipMemcpyDefault);
    CHECK_CUDA();
    hipMemcpy(p->d_iembed, p->inembed, sizeof(int)*ndim, hipMemcpyDefault);
    CHECK_CUDA();
    hipMemcpy(p->d_oembed, p->oembed, sizeof(int)*ndim, hipMemcpyDefault);
    CHECK_CUDA();
    hipMemcpy(p->d_length, h_length, sizeof(int)*ndim, hipMemcpyDefault);
    CHECK_CUDA();
    free(h_length);

    int dmax = p->max_depth + 2;
    p->d_rev_j1 = (int **) malloc(sizeof(int *)*dmax);
    p->d_rev_global = (int **) malloc(sizeof(int *)*dmax);
    p->d_rev_partial = (int **) malloc(sizeof(int *)*dmax);
    p->d_c0 = (int **) malloc(sizeof(int *)*dmax);
    p->d_c1 = (int **) malloc(sizeof(int *)*dmax);
    if (p->max_depth)
        {
        p->h_alpha = (cuda_scalar_t **) malloc(sizeof(cuda_scalar_t *)*p->max_depth);
        p->d_alpha = (cuda_scalar_t **) malloc(sizeof(cuda_scalar_t *)*p->max_depth);
        }

    int d;
    for (d = 0; d < dmax; ++d)
        {
        hipMalloc((void **)&(p->d_rev_j1[d]), sizeof(int)*ndim);
        CHECK_CUDA();
        hipMalloc((void **)&(p->d_rev_partial[d]), sizeof(int)*ndim);
        CHECK_CUDA();
        hipMalloc((void **)&(p->d_rev_global[d]), sizeof(int)*ndim);
        CHECK_CUDA();
        hipMalloc((void **)&(p->d_c0[d]), sizeof(int)*ndim);
        CHECK_CUDA();
        hipMalloc((void **)&(p->d_c1[d]), sizeof(int)*ndim);
        CHECK_CUDA();
        }

    for (d = 0; d < p->max_depth; ++d)
        {
        hipMalloc((void **)&(p->d_alpha[d]), sizeof(cuda_scalar_t)*ndim);
        CHECK_CUDA();
        p->h_alpha[d] = (cuda_scalar_t *) malloc(sizeof(cuda_scalar_t)*ndim);
        }

    /* perform initialization run */
    dfft_cuda_execute(NULL, NULL, 0, p);

    /* initialization finished */
    p->init = 0;

    return res;
    }

void dfft_cuda_destroy_plan(dfft_plan plan)
    {
    dfft_destroy_plan_common(plan, 1);
    hipHostUnregister(plan.h_stage_in);
    hipHostUnregister(plan.h_stage_out);
    free(plan.h_stage_in);
    free(plan.h_stage_out);
    int dmax = plan.max_depth + 2;
    int d;
    for (d = 0; d < dmax; ++d)
        {
        hipFree(plan.d_rev_j1[d]);
        hipFree(plan.d_rev_partial[d]);
        hipFree(plan.d_rev_global[d]);
        hipFree(plan.d_c0[d]);
        hipFree(plan.d_c1[d]);
        }
    for (d = 0; d < plan.max_depth; ++d)
        {
        hipFree(plan.d_alpha[d]);
        free(plan.h_alpha[d]);
        }
    free(plan.d_c0);
    free(plan.d_c1);
    free(plan.d_rev_j1);
    free(plan.d_rev_partial);
    free(plan.d_rev_global);
    if (plan.max_depth)
        {
        free(plan.d_alpha);
        free(plan.h_alpha);
        }

    hipFree(plan.d_pidx);
    hipFree(plan.d_pdim);
    hipFree(plan.d_iembed);
    hipFree(plan.d_oembed);
    hipFree(plan.d_length);
    }

void dfft_cuda_check_errors(dfft_plan *plan, int check_err)
    {
    plan->check_cuda_errors = check_err;
    }
