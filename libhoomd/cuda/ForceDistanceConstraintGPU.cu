/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#include "ForceDistanceConstraintGPU.cuh"

#include <cuda_runtime_api.h>
#include <cusparse.h>

/*! \file ForceDistanceConstraingGPU.cu
    \brief Defines GPU kernel code for pairwise distance constraints on the GPU
*/

//! Kernel to fill the matrix for the linear constraint equation
__global__ void gpu_fill_matrix_vector_kernel(unsigned int n_constraint,
                                              unsigned int nptl_local,
                                              Scalar *d_matrix,
                                              Scalar *d_vec,
                                              const Scalar4 *d_pos,
                                              const Scalar4 *d_vel,
                                              const Scalar4 *d_netforce,
                                              const group_storage<2> *d_gpu_clist,
                                              const Index2D gpu_clist_indexer,
                                              const unsigned int *d_gpu_n_constraints,
                                              const unsigned int *d_gpu_cpos,
                                              const typeval_union *d_group_typeval,
                                              Scalar deltaT,
                                              const BoxDim box,
                                              bool fill_ones)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // if beyond the number of local ptls, exit early
    if (idx >= nptl_local)
        return;

    // load number of constraints per this ptl
    unsigned int n_constraint_ptl = d_gpu_n_constraints[idx];

    // small O(N^2) loop over ptl connectivity
    for (unsigned int cidx_i = 0; cidx_i < n_constraint_ptl; cidx_i++)
        {
        group_storage<2> cur_constraint_i = d_gpu_clist[gpu_clist_indexer(idx, cidx_i)];

        // the other ptl in the constraint
        unsigned int cur_constraint_idx_i = cur_constraint_i.idx[0];

        // constraint index
        unsigned int n = cur_constraint_i.idx[1];

        // the constraint distance
        Scalar d = d_group_typeval[n].val;

        // indices of constrained ptls in correct order
        unsigned int idx_na, idx_nb;
        unsigned int cpos = d_gpu_cpos[gpu_clist_indexer(idx, cidx_i)];
        if (cpos == 0)
            {
            idx_na = idx;
            idx_nb = cur_constraint_idx_i;
            }
        else
            {
            idx_na = cur_constraint_idx_i;
            idx_nb = idx;
            }

        // constraint separation
        vec3<Scalar> rn(vec3<Scalar>(d_pos[idx_na])-vec3<Scalar>(d_pos[idx_nb]));

        // apply minimum image
        rn = box.minImage(rn);

        // get masses
        Scalar ma = d_vel[idx_na].w;
        Scalar mb = d_vel[idx_nb].w;

        // constraint time derivative
        vec3<Scalar> rndot(vec3<Scalar>(d_vel[idx_na]) - vec3<Scalar>(d_vel[idx_nb]));

        // constraint separation at t+2*deltaT
        vec3<Scalar> qn(rn + deltaT*rndot);

        #if 0
        for (unsigned int cidx_j = 0; cidx_j < n_constraint_ptl; cidx_j++)
            {
            group_storage<2> cur_constraint_j = d_gpu_clist[gpu_clist_indexer(idx, cidx_j)];

            // the other ptl in the constraint
            unsigned int cur_constraint_idx_j = cur_constraint_j.idx[0];

            // indices of constrained ptls in correct order
            unsigned int idx_ma, idx_mb;
            if (d_gpu_cpos[gpu_clist_indexer(idx, cidx_j)] == 0)
                {
                idx_ma = idx;
                idx_mb = cur_constraint_idx_j;
                }
            else
                {
                idx_ma = cur_constraint_idx_j;
                idx_mb = idx;
                }

            // other constraint index
            unsigned int m = cur_constraint_j.idx[1];

            // constraint separation
            vec3<Scalar> rm(vec3<Scalar>(d_pos[idx_ma])-vec3<Scalar>(d_pos[idx_mb]));

            rm = box.minImage(rm);

            // the constraint matrix element
            Scalar mat_element(0.0);

            if (idx_na == idx_ma)
                {
                if (!fill_ones)
                    {
                    mat_element += Scalar(4.0)*dot(qn,rm)/ma;
                    }
                else
                    {
                    // fill with dummy value
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_na == idx_mb)
                {
                if (!fill_ones)
                    {
                    mat_element -= Scalar(4.0)*dot(qn,rm)/ma;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_nb == idx_ma)
                {
                if (!fill_ones)
                    {
                    mat_element -= Scalar(4.0)*dot(qn,rm)/mb;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_nb == idx_mb)
                {
                if (!fill_ones)
                    {
                    mat_element += Scalar(4.0)*dot(qn,rm)/mb;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }

            // write out matrix element once per unique particle pair (ghost particles occur only once)
            if (cpos == 0 || idx_na >= nptl_local || idx_nb >= nptl_local)
                {
                // matrix in column major
                d_matrix[m*n_constraint+n] = mat_element;
                }
            }
        #endif

        // load number of constraints per this ptl
        unsigned int n_constraint_i = d_gpu_n_constraints[cur_constraint_idx_i];

        for (unsigned int cidx_j = 0; cidx_j < n_constraint_i; cidx_j++)
            {
            group_storage<2> cur_constraint_j = d_gpu_clist[gpu_clist_indexer(cur_constraint_idx_i, cidx_j)];

            // the other ptl in the constraint
            unsigned int cur_constraint_idx_j = cur_constraint_j.idx[0];

            // indices of constrained ptls in correct order
            unsigned int idx_ma, idx_mb;
            if (d_gpu_cpos[gpu_clist_indexer(cur_constraint_idx_i, cidx_j)] == 0)
                {
                idx_ma = cur_constraint_idx_i;
                idx_mb = cur_constraint_idx_j;
                }
            else
                {
                idx_ma = cur_constraint_idx_j;
                idx_mb = cur_constraint_idx_i;
                }

            unsigned int m = cur_constraint_j.idx[1];

            // constraint separation
            vec3<Scalar> rm(vec3<Scalar>(d_pos[idx_ma])-vec3<Scalar>(d_pos[idx_mb]));

            rm = box.minImage(rm);

            Scalar mat_element(0.0);
            if (idx_na == idx_ma)
                {
                if (!fill_ones)
                    {
                    mat_element += Scalar(4.0)*dot(qn,rm)/ma;
                    }
                else
                    {
                    // fill with dummy value
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_na == idx_mb)
                {
                if (!fill_ones)
                    {
                    mat_element -= Scalar(4.0)*dot(qn,rm)/ma;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_nb == idx_ma)
                {
                if (!fill_ones)
                    {
                    mat_element -= Scalar(4.0)*dot(qn,rm)/mb;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }
            if (idx_nb == idx_mb)
                {
                if (!fill_ones)
                    {
                    mat_element += Scalar(4.0)*dot(qn,rm)/mb;
                    }
                else
                    {
                    mat_element = Scalar(1.0);
                    }
                }

            // write out matrix element once per unique particle pair (ghost particles occur only once)
            if (cpos == 0 || idx_na >= nptl_local || idx_nb >= nptl_local)
                {
                // matrix in column major
                d_matrix[m*n_constraint+n] = mat_element;
                }
            }

        if (!fill_ones)
            {
            if (cpos == 0 || idx_na >= nptl_local || idx_nb >= nptl_local)
                {
                // fill vector component
                d_vec[n] = (dot(qn,qn)-d*d)/deltaT/deltaT
                    + Scalar(2.0)*dot(qn, vec3<Scalar>(d_netforce[idx_na])/ma-vec3<Scalar>(d_netforce[idx_nb])/mb);
                }
            }
        }
    }

cudaError_t gpu_fill_matrix_vector(unsigned int n_constraint,
                          unsigned int nptl_local,
                          Scalar *d_matrix,
                          Scalar *d_vec,
                          const Scalar4 *d_pos,
                          const Scalar4 *d_vel,
                          const Scalar4 *d_netforce,
                          const group_storage<2> *d_gpu_clist,
                          const Index2D & gpu_clist_indexer,
                          const unsigned int *d_gpu_n_constraints,
                          const unsigned int *d_gpu_cpos,
                          const typeval_union *d_group_typeval,
                          Scalar deltaT,
                          const BoxDim box,
                          bool connectivity_changed,
                          unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)gpu_fill_matrix_vector_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // run configuration
    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = nptl_local/run_block_size + 1;

    // reset RHS matrix (b)
    cudaMemset(d_vec, 0, sizeof(Scalar)*n_constraint);

    // reset A matrix
    cudaMemset(d_matrix, 0, sizeof(Scalar)*n_constraint*n_constraint);

    // run GPU kernel
    gpu_fill_matrix_vector_kernel<<<n_blocks, run_block_size>>>(
        n_constraint,
        nptl_local,
        d_matrix,
        d_vec,
        d_pos,
        d_vel,
        d_netforce,
        d_gpu_clist,
        gpu_clist_indexer,
        d_gpu_n_constraints,
        d_gpu_cpos,
        d_group_typeval,
        deltaT,
        box,
        connectivity_changed);

    return cudaSuccess;
    }

__global__ void gpu_fill_constraint_forces_kernel(unsigned int nptl_local,
                                        const Scalar4 *d_pos,
                                        const group_storage<2> *d_gpu_clist,
                                        const Index2D gpu_clist_indexer,
                                        const unsigned int *d_gpu_n_constraints,
                                        const unsigned int *d_gpu_cpos,
                                        Scalar *d_lagrange,
                                        Scalar4 *d_force,
                                        const BoxDim box)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= nptl_local)
        return;

    // load number of constraints per this ptl
    unsigned int n_constraint_ptl = d_gpu_n_constraints[idx];

    // the accumulated force on this ptl
    vec3<Scalar> f(0.0,0.0,0.0);

    // iterate over constraints involving ptl with index idx
    for (unsigned int cidx = 0; cidx < n_constraint_ptl; cidx++)
        {
        group_storage<2> cur_constraint = d_gpu_clist[gpu_clist_indexer(idx, cidx)];

        // the other ptl in the constraint
        unsigned int cur_constraint_idx = cur_constraint.idx[0];

        // group idx
        unsigned int n = cur_constraint.idx[1];

        // position of ptl in constraint
        unsigned int cpos = d_gpu_cpos[gpu_clist_indexer(idx, cidx)];

        // indices of constrained ptls in correct order
        unsigned int idx_na, idx_nb;
        if (cpos == 0)
            {
            idx_na = idx;
            idx_nb = cur_constraint_idx;
            }
        else
            {
            idx_na = cur_constraint_idx;
            idx_nb = idx;
            }

        // constraint separation
        vec3<Scalar> rn(vec3<Scalar>(d_pos[idx_na])-vec3<Scalar>(d_pos[idx_nb]));

        // apply minimum image
        rn = box.minImage(rn);

        if (idx < nptl_local)
            {
            // write out force
            if (cpos == 0)
                {
                f += -Scalar(2.0)*d_lagrange[n]*rn;
                }
            else
                {
                f += Scalar(2.0)*d_lagrange[n]*rn;
                }
            }
        }

    d_force[idx] = make_scalar4(f.x,f.y,f.z,Scalar(0.0));
    }

cudaError_t gpu_compute_constraint_forces(unsigned int n_constraint,
                                   Scalar *d_matrix,
                                   Scalar *d_vec,
                                   int *d_nnz,
                                   int &nnz,
                                   const Scalar4 *d_pos,
                                   const group_storage<2> *d_gpu_clist,
                                   const Index2D& gpu_clist_indexer,
                                   const unsigned int *d_gpu_n_constraints,
                                   const unsigned int *d_gpu_cpos,
                                   Scalar4 *d_force,
                                   const BoxDim box,
                                   unsigned int nptl_local,
                                   unsigned int block_size,
                                   cusparseHandle_t cusparse_handle,
                                   cusparseMatDescr_t cusparse_mat_descr,
                                   cusparseSolveAnalysisInfo_t cusparse_solve_info,
                                   bool connectivity_changed,
                                   int *d_csr_rowptr,
                                   int *d_csr_colind,
                                   Scalar *d_csr_val,
                                   Scalar *d_lagrange)
    {
    // convert dense matrix to compressed sparse row
    if (connectivity_changed)
        {
        // count zeros
        #ifdef SINGLE_PRECISION
        cusparseSnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, n_constraint, n_constraint,
            cusparse_mat_descr, d_matrix, n_constraint, d_nnz, &nnz);
        #else
        cusparseDnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, n_constraint, n_constraint,
            cusparse_mat_descr, d_matrix, n_constraint, d_nnz, &nnz);
        #endif
        }

    // update values in CSR format
    #ifdef SINGLE_PRECISION
    cusparseSdense2csr(cusparse_handle, n_constraint, n_constraint, cusparse_mat_descr, d_matrix, n_constraint, d_nnz,
        d_csr_val, d_csr_rowptr, d_csr_colind);
    #else
    cusparseDdense2csr(cusparse_handle, n_constraint, n_constraint, cusparse_mat_descr, d_matrix, n_constraint, d_nnz,
        d_csr_val, d_csr_rowptr, d_csr_colind);
    #endif

    if (connectivity_changed)
        {
        // run the expensive analysis routine
        #ifdef SINGLE_PRECISION
        cusparseScsrsv_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            n_constraint, nnz, cusparse_mat_descr, d_csr_val, d_csr_rowptr, d_csr_colind, cusparse_solve_info);
        #else
        cusparseDcsrsv_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            n_constraint, nnz, cusparse_mat_descr, d_csr_val, d_csr_rowptr, d_csr_colind, cusparse_solve_info);
        #endif
        }

    // solve the sparse systems of linear equations
    Scalar one(1.0);
    #ifdef SINGLE_PRECISION
    cusparseScsrsv_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_constraint,
        &one, cusparse_mat_descr, d_csr_val, d_csr_rowptr, d_csr_colind, cusparse_solve_info, d_vec, d_lagrange);
    #else
    cusparseDcsrsv_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_constraint,
        &one, cusparse_mat_descr, d_csr_val, d_csr_rowptr, d_csr_colind, cusparse_solve_info, d_vec, d_lagrange);
    #endif

    // d_vec contains the Lagrange multipliers

    // fill out force array
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)gpu_fill_constraint_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // run configuration
    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = nptl_local/run_block_size + 1;

    // invoke kernel
    gpu_fill_constraint_forces_kernel<<<n_blocks,run_block_size>>>(nptl_local,
        d_pos,
        d_gpu_clist,
        gpu_clist_indexer,
        d_gpu_n_constraints,
        d_gpu_cpos,
        d_lagrange,
        d_force,
        box);

    return cudaSuccess;
    }
