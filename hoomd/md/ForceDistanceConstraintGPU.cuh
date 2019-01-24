// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "hoomd/HOOMDMath.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"

#include <cusparse.h>

#ifndef __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__
#define __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__

cudaError_t gpu_fill_matrix_vector(unsigned int n_constraint,
                          unsigned int nptl_local,
                          double *d_matrix,
                          double *d_vec,
                          double *d_csr_val,
                          const int *d_csr_idxlookup,
                          unsigned int *d_sparsity_pattern_changed,
                          Scalar rel_tol,
                          unsigned int *d_constraint_violated,
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
                          unsigned int block_size);

cudaError_t gpu_count_nnz(unsigned int n_constraint,
                           double *d_matrix,
                           int *d_nnz,
                           int &nnz,
                           cusparseHandle_t cusparse_handle,
                           cusparseMatDescr_t cusparse_mat_descr);

cudaError_t gpu_dense2sparse(unsigned int n_constraint,
                               double *d_matrix,
                               int *d_nnz,
                               cusparseHandle_t cusparse_handle,
                               cusparseMatDescr_t cusparse_mat_descr,
                               int *d_csr_rowptr,
                               int *d_csr_colind,
                               double *d_csr_val);

cudaError_t gpu_compute_constraint_forces(const Scalar4 *d_pos,
                                   const group_storage<2> *d_gpu_clist,
                                   const Index2D & gpu_clist_indexer,
                                   const unsigned int *d_gpu_n_constraints,
                                   const unsigned int *d_gpu_cpos,
                                   Scalar4 *d_force,
                                   Scalar *d_virial,
                                   unsigned int virial_pitch,
                                   const BoxDim box,
                                   unsigned int nptl_local,
                                   unsigned int block_size,
                                   double *d_lagrange);
#endif
