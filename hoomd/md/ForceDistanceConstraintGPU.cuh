// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef CUSOLVER_AVAILABLE
#include <cusparse.h>
#endif

#ifndef __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__
#define __FORCE_DISTANCE_CONSTRAINT_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_fill_matrix_vector(unsigned int n_constraint,
                                  unsigned int nptl_local,
                                  double* d_matrix,
                                  double* d_vec,
                                  double* d_csr_val,
                                  const int* d_csr_idxlookup,
                                  unsigned int* d_sparsity_pattern_changed,
                                  Scalar rel_tol,
                                  unsigned int* d_constraint_violated,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const Scalar4* d_netforce,
                                  const group_storage<2>* d_gpu_clist,
                                  const Index2D& gpu_clist_indexer,
                                  const unsigned int* d_gpu_n_constraints,
                                  const unsigned int* d_gpu_cpos,
                                  const typeval_union* d_group_typeval,
                                  Scalar deltaT,
                                  const BoxDim box,
                                  unsigned int block_size);

#ifdef CUSOLVER_AVAILABLE

hipError_t gpu_count_nnz(unsigned int n_constraint,
                         double* d_matrix,
                         int* d_nnz,
                         int& nnz,
                         cusparseHandle_t cusparse_handle,
                         cusparseMatDescr_t cusparse_mat_descr);
#ifndef CUSPARSE_NEW_API
hipError_t gpu_dense2sparse(unsigned int n_constraint,
                            double* d_matrix,
                            int* d_nnz,
                            cusparseHandle_t cusparse_handle,
                            cusparseMatDescr_t cusparse_mat_descr,
                            int* d_csr_rowptr,
                            int* d_csr_colind,
                            double* d_csr_val);
#endif
#endif

hipError_t gpu_compute_constraint_forces(const Scalar4* d_pos,
                                         const group_storage<2>* d_gpu_clist,
                                         const Index2D& gpu_clist_indexer,
                                         const unsigned int* d_gpu_n_constraints,
                                         const unsigned int* d_gpu_cpos,
                                         Scalar4* d_force,
                                         Scalar* d_virial,
                                         size_t virial_pitch,
                                         const BoxDim box,
                                         unsigned int nptl_local,
                                         unsigned int block_size,
                                         double* d_lagrange);
#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
