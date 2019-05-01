// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceDistanceConstraint.h"

#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

#ifdef CUSOLVER_AVAILABLE
#include <cusparse.h>

// CUDA 7.0
#include <cusolverRf.h>
#include <cusolverSp.h>

// CUDA 7.5
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#endif

/*! \file ForceDistanceConstraint.h
    \brief Declares a class to implement pairwise distance constraint
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ForceDistanceConstraintGPU_H__
#define __ForceDistanceConstraintGPU_H__

#include "hoomd/GPUVector.h"

/*! Implements a pairwise distance constraint on the GPU

    See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class ForceDistanceConstraintGPU : public ForceDistanceConstraint
    {
    public:
        //! Constructs the compute
        ForceDistanceConstraintGPU(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~ForceDistanceConstraintGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ForceDistanceConstraint::setAutotunerParams(enable, period);

            m_tuner_fill->setPeriod(period);
            m_tuner_force->setPeriod(period);

            m_tuner_fill->setEnabled(enable);
            m_tuner_force->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_fill;  //!< Autotuner for filling the constraint matrix
        std::unique_ptr<Autotuner> m_tuner_force; //!< Autotuner for populating the force array

        #ifdef CUSOLVER_AVAILABLE
        cusparseHandle_t m_cusparse_handle;                //!< cuSPARSE handle
        cusparseMatDescr_t m_cusparse_mat_descr;           //!< Persistent matrix descriptor
        cusparseMatDescr_t m_cusparse_mat_descr_L;         //!< L matrix descriptor
        cusparseMatDescr_t m_cusparse_mat_descr_U;         //!< U matrix descriptor
        cusolverRfHandle_t m_cusolver_rf_handle;           //!< cusolverRf handle
        cusolverSpHandle_t m_cusolver_sp_handle;           //!< cusolverSp handle
        csrluInfoHost_t m_cusolver_csrlu_info;             //!< Opaque handle for cusolver LU decomp
        bool m_cusolver_rf_initialized;                    //!< True if we have a cusolverRF handle

        std::vector<int> m_Qreorder;                       //!< permutation matrix
        std::vector<char> m_reorder_work;                  //!< Work buffer for reordering
        std::vector<int> m_mapBfromA;                      //!< Map vector
        std::vector<int> m_csr_rowptr_B;                   //!< Row offsets for sparse matrix B
        std::vector<int> m_csr_colind_B;                   //!< Column index for sparse matrix B
        std::vector<double> m_csr_val_B;                   //!< Values for sparse matrix B
        std::vector<char> m_lu_work;                       //!< Work buffer for host LU decomp
        std::vector<double> m_bhat;                        //!< Reordered RHS vector
        std::vector<double> m_xhat;                        //!< Solution to reordered equation system

        int m_nnz_L_tot;                   //!< Number of non-zeros in L
        int m_nnz_U_tot;                   //!< Number of non-zeros in U
        std::vector<int> m_Plu;            //!< Permutation P
        std::vector<int> m_Qlu;            //!< Permutation Q

        GPUVector<double> m_csr_val_L;     //!< Values of sparse matrix L
        GPUVector<int> m_csr_rowptr_L;     //!< Row offset of sparse matrix L
        GPUVector<int> m_csr_colind_L;     //!< Column index of sparse matrix L

        GPUVector<double> m_csr_val_U;     //!< Values of sparse matrix U
        GPUVector<int> m_csr_rowptr_U;     //!< Row offset of sparse matrix U
        GPUVector<int> m_csr_colind_U;     //!< Column index of sparse matrix U

        GPUVector<int> m_P;                //!< reordered permutation P
        GPUVector<int> m_Q;                //!< reordered permutation Q
        GPUVector<double> m_T;             //!< cusolverRf working space

        GPUVector<int> m_nnz;              //!< Vector of number of non-zero elements per row
        int m_nnz_tot;                     //!< Total number of non-zero elements
        GPUVector<int> m_csr_rowptr;       //!< Row offset for CSR
        GPUVector<int> m_csr_colind;       //!< Column index for CSR
        #endif

        GPUVector<double> m_sparse_val;    //!< Sparse matrix value list

        //! Populate the quantities in the constraint-force equation
        virtual void fillMatrixVector(unsigned int timestep);

        //! Solve the matrix equation
        virtual void solveConstraints(unsigned int timestep);

        //! Compute the constraint forces using the Lagrange multipliers
        virtual void computeConstraintForces(unsigned int timestep);
    };

//! Exports the ForceDistanceConstraint to python
void export_ForceDistanceConstraintGPU(pybind11::module& m);

#endif
