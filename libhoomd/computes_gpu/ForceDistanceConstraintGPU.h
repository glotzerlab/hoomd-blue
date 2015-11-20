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

#include "ForceDistanceConstraint.h"

#include "Autotuner.h"
#include "util/mgpucontext.h"

#include <boost/signals2.hpp>

#include <cusparse.h>
#include <cusolverRf.h>
#include <cusolverSp.h>

// CUDA 7.5
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

/*! \file ForceDistanceConstraint.h
    \brief Declares a class to implement pairwise distance constraint
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ForceDistanceConstraintGPU_H__
#define __ForceDistanceConstraintGPU_H__

#include "GPUVector.h"

/*! Implements a pairwise distance constraint on the GPU

    See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class ForceDistanceConstraintGPU : public ForceDistanceConstraint
    {
    public:
        //! Constructs the compute
        ForceDistanceConstraintGPU(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~ForceDistanceConstraintGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_fill->setPeriod(period);
            m_tuner_force->setPeriod(period);

            m_tuner_fill->setEnabled(enable);
            m_tuner_force->setEnabled(enable);
            }

    protected:
        boost::scoped_ptr<Autotuner> m_tuner_fill;  //!< Autotuner for filling the constraint matrix
        boost::scoped_ptr<Autotuner> m_tuner_force; //!< Autotuner for populating the force array

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

        bool m_constraints_dirty;          //!< True if groups have changed
        GPUVector<int> m_nnz;              //!< Vector of number of non-zero elements per row
        int m_nnz_tot;                     //!< Total number of non-zero elements
        GPUVector<double> m_csr_val;       //!< Matrix values in compressed sparse row (CSR) format
        GPUVector<int> m_csr_rowptr;       //!< Row offset for CSR
        GPUVector<int> m_csr_colind;       //!< Column index for CSR

        //! Connection to the signal notifying when groups are resorted
        boost::signals2::connection m_constraints_dirty_connection;

        //! Populate the quantities in the constraint-force equatino
        virtual void fillMatrixVector(unsigned int timestep);

        //! Solve the linear matrix-vector equation
        virtual void computeConstraintForces(unsigned int timestep);

        //! Method called when constraint order changes
        virtual void slotConstraintsDirty()
            {
            m_constraints_dirty = true;
            }
    };

//! Exports the ForceDistanceConstraint to python
void export_ForceDistanceConstraintGPU();

#endif
