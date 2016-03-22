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

#include "MolecularForceCompute.h"

/*! \file ForceDistanceConstraint.h
    \brief Declares a class to implement pairwise distance constraint
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ForceDistanceConstraint_H__
#define __ForceDistanceConstraint_H__

#include "hoomd/GPUVector.h"
#include "hoomd/GPUFlags.h"

#include "hoomd/extern/Eigen/Dense"
#include "hoomd/extern/Eigen/SparseLU"

/*! Implements a pairwise distance constraint using the algorithm of

    [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, “A Non-Iterative Matrix Method for Constraint Molecular Dynamics Simulations,” Mol. Simul., vol. 13, no. 6, pp. 395–405, 1994.
    [2] M. Yoneya, “A Generalized Non-iterative Matrix Method for Constraint Molecular Dynamics Simulations,” J. Comput. Phys., vol. 172, no. 1, pp. 188–197, Sep. 2001.

    See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class ForceDistanceConstraint : public MolecularForceCompute
    {
    public:
        //! Constructs the compute
        ForceDistanceConstraint(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~ForceDistanceConstraint();

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved()
            {
            return m_cdata->getNGlobal();
            }

        //! Set the relative tolerance for constraint warnings
        void setRelativeTolerance(Scalar rel_tol)
            {
            m_rel_tol = rel_tol;
            }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep);
        #endif

        //! Assign global molecule tags
        virtual void assignMoleculeTags();

    protected:
        boost::shared_ptr<ConstraintData> m_cdata; //! The constraint data

        GPUVector<double> m_cmatrix;                //!< The matrix for the constraint force equation (column-major)
        GPUVector<double> m_cvec;                   //!< The vector on the RHS of the constraint equation
        GPUVector<double> m_lagrange;               //!< The solution for the lagrange multipliers

        Scalar m_rel_tol;                           //!< Rel. tolerance for constraint violation warning
        GPUFlags<unsigned int> m_constraint_violated; //!< The id of the violated constraint + 1

        GPUFlags<unsigned int> m_condition; //!< ==1 if sparsity pattern has changed
        Eigen::SparseMatrix<double, Eigen::ColMajor> m_sparse;    //!< The sparse constraint matrix representation
        Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>, Eigen::COLAMDOrdering<int> > m_sparse_solver;
            //!< The persistent state of the sparse matrix solver
        GPUVector<int> m_sparse_idxlookup;          //!< Reverse lookup from column-major to sparse matrix element

        //! Connection to the signal notifying when groups are resorted
        boost::signals2::connection m_constraint_reorder_connection;

        //!< Connection to the signal for global topology changes
        boost::signals2::connection m_group_num_change_connection;

        bool m_constraint_reorder;         //!< True if groups have changed
        bool m_constraints_added_removed;  //!< True if global constraint topology has changed

        Scalar m_d_max;                    //!< Maximum constraint extension

        //! Compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Populate the quantities in the constraint-force equatino
        virtual void fillMatrixVector(unsigned int timestep);

        //! Check violation of constraints
        virtual void checkConstraints(unsigned int timestep);

        //! Solve the constraint matrix equation
        virtual void solveConstraints(unsigned int timestep);

        //! Solve the linear matrix-vector equation
        virtual void computeConstraintForces(unsigned int timestep);

        //! Method called when constraint order changes
        virtual void slotConstraintReorder()
            {
            m_constraint_reorder = true;
            }

        //! Method called when constraint order changes
        virtual void slotConstraintsAddedRemoved()
            {
            m_constraints_added_removed = true;
            }

        //! Returns the requested ghost layer width for all types
        /*! \param type the type for which we are requesting info
         */
        virtual Scalar askGhostLayerWidth(unsigned int type);

        //! Fill the molecule list
        virtual void initMolecules();

        #ifdef ENABLE_MPI
        //! Set the communicator object
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            // call base class method to set m_comm
            MolecularForceCompute::setCommunicator(comm);

            if (!m_comm_ghost_layer_connection.connected())
                {
                // register this class with the communciator
                m_comm_ghost_layer_connection = m_comm->addGhostLayerWidthRequest(
                    boost::bind(&ForceDistanceConstraint::askGhostLayerWidth, this, _1));
                }
           }
        #endif

    private:
        //! Helper function to perform a depth-first search
        Scalar dfs(unsigned int iconstraint, unsigned int molecule, std::vector<int>& visited,
            unsigned int *label, std::vector<ConstraintData::members_t>& groups, std::vector<Scalar>& length);

        boost::signals2::connection m_comm_ghost_layer_connection; //!< Connection to be asked for ghost layer width requests

    };

//! Exports the ForceDistanceConstraint to python
void export_ForceDistanceConstraint();

#endif
