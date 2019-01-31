// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "MolecularForceCompute.h"

/*! \file ForceDistanceConstraint.h
    \brief Declares a class to implement pairwise distance constraint
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ForceDistanceConstraint_H__
#define __ForceDistanceConstraint_H__

#include "hoomd/GPUVector.h"
#include "hoomd/GPUFlags.h"

#include "hoomd/extern/Eigen/Eigen/Dense"
#include "hoomd/extern/Eigen/Eigen/SparseLU"

/*! Implements a pairwise distance constraint using the algorithm of

    [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, “A Non-Iterative Matrix Method for Constraint Molecular Dynamics Simulations,” Mol. Simul., vol. 13, no. 6, pp. 395–405, 1994.
    [2] M. Yoneya, “A Generalized Non-iterative Matrix Method for Constraint Molecular Dynamics Simulations,” J. Comput. Phys., vol. 172, no. 1, pp. 188–197, Sep. 2001.

    See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class PYBIND11_EXPORT ForceDistanceConstraint : public MolecularForceCompute
    {
    public:
        //! Constructs the compute
        ForceDistanceConstraint(std::shared_ptr<SystemDefinition> sysdef);

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
        std::shared_ptr<ConstraintData> m_cdata; //! The constraint data

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

        bool m_constraint_reorder;         //!< True if groups have changed
        bool m_constraints_added_removed;  //!< True if global constraint topology has changed

        Scalar m_d_max;                    //!< Maximum constraint extension

        //! Compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Populate the quantities in the constraint-force equation
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

        #ifdef ENABLE_MPI
        //! Set the communicator object
        virtual void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            // call base class method to set m_comm
            MolecularForceCompute::setCommunicator(comm);

            if (!m_comm_ghost_layer_connected)
                {
                // register this class with the communicator
                m_comm->getGhostLayerWidthRequestSignal().connect<ForceDistanceConstraint, &ForceDistanceConstraint::askGhostLayerWidth>(this);
                m_comm_ghost_layer_connected = true;
                }
           }
        #endif

    private:
        //! Helper function to perform a depth-first search
        Scalar dfs(unsigned int iconstraint, unsigned int molecule, std::vector<int>& visited,
            unsigned int *label, std::vector<ConstraintData::members_t>& groups, std::vector<Scalar>& length);

        #ifdef ENABLE_MPI
        bool m_comm_ghost_layer_connected = false; //!< Track if we have already connected to ghost layer width requests
        #endif

    };

//! Exports the ForceDistanceConstraint to python
void export_ForceDistanceConstraint(pybind11::module& m);

#endif
