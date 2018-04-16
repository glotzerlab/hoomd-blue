/* Copyright (c) 2015, Michael P. Howard. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *      1. Redistributions of source code must retain the above copyright
 *         notice, this list of conditions and the following disclaimer.
 *
 *      2. Redistributions in binary form must reproduce the above copyright
 *         notice, this list of conditions and the following disclaimer in the
 *         documentation and/or other materials provided with the distribution.
 *
 *      3. Neither the name of the copyright holder nor the names of its
 *         contributors may be used to endorse or promote products derived from
 *         this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

#ifndef BVLS_SRC_BVLS_SOLVER_H_
#define BVLS_SRC_BVLS_SOLVER_H_

#include "Eigen/Eigen/Dense"
#include <vector>
#include <stdexcept>

//! Solves Bounded Variable Least Squares problem
/*!
 * Implements the algorithm described in P.B. Stark and R.L. Parker, "Bounded-Variable Least Squares: an Algorithm and
 * Applications", Computational Statistics, 10(2), 129--141, 1995. PDF available online at:
 * https://www.stat.berkeley.edu/~stark/Preprints/bvls.pdf.
 *
 * The BVLS optimization problem is defined by
 * \f[
 * \min_{\mathbf{l} \le \mathbf{x} \le \mathbf{u}} || \mathbf{A}\mathbf{x} - \mathbf{b} ||
 * \f]
 *
 * BVLSSolver is dependent on Eigen for linear algebra operations.
 */
class BVLSSolver
    {
    public:
        //! Constructor
        BVLSSolver(const Eigen::MatrixXd& A,
                   const Eigen::VectorXd& b,
                   const Eigen::VectorXd& lower_bound,
                   const Eigen::VectorXd& upper_bound);

        //! Destructor
        ~BVLSSolver() {}

        //! Solves the defined problem
        const Eigen::VectorXd& solve();

        //! Check if a converged solution has been obtained
        /*!
         * \returns true if a solution has been found, false otherwise
         */
        bool converged() const
            {
            return m_found_solution;
            }

        //! Get the solution vector
        /*!
         * \returns n element solution vector
         * \warning An exception is thrown if a solution has not yet been determined.
         */
        const Eigen::VectorXd& getSolution() const
            {
            if (!m_found_solution)
                {
                throw std::runtime_error("Converged solution has not been found");
                }
            return m_x;
            }

        //! Get the value of the objective function at the minimum
        /*!
         * \returns Value of the objective function
         * \warning An exception is thrown if a solution has not yet been determined.
         */
        double getObjectiveMinimum() const
            {
            if (!m_found_solution)
                {
                throw std::runtime_error("Converged solution has not been found");
                }
            return m_obj_val;
            }

        //! Get the number of iterations required to converge to the minimum
        /*!
         * \returns Number of iterations
         * \warning An exception is thrown if a solution has not yet been determined.
         */
        unsigned int getNIterations() const
            {
            if (!m_found_solution)
                {
                throw std::runtime_error("Converged solution has not been found");
                }
            return m_n_iter;
            }

        //! Set the matrix A in the objective function
        /*!
         * \param A m x n matrix
         * \note Solutions are easier if A is full column rank (m >= n), so that there are fewer variables n than
         *       equations m.
         */
        void setMatrix(const Eigen::MatrixXd& A)
            {
            m_A = A;
            }

        //! Set the target b in the objective function
        /*!
         * \param b m element column vector
         * The length of \a b should be equal to the number of equations.
         */
        void setTarget(const Eigen::VectorXd& b)
            {
            m_b = b;
            }

        //! Set the lower bound on the solution vector
        /*!
         * \param lower_bound n element column vector
         * The length of \a lower_bound should be equal to the number of variables.
         */
        void setLowerBound(const Eigen::VectorXd& lower_bound)
            {
            m_lower_bound = lower_bound;
            }

        //! Set the upper bound on the solution vector
        /*!
         * \param upper_bound n element column vector
         * The length of \a upper_bound should be equal to the number of variables.
         */
        void setUpperBound(const Eigen::VectorXd& upper_bound)
            {
            m_upper_bound = upper_bound;
            }

        //! Set the maximum number of iterations
        /*!
         * \param max_iter The maximum number of iterations to attempt to minimize
         * By default, up to 100 iterations are attempted.
         */
        void setMaxIterations(unsigned int max_iter)
            {
            m_max_iter = max_iter;
            }

    private:
        // problem definition
        Eigen::MatrixXd m_A;            //!< The matrix
        Eigen::VectorXd m_b;            //!< The target vector
        Eigen::VectorXd m_lower_bound;  //!< Lower bound on variables
        Eigen::VectorXd m_upper_bound;  //!< Upper bound on variables

        // problem solution
        Eigen::VectorXd m_x;            //!< The solution vector
        double m_obj_val;               //!< Value of the objective function at solution
        unsigned int m_n_iter;          //!< Number of iterations required for solution
        bool m_found_solution;          //!< Flag if a solution was found

        // temporary variables
        Eigen::VectorXd m_w;            //!< Negative gradient at current x
        Eigen::VectorXd m_b_prime;      //!< Residual vector for free variables
        Eigen::VectorXd m_z;            //!< Least squares solution for free variables
        unsigned int m_max_iter;        //!< Maximum number of iterations to attempt

        unsigned int m_n_free;      //!< Number of free variables
        std::vector<int> m_free;    //!< Indexes of all free variables
        int m_next_bound_free;      //!< Index of the next variable to free from the bound vector
        int m_last_freed_var;       //!< Index of the last variable freed

        unsigned int m_n_bound;     //!< Number of bound variables
        std::vector<int> m_bound;   //!< Indexes of all bound variables (offset by one and signed to indicate upper or lower set)
        
        //! Initializes memory and variables for solving the problem
        void initialize();
        
        //! Finds the next variable to free
        bool findNextFree();
        
        //! Frees the next bound variable marked for freeing
        /*!
         * Fills in the hole in the bound variables by popping off the end
         */
        void freeNextVariable()
            {
            int j = abs(m_bound[m_next_bound_free]) - 1;

            m_bound[m_next_bound_free] = m_bound.back();
            m_bound.pop_back(); --m_n_bound;
            m_free.push_back(j);
            ++m_n_free;
            
            m_b_prime += m_A.col(j) * m_x(j);
            }
        
        //! Solves the least squares minimization for the free variables
        bool solveFreeLeastSquares();
        
        //! Applies interpolation to the bound variables
        void interpolateConstraints(const std::vector<int>::iterator& first_out_of_bounds);
        
        //! Performs common wrap-up routines to finish the solver
        /*!
         * \param n_iter Number of iterations required to converge solution
         */
        void saveSolution(unsigned int n_iter)
            {
            m_found_solution = true;
            m_obj_val = (m_A * m_x - m_b).norm();
            m_n_iter = n_iter;
            }
    };

#endif // BVLS_SRC_BVLS_SOLVER_H_
