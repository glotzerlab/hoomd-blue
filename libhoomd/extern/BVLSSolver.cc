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

#include "BVLSSolver.h"

using namespace std;
using namespace Eigen;

/*!
 * \param A m equation x n variable matrix
 * \param b m element target column vector
 * \param lower_bound n element column vector
 * \param upper_bound n element column vector
 */
BVLSSolver::BVLSSolver(const MatrixXd& A,
                       const VectorXd& b,
                       const VectorXd& lower_bound,
                       const VectorXd& upper_bound)
    : m_A(A), m_b(b), m_lower_bound(lower_bound), m_upper_bound(upper_bound), m_obj_val(0.0), m_n_iter(0),
      m_found_solution(false), m_max_iter(100), m_n_free(0), m_next_bound_free(-1), m_last_freed_var(-1),
      m_n_bound(m_A.cols())
    { }

const Eigen::VectorXd& BVLSSolver::solve()
    {
    // initialize the problem
    initialize();

    for (unsigned int cur_iter = 0; cur_iter < m_max_iter; ++cur_iter)
        {
        // step 3a: quit if all variables are free after at least one iteration
        if (cur_iter > 0 && m_n_free == m_A.cols())
            {
            saveSolution(cur_iter);
            return m_x;
            }

        // step 2: calculate the residual and use it to get the gradient
        // step 6a: prefill b'
        m_b_prime = m_b - m_A * m_x;
        m_w = m_A.transpose() * m_b_prime;
        for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
            {
            m_b_prime += m_A.col(*it) * m_x(*it);
            }

        bool out_of_bounds(false); // flag if variable is out of bounds
        do
            {
            // try to pick a variable to free (could do error checking on the QR decomposition for instabilities
            // but for now we just assume full column rank
            bool lsq_solved(false);
            do
                {
                // if out of bounds is false, then this is the first time through and we need to free a variable                
                if (!out_of_bounds)
                    {
                    // step 3b and 4: t* = max s_t w_t
                    bool pass_kuhn_tucker = findNextFree();
                    if (pass_kuhn_tucker)
                        {
                        saveSolution(cur_iter);
                        return m_x;
                        }

                    // step 5: move t* to f, filling in the hole in the bound variables by popping off the end
                    freeNextVariable();
                    }

                lsq_solved = solveFreeLeastSquares();
                } while (!lsq_solved);

            // reset the last freed variable now that one has successfully been picked (don't need to worry about cycles)
            m_last_freed_var = -1;

            // step 7: check if the solution respects the constraints
            out_of_bounds = false;
            vector<int>::iterator first_out_of_bounds;
            for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
                {
                if (m_z(it-m_free.begin()) < m_lower_bound(*it) || m_z(it-m_free.begin()) > m_upper_bound(*it))
                    {
                    out_of_bounds = true;
                    first_out_of_bounds = it;
                    break;
                    }
                }

            if (!out_of_bounds) // solution is interior, update and repeat loop
                {
                for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
                    {
                    m_x(*it) = m_z(it-m_free.begin());
                    }
                }
            else // proceed to step 8
                {
                interpolateConstraints(first_out_of_bounds);
                }
            } while (m_n_free > 0 && out_of_bounds);
        }
    return m_x;
    }

/*!
 * Validates the BVLS problem definition (checks for consistency of matrix dimensions), and (re-)initializes the
 * solution and temporary vectors. The solution vector is initialized by putting variables at their lower bounds.
 */
void BVLSSolver::initialize()
    {
    // reset solution values
    m_x = VectorXd::Zero(m_A.cols());
    m_obj_val = 0.0;
    m_n_iter = 0;
    m_found_solution = false;

    // reset temporary arrays
    m_w = VectorXd::Zero(m_A.rows());
    m_b_prime = VectorXd::Zero(m_A.rows());

    m_free.clear();
    m_free.reserve(m_A.cols());
    m_n_free = 0;
    m_last_freed_var = -1;
    
    m_bound.clear();
    m_bound.reserve(m_A.cols());
    m_n_bound = m_A.cols();
    m_next_bound_free = -1;

    // validate the b vector
    if (m_b.size() != m_A.rows())
        {
        throw runtime_error("b vector has incorrect dimensions");
        }

    // validate the bounds
    if (m_lower_bound.size() != m_A.cols() || m_upper_bound.size() != m_A.cols())
        {
        throw runtime_error("Bounds have incorrect dimensions");
        }
    for (unsigned int i=0; i < m_A.cols(); ++i)
        {
        if (m_lower_bound(i) >= m_upper_bound(i))
            {
            throw runtime_error("Bounds are reversed");
            }
        }

    // put all variables at the lower bound to start
    // the sign of the bound variable determines if it is at lower or upper, so everything is offset by 1
    for (int j=0; j < (int)m_A.cols(); ++j)
        {
        m_bound.push_back(-(j+1));
        m_x(j) = m_lower_bound(j);
        }
    }

/*!
 * \returns true if current solution passes the Kuhn-Tucker test, false otherwise
 *
 * step 3b and 4: t* = max s_t w_t
 */
bool BVLSSolver::findNextFree()
    {
    double t_star(0.0);
    int t_max_j(-1);
    bool pass_kuhn_tucker(true);
    do
        {
        t_star = 0.0;
        pass_kuhn_tucker = true;
        for (vector<int>::iterator it = m_bound.begin(); it != m_bound.end(); ++it)
            {
            int j = abs(*it) - 1;
            double t(0.0);
            if (*it < 0) // lower
                {
                t = m_w(j);
                if (m_w(j) > 0.0) pass_kuhn_tucker = false;
                }
            else // upper
                {
                t = -m_w(j);
                if (m_w(j) < 0.0) pass_kuhn_tucker = false;
                }

            // argument maximization
            if (t > t_star)
                {
                t_star = t;
                t_max_j = j;
                m_next_bound_free = it - m_bound.begin();
                }
            }

        // last paragraph on p.4
        if (!pass_kuhn_tucker && t_max_j == m_last_freed_var)
            {
            m_w(t_max_j) = 0.0;
            }
        } while (t_max_j == m_last_freed_var && !pass_kuhn_tucker);
    return pass_kuhn_tucker;
    }

bool BVLSSolver::solveFreeLeastSquares()
    {
    // step 6b: construct A'
    MatrixXd A_prime = MatrixXd::Zero(m_A.rows(), m_n_free);
    for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
        {
        A_prime.col(it-m_free.begin()) = m_A.col(*it);
        }
    // step 6c: minimize ||A'z - b'|| using QR decomposition (regular least squares)
    ColPivHouseholderQR<MatrixXd> qr = A_prime.colPivHouseholderQr();
    m_z = qr.solve(m_b_prime);
    
    // check for linear dependence of A' or roundoff errors that would take an attempted freed variable back over
    // the bound it came from (bottom of p.4)
    int last_free_try = m_free.back();
    if ( qr.rank() < m_n_free ||
         (m_z(m_n_free-1) > m_upper_bound(last_free_try) && m_x(last_free_try) >= m_upper_bound(last_free_try)) ||
         (m_z(m_n_free-1) < m_lower_bound(last_free_try) && m_x(last_free_try) <= m_lower_bound(last_free_try)) )
        {
        // determine if the variable is at the lower bound or upper bound
        int sign = (m_x(last_free_try) <= m_lower_bound(last_free_try)) ? -1 : 1;

        // rebind the variable
        m_bound.push_back(sign*(last_free_try+1));
        m_b_prime -= m_x(last_free_try) * m_A.col(last_free_try);
        m_w(last_free_try) = 0.0;
        ++m_n_bound;

        // pop off the tried variable from the end of the free list
        m_free.pop_back();
        --m_n_free;

        return false;
        }

    return true;
    }

void BVLSSolver::interpolateConstraints(const vector<int>::iterator& first_out_of_bounds)
    {
    // alpha is the interpolation factor that brings at least x_j to its boundary, and must lie
    // between 0 and 1 because the operation is done only on points *outside* the region
    double alpha(2.0); // start out of range
    vector<int>::iterator min_alpha_it; // iterator corresponding to the minimum alpha
    bool min_it_at_lower(false); // flag if the iterator is out of bounds lower or upper for resetting

    for (vector<int>::iterator it = first_out_of_bounds; it != m_free.end(); ++it)
        {
        double z_j = m_z(it-m_free.begin());

        double cur_alpha(2.0); // out of range indicator
        if (z_j > m_upper_bound(*it))
            {
            cur_alpha = (m_upper_bound(*it) - m_x(*it))/(z_j - m_x(*it));
            }
        else if (z_j < m_lower_bound(*it))
            {
            cur_alpha = (m_lower_bound(*it) - m_x(*it))/(z_j-m_x(*it));
            }
        if (cur_alpha < alpha)
            {
            alpha = cur_alpha;
            min_alpha_it = it;
            min_it_at_lower = (z_j < m_lower_bound(*it));
            }
        }

    // step 10: interpolate all free points
    for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
        {
        m_x(*it) += alpha * (m_z(it-m_free.begin()) - m_x(*it));
        }

    // step 11: check the points, and move any that are out of bounds to the boundary
    vector<int> new_free;
    new_free.reserve(m_n_free);
    for (vector<int>::iterator it = m_free.begin(); it != m_free.end(); ++it)
        {
        // second condition is needed to satisfy para 2. on p. 4 about making loop finite
        if (m_x(*it) >= m_upper_bound(*it) || (it == min_alpha_it && !min_it_at_lower))
            {
            m_x(*it) = m_upper_bound(*it);
            m_bound.push_back(*it+1);
            ++m_n_bound;
            --m_n_free;
            m_b_prime -= m_upper_bound(*it) * m_A.col(*it);
            }
        else if (m_x(*it) <= m_lower_bound(*it) || (it == min_alpha_it && min_it_at_lower))
            {
            m_x(*it) = m_lower_bound(*it);
            m_bound.push_back(-(*it+1));
            ++m_n_bound;
            --m_n_free;
            m_b_prime -= m_lower_bound(*it) * m_A.col(*it);
            }
        else
            {
            new_free.push_back(*it);
            }
        }

    m_last_freed_var = *min_alpha_it;
    m_free = new_free;
    }
