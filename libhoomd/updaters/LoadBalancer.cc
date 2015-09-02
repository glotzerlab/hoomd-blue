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
// Maintainer: mphoward

/*! \file LoadBalancer.cc
    \brief Defines the LoadBalancer class
*/

#ifdef ENABLE_MPI
#include "LoadBalancer.h"
#include "Communicator.h"

#include "BVLSSolver.h"
#include "Eigen/Dense"

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

/*!
 * \param sysdef System definition
 */
LoadBalancer::LoadBalancer(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<BalancedDomainDecomposition> decomposition)
        : Updater(sysdef), m_decomposition(decomposition), m_mpi_comm(m_exec_conf->getMPICommunicator()),
          m_N_own(m_pdata->getN()), m_needs_recount(false), m_tolerance(Scalar(1.05)), m_maxiter(1),
          m_enable_x(true), m_enable_y(true), m_enable_z(true), m_max_scale(Scalar(0.05))
    {
    m_exec_conf->msg->notice(5) << "Constructing LoadBalancer" << endl;

    // now all the stuff with the reductions
    const Index3D& di = m_decomposition->getDomainIndexer();
    uint3 my_grid_pos = m_decomposition->getGridPos();

    // setup the MPI_Comms that reduce 3D to two 2D planes
    MPI_Comm_split(m_mpi_comm, di(my_grid_pos.x, my_grid_pos.y, 0), my_grid_pos.z, &m_mpi_comm_xy);
    MPI_Comm_split(m_mpi_comm, di(my_grid_pos.x, 0, my_grid_pos.z), my_grid_pos.y, &m_mpi_comm_xz);

    // get the world group and then select only those ranks that are in the xy plane
    MPI_Comm_group(m_mpi_comm, &m_mpi_comm_group);

    // allocate one group and communicator for every slice along each dimension that is not reduced
    m_mpi_group_xy_red_y.resize(di.getW()); m_mpi_comm_xy_red_y.resize(di.getW());
    m_mpi_group_xy_red_x.resize(di.getH()); m_mpi_comm_xy_red_x.resize(di.getH());
    m_mpi_group_xz_red_x.resize(di.getD()); m_mpi_comm_xz_red_x.resize(di.getD());

    // lists to hold the rank ids along each dimension corresponding to a given slice
    int *x_ranks = new int[di.getW()];
    int *y_ranks = new int[di.getH()];
    int *z_ranks = new int[di.getD()];
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);

    // xy reduced down y
    for (unsigned int i=0; i < di.getW(); ++i)
        {
        // we are stuffing the ranks into the communicator in the Cartesian order, so the root should be 0
        for (unsigned int j=0; j < di.getH(); ++j)
            {
            y_ranks[j] = h_cart_ranks.data[di(i,j,0)];
            }
        MPI_Group_incl(m_mpi_comm_group, di.getH(), y_ranks, &m_mpi_group_xy_red_y[i]);
        MPI_Comm_create(m_mpi_comm, m_mpi_group_xy_red_y[i], &m_mpi_comm_xy_red_y[i]);
        
        // save the ranks of the x roots for making a new communicator
        x_ranks[i] = h_cart_ranks.data[di(i,0,0)];
        }
    // create the communicator to gather down x
    MPI_Group_incl(m_mpi_comm_group, di.getW(), x_ranks, &m_mpi_group_x);
    MPI_Comm_create(m_mpi_comm, m_mpi_group_x, &m_mpi_comm_x);

    // xy reduced down x
    for (unsigned int j=0; j < di.getH(); ++j)
        {
        for (unsigned int i=0; i < di.getW(); ++i)
            {
            x_ranks[i] = h_cart_ranks.data[di(i,j,0)];
            }
        MPI_Group_incl(m_mpi_comm_group, di.getW(), x_ranks, &m_mpi_group_xy_red_x[j]);
        MPI_Comm_create(m_mpi_comm, m_mpi_group_xy_red_x[j], &m_mpi_comm_xy_red_x[j]);

        y_ranks[j] = h_cart_ranks.data[di(0,j,0)];
        }
    MPI_Group_incl(m_mpi_comm_group, di.getH(), y_ranks, &m_mpi_group_y);
    MPI_Comm_create(m_mpi_comm, m_mpi_group_y, &m_mpi_comm_y);

    // xz reduced down x
    for (unsigned int k=0; k < di.getD(); ++k)
        {
        for (unsigned int i=0; i < di.getW(); ++i)
            {
            x_ranks[i] = h_cart_ranks.data[di(i,0,k)];
            }
        MPI_Group_incl(m_mpi_comm_group, di.getW(), x_ranks, &m_mpi_group_xz_red_x[k]);
        MPI_Comm_create(m_mpi_comm, m_mpi_group_xz_red_x[k], &m_mpi_comm_xz_red_x[k]);

        z_ranks[k] = h_cart_ranks.data[di(0,0,k)];
        }
    MPI_Group_incl(m_mpi_comm_group, di.getD(), z_ranks, &m_mpi_group_z);
    MPI_Comm_create(m_mpi_comm, m_mpi_group_z, &m_mpi_comm_z);

    delete[] x_ranks;
    delete[] y_ranks;
    delete[] z_ranks;
    }

LoadBalancer::~LoadBalancer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LoadBalancer" << endl;

    // free the communicators and groups that we have made
    
    if (m_mpi_comm_xy != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_xy);
    if (m_mpi_comm_xz != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_xz);

    for (unsigned int i=0; i < m_mpi_comm_xy_red_y.size(); ++i)
        {
        if (m_mpi_comm_xy_red_y[i] != MPI_COMM_NULL)
            MPI_Comm_free(&m_mpi_comm_xy_red_y[i]);
        if (m_mpi_group_xy_red_y[i] != MPI_GROUP_NULL)
            MPI_Group_free(&m_mpi_group_xy_red_y[i]);
        }
    for (unsigned int i=0; i < m_mpi_comm_xy_red_x.size(); ++i)
        {
        if (m_mpi_comm_xy_red_x[i] != MPI_COMM_NULL)
            MPI_Comm_free(&m_mpi_comm_xy_red_x[i]);
        if (m_mpi_group_xy_red_x[i] != MPI_GROUP_NULL)
            MPI_Group_free(&m_mpi_group_xy_red_x[i]);
        }
    for (unsigned int i=0; i < m_mpi_comm_xz_red_x.size(); ++i)
        {
        if (m_mpi_comm_xz_red_x[i] != MPI_COMM_NULL)
            MPI_Comm_free(&m_mpi_comm_xz_red_x[i]);
        if (m_mpi_group_xz_red_x[i] != MPI_GROUP_NULL)
            MPI_Group_free(&m_mpi_group_xz_red_x[i]);
        }

    if (m_mpi_comm_x != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_x);
    if (m_mpi_group_x != MPI_GROUP_NULL)
        MPI_Group_free(&m_mpi_group_x);

    if (m_mpi_comm_y != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_y);
    if (m_mpi_group_y != MPI_GROUP_NULL)
        MPI_Group_free(&m_mpi_group_y);

    if (m_mpi_comm_z != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_z);
    if (m_mpi_group_z != MPI_GROUP_NULL)
        MPI_Group_free(&m_mpi_group_z);
    }

/*!
 * \param timestep Current time step of the simulation
 *
 * Computes the load imbalance along each slice and adjusts the domain boundaries. This process is repeated iteratively
 * in each dimension taking into account the adjusted boundaries each time. An adjustment of 50% of the load imbalance
 * is attempted, subject to the following constraints:
 *  - No length may change by more than 5% for stability
 *  - No length may shrink smaller than the ghost width (or the system is over-decomposed)
 *  - No boundary may move past than halfway point of an adjacent cell (conservative way to stop a particle jumping 2 cells)
 *  - The total volume must be conserved after adjustment
 */
void LoadBalancer::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("balance");

    // we need a communicator, but don't want to check for it in release builds
    assert(m_comm);

    // no adjustment has been made yet, so set m_N_own to the number of particles on the rank
    m_N_own = m_pdata->getN();
    m_needs_recount = false;

    // figure out which rank is the reduction root for broadcasting
    const Index3D& di = m_decomposition->getDomainIndexer();
    unsigned int reduce_root(0);
        {
        ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);
        reduce_root = h_cart_ranks.data[di(0,0,0)];
        }

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();

    for (unsigned int cur_iter=0; cur_iter < m_maxiter && getMaxImbalance() > m_tolerance; ++cur_iter)
        {
        for (unsigned int dim=0; dim < m_sysdef->getNDimensions() && getMaxImbalance() > m_tolerance; ++dim)
            {
            Scalar L_i(0.0);
            vector<unsigned int> N_i;
            if (dim == 0)
                {
                if (!m_enable_x) continue; // skip this dimension if balancing is turned off

                L_i = L.x;
                N_i.resize(di.getW());
                }
            else if (dim == 1)
                {
                if (!m_enable_y) continue;

                L_i = L.y;
                N_i.resize(di.getH());
                }
            else
                {
                if (!m_enable_z) continue;

                L_i = L.z;
                N_i.resize(di.getD());
                }

            bool active = reduce(N_i, dim);
            vector<Scalar> cum_frac = m_decomposition->getCumulativeFractions(dim);
            if (active)
                {
                adjust(cum_frac, N_i, L_i);
                }
            bcast(m_needs_recount, reduce_root, m_mpi_comm);
            if (m_needs_recount)
                {
                m_decomposition->setCumulativeFractions(dim, cum_frac, reduce_root);
                m_pdata->setGlobalBox(box); // force a domain resizing to trigger
                computeParticleChange();
                }
            }

        // migrate particles to their final locations NOW because we have modified the domains
        m_comm->forceMigrate();
        m_comm->communicate(timestep);

        // after migration, no recounting is necessary because all ranks own their particles
        m_N_own = m_pdata->getN();
        m_needs_recount = false;

        Scalar max_imb = getMaxImbalance();
        if (m_exec_conf->getRank() == 0)
            {
            const BoxDim& root_box = m_pdata->getBox();
            Scalar3 root_hi = root_box.getHi();
            vector<Scalar> cum_frac_x = m_decomposition->getCumulativeFractions(0);
            vector<Scalar> cum_frac_y = m_decomposition->getCumulativeFractions(1);
            vector<Scalar> cum_frac_z = m_decomposition->getCumulativeFractions(2);
            cout << timestep << " " << max_imb << " " << cum_frac_x[1] << " " << cum_frac_y[1] << " " << cum_frac_z[1] << " " << root_hi.x << " " << root_hi.y << " " << root_hi.z << endl;
            }
        }

    if (m_prof) m_prof->pop();
    }

/*!
 * Computes the imbalance factor I = N / <N> for each rank, and computes the maximum among all ranks.
 */
Scalar LoadBalancer::getMaxImbalance()
    {
    if (m_needs_recount)
        {
        m_exec_conf->msg->error() << "comm: cannot compute imbalance factor while recounting is pending" << endl;
        throw runtime_error("Cannot compute imbalance factor while recounting is pending");
        }

    Scalar cur_imb = Scalar(m_N_own) / (Scalar(m_pdata->getNGlobal()) / Scalar(m_exec_conf->getNRanks()));
    Scalar max_imb(0.0);
    MPI_Allreduce(&cur_imb, &max_imb, 1, MPI_HOOMD_SCALAR, MPI_MAX, m_mpi_comm);
    
    return max_imb;
    }

/*!
 * \param N_i Vector holding the total number of particles in each slice
 * \param dim The dimension of the slices (x=0, y=1, z=2)
 * \returns true if the current rank holds the active \a N_i
 *
 * \post \a N_i holds the number of particles in each slice along \a dim
 *
 * \note reduce() relies on collective MPI calls, and so all ranks must call it. However, for efficiency the data will
 *       be active only on Cartesian rank 0 along \a dim, as indicated by the return value. As a result, only rank 0
 *       needs to actually allocate memory for \a N_i. The return value should be used to check the active buffer in
 *       case there is a situation where rank 0 is not Cartesian rank 0.
 */
bool LoadBalancer::reduce(std::vector<unsigned int>& N_i, unsigned int dim)
    {
    // do nothing if there is only one rank
    if (N_i.size() == 1) return false;

    const uint3 my_grid_pos = m_decomposition->getGridPos();

    computeParticleChange();

    unsigned int sum_N(0), sum_sum_N(0);

    if (dim == 0) // to x
        {
        MPI_Reduce(&m_N_own, &sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xy);

        if (my_grid_pos.z == 0)
            {
            MPI_Reduce(&sum_N, &sum_sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xy_red_y[my_grid_pos.x]);

            if (my_grid_pos.y == 0)
                {
                MPI_Gather(&sum_sum_N, 1, MPI_INT, &N_i.front(), 1, MPI_INT, 0, m_mpi_comm_x);
                if (my_grid_pos.x == 0)
                    {
                    return true;
                    }
                }
            }
        }
    else if (dim == 1) // to y
        {
        MPI_Reduce(&m_N_own, &sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xy);

        if (my_grid_pos.z == 0)
            {
            MPI_Reduce(&sum_N, &sum_sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xy_red_x[my_grid_pos.y]);

            if (my_grid_pos.x == 0)
                {
                MPI_Gather(&sum_sum_N, 1, MPI_INT, &N_i.front(), 1, MPI_INT, 0, m_mpi_comm_y);
                if (my_grid_pos.y == 0)
                    {
                    return true;
                    }
                }
            }
        }
    else if (dim == 2) // to z
        {
        MPI_Reduce(&m_N_own, &sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xz);

        if (my_grid_pos.y == 0)
            {
            MPI_Reduce(&sum_N, &sum_sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_xz_red_x[my_grid_pos.z]);

            if (my_grid_pos.x == 0)
                {
                MPI_Gather(&sum_sum_N, 1, MPI_INT, &N_i.front(), 1, MPI_INT, 0, m_mpi_comm_z);
                if (my_grid_pos.z == 0)
                    {
                    return true;
                    }
                }
            }
        }
    else
        {
        m_exec_conf->msg->error() << "comm.balance: unknown dimension for particle reduction" << endl;
        throw runtime_error("Unknown dimension for particle reduction");
        }
    return false;
    }

/*!
 * \returns true if an adjustment occurred
 */
bool LoadBalancer::adjust(vector<Scalar>& cum_frac_i,
                          const vector<unsigned int>& N_i,
                          Scalar L_i)
    {
    if (N_i.size() == 1)
        return false;

    // target particles per rank is uniform distribution
    const Scalar target = Scalar(m_pdata->getNGlobal()) / Scalar(N_i.size());

    // make the minimum domain slightly bigger so that the optimization won't fail
    const Scalar min_domain_size = Scalar(2.0)*m_comm->getGhostLayerWidth();

    // imbalance factors for each rank
    vector<Scalar> new_widths(N_i.size());
    
    // flag which variables are free choices to setup the optimization problem
    vector<int> map_rank_to_var(N_i.size(), -1); // map of where the rank is in free variables (< 0 indicates fixed)
    vector<unsigned int> free_vars;
    free_vars.reserve(N_i.size());

    Scalar max_imb_factor(1.0);
    for (unsigned int i=0; i < N_i.size(); ++i)
        {
        const Scalar imb_factor = Scalar(N_i[i]) / target;
        Scalar scale_factor = (N_i[i] > 0) ? Scalar(0.5) / imb_factor : (Scalar(1.0) + m_max_scale); // as in gromacs, use half the imbalance factor to scale

        // limit rescaling to 5% either direction
        if (scale_factor > (Scalar(1.0) + m_max_scale))
            {
            scale_factor = (Scalar(1.0) + m_max_scale);
            }
        else if(scale_factor < (Scalar(1.0) - m_max_scale))
            {
            scale_factor = (Scalar(1.0) - m_max_scale);
            }

        // compute the new domain width (can't be smaller than the threshold, if it is, this is not a free variable)
        Scalar new_width = scale_factor * (cum_frac_i[i+1] - cum_frac_i[i]) * L_i;
        // enforce a hard limit on the minimum domain size
        // this is a good assumption for optimization because chances are that reducing this domain to the minimum
        // will minimize the total objective function. if it needs to get bigger, it can be expanded on next iteration
        if (new_width < min_domain_size)
            {
            new_width = min_domain_size;
            }
        else
            {
            map_rank_to_var[i] = free_vars.size(); // current size is the index we will push into
            free_vars.push_back(i); // if not at the minimum, then this domain is a free variable
            }
        new_widths[i] = new_width;

        if (imb_factor > max_imb_factor)
            max_imb_factor = imb_factor;
        }

    // balancing can be expensive, so don't do anything if we're already good enough in this direction
    if (max_imb_factor <= m_tolerance)
        {
        return false;
        }
    
    // if there's zero or one free variables, then the system is totally constrained and solved algebraically
    if (free_vars.size() == 0)
        {
        // either the system is overconstrained (an error), or the system is already perfectly balanced, so do nothing
        return false;
        }
    else if (free_vars.size() == 1)
        {
        // fix the one free variable width algebraically
        new_widths[free_vars.front()] = L_i - min_domain_size * Scalar(N_i.size() - 1);
        
        // loop through and rescale widths to fractions
        for (unsigned int cur_rank = 0; cur_rank < N_i.size(); ++cur_rank)
            {
            new_widths[cur_rank] /= L_i; // to fractional width
            }
        std::partial_sum(new_widths.begin(), new_widths.end()-1, cum_frac_i.begin() + 1);
        m_needs_recount = true;
        return true;
        }


    /* Here come's the harder case -- there are at least two free variables */

    // setup the A matrix just based on the number of free variables (we can map these back later)
    // if there are N free variables, then the length constraint of the box gives N-1 true free variables for optimization
    unsigned int rows = free_vars.size();
    unsigned int cols = free_vars.size() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
    A(0,0) = 1.0;
    A(rows-1, cols-1) = -1.0;
    for (unsigned int i=1; i < rows-1; ++i)
        {
        A(i,i-1) = -1.0;
        A(i,i) = 1.0;
        }

    // the tricky part is figuring out what the target values should be
    // we need to add amounts that correspond to the number of fixed lengths
    
    // the number of fixed variables at the front is equal to the index and at the tail is equal to the number of
    // ranks minus the rank index of the last free choice
    // (i.e., if the last one is free, then there is 1, if the last one is fixed, there are 2, etc.
    // if the first one is free, then there are 0 dead, if the first is fixed, then there is 1, ...)
    unsigned int n_dead_front = free_vars.front();
    unsigned int n_dead_end = N_i.size() - free_vars.back();
    Eigen::VectorXd b(rows);
    for (unsigned int i=0; i < rows; ++i)
        {
        const unsigned int cur_rank = free_vars[i];
        b(i) = new_widths[cur_rank];

        // add the appropriate correction for the number of constrained zones between free variables
        if (i == 0)
            {
            b(0) += min_domain_size * Scalar(n_dead_front);
            }
        else if (i == rows - 1)
            {
            b(rows-1) += min_domain_size * Scalar(n_dead_end);
            }
        else
            {
            b(i) += min_domain_size * Scalar(free_vars[i+1] - cur_rank - 1); // -1 so that adjacent var needs no correction
            }
        }
    b(rows-1) -= L_i; // the last equation applies the total length constraint
    free_vars.pop_back(); // this variable is not really free anymore, so pop it off

    // position constraints limit movement due to over-decomposition
    Eigen::VectorXd l(cols), u(cols);
    for (unsigned int j=0; j < cols; ++j)
        {
        const unsigned int cur_rank = free_vars[j];
        l(cur_rank) = Scalar(0.5) * (cum_frac_i[cur_rank] + cum_frac_i[cur_rank+1]) * L_i;
        u(cur_rank) = Scalar(0.5) * (cum_frac_i[cur_rank+1] + cum_frac_i[cur_rank+2]) * L_i;
        }
    try
        {
        BVLSSolver solver(A, b, l, u);
        solver.setMaxIterations(3*free_vars.size());
        solver.solve();
        if (solver.converged())
            {
            Eigen::VectorXd x = solver.getSolution();
            
            // now we need to map the solution back and fill in the holes
            vector<Scalar> sorted_f(N_i.size()-1);
            Scalar cur_div_pos(0.0);
            for (unsigned int cur_div=0; cur_div < N_i.size() - 1; ++cur_div)
                {
                if (map_rank_to_var[cur_div] >= 0)
                    {
                    if (x(map_rank_to_var[cur_div]) < min_domain_size)
                        {
                        m_exec_conf->msg->warning() << "comm.balance: no convergence, domains too small" << endl;
                        return false;
                        }
                    cur_div_pos = x(map_rank_to_var[cur_div]);
                    }
                else
                    {
                    cur_div_pos += new_widths[cur_div];
                    }
                sorted_f[cur_div] = cur_div_pos / L_i;
                if (cur_div > 0 && sorted_f[cur_div] < sorted_f[cur_div-1])
                    {
                    m_exec_conf->msg->warning() << "comm.balance: domains attempting to flip" << endl;
                    return false;
                    }
                }
            for (unsigned int cur_div=0; cur_div < sorted_f.size(); ++cur_div)
                {
                cum_frac_i[cur_div+1] = sorted_f[cur_div];
                }

            m_needs_recount = true;
            return true;
            }
        else
            {
            m_exec_conf->msg->warning() << "comm.balance: converged load balance not found" << endl;
            return false;
            }
        }
    catch (const runtime_error& e)
        {
        m_exec_conf->msg->error() << "comm.balance: an error occurred seeking optimal load balance" << endl;
        throw e;
        }

    return false;
    }

void LoadBalancer::countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    const Index3D& di = m_decomposition->getDomainIndexer();
    const uint3 rank_pos = m_decomposition->getGridPos();

    for (unsigned int cur_p=0; cur_p < m_pdata->getN(); ++cur_p)
        {
        const Scalar4 cur_postype = h_pos.data[cur_p];
        const Scalar3 cur_pos = make_scalar3(cur_postype.x, cur_postype.y, cur_postype.z);
        const Scalar3 f = box.makeFraction(cur_pos);

        int3 grid_pos = make_int3(rank_pos.x, rank_pos.y, rank_pos.z);

        bool moved(false);
        if (f.x >= Scalar(1.0))
            {
            ++grid_pos.x;
            moved = true;
            }
        if (f.x < Scalar(0.0))
            {
            --grid_pos.x;
            moved = true;
            }

        if (f.y >= Scalar(1.0))
            {
            ++grid_pos.y;
            moved = true;
            }
        if (f.y < Scalar(0.0))
            {
            --grid_pos.y;
            moved = true;
            }

        if (f.z >= Scalar(1.0))
            {
            ++grid_pos.z;
            moved = true;
            }
        if (f.z < Scalar(0.0))
            {
            --grid_pos.z;
            moved = true;
            }

        if (moved)
            {
            if (grid_pos.x == (int)di.getW())
                grid_pos.x = 0;
            else if (grid_pos.x < 0)
                grid_pos.x += di.getW();

            if (grid_pos.y == (int)di.getH())
                grid_pos.y = 0;
            else if (grid_pos.y < 0)
                grid_pos.y += di.getH();

            if (grid_pos.z == (int)di.getD())
                grid_pos.z = 0;
            else if (grid_pos.z < 0)
                grid_pos.z += di.getD();

            unsigned int cur_rank = h_cart_ranks.data[di(grid_pos.x,grid_pos.y,grid_pos.z)];
            cnts[cur_rank]++;
            }
        }
    }

void LoadBalancer::computeParticleChange()
    {
    // don't do anything if nobody has signaled a change
    if (!m_needs_recount) return;

    // count the particles that are off the rank
    ArrayHandle<unsigned int> h_unique_neigh(m_comm->getUniqueNeighbors(), access_location::host, access_mode::read);

    // fill the map initially to zeros (not necessary since should be auto-initialized to zero, but just playing it safe)
    std::map<unsigned int, unsigned int> cnts;
    for (unsigned int i=0; i < m_comm->getNUniqueNeighbors(); ++i)
        {
        cnts[h_unique_neigh.data[i]] = 0;
        }
    countParticlesOffRank(cnts);

    MPI_Request req[2*m_comm->getNUniqueNeighbors()];
    MPI_Status stat[2*m_comm->getNUniqueNeighbors()];
    unsigned int nreq = 0;

    unsigned int n_send_ptls[m_comm->getNUniqueNeighbors()];
    unsigned int n_recv_ptls[m_comm->getNUniqueNeighbors()];
    for (unsigned int cur_neigh=0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        unsigned int neigh_rank = h_unique_neigh.data[cur_neigh];
        n_send_ptls[cur_neigh] = cnts[neigh_rank];

        MPI_Isend(&n_send_ptls[cur_neigh], 1, MPI_UNSIGNED, neigh_rank, 0, m_mpi_comm, & req[nreq++]);
        MPI_Irecv(&n_recv_ptls[cur_neigh], 1, MPI_UNSIGNED, neigh_rank, 0, m_mpi_comm, & req[nreq++]);
        }
    MPI_Waitall(nreq, req, stat);

    // reduce the particles sent to me
    int N_own = m_pdata->getN();
    for (unsigned int cur_neigh = 0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        N_own += n_recv_ptls[cur_neigh];
        N_own -= n_send_ptls[cur_neigh];
        }
    m_N_own = N_own;

    // we are done until someone needs us to recount again
    m_needs_recount = false;
    }

void export_LoadBalancer()
    {
    class_<LoadBalancer, boost::shared_ptr<LoadBalancer>, bases<Updater>, boost::noncopyable>
    ("LoadBalancer", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<BalancedDomainDecomposition> >())
    .def("enableDimension", &LoadBalancer::enableDimension)
    .def("getTolerance", &LoadBalancer::getTolerance)
    .def("setTolerance", &LoadBalancer::setTolerance)
    .def("getMaxIterations", &LoadBalancer::getMaxIterations)
    .def("setMaxIterations", &LoadBalancer::setMaxIterations)
    ;
    }
#endif // ENABLE_MPI
