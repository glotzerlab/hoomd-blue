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
          m_max_imbalance(Scalar(1.0)), m_recompute_max_imbalance(true), m_needs_migrate(false),
          m_needs_recount(false), m_tolerance(Scalar(1.05)), m_maxiter(1),
          m_enable_x(true), m_enable_y(true), m_enable_z(true), m_max_scale(Scalar(0.05)), m_N_own(m_pdata->getN()),
          m_max_max_imbalance(1.0), m_total_max_imbalance(0.0), m_n_calls(0), m_n_iterations(0), m_n_rebalances(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing LoadBalancer" << endl;
    }

LoadBalancer::~LoadBalancer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LoadBalancer" << endl;
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
    resetNOwn(m_pdata->getN());

    // figure out which rank is the reduction root for broadcasting
    const Index3D& di = m_decomposition->getDomainIndexer();
    unsigned int reduce_root(0);
        {
        ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);
        reduce_root = h_cart_ranks.data[di(0,0,0)];
        }

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();

    // compute the current imbalance always for the average
    m_total_max_imbalance += getMaxImbalance();
    ++m_n_calls;

    for (unsigned int cur_iter=0; cur_iter < m_maxiter && getMaxImbalance() > m_tolerance; ++cur_iter)
        {
        // increment the number of attempted balances
        ++m_n_iterations;

        for (unsigned int dim=0; dim < m_sysdef->getNDimensions() && getMaxImbalance() > m_tolerance; ++dim)
            {
            Scalar L_i(0.0);
            if (dim == 0)
                {
                if (!m_enable_x) continue; // skip this dimension if balancing is turned off
                L_i = L.x;
                }
            else if (dim == 1)
                {
                if (!m_enable_y) continue;
                L_i = L.y;
                }
            else
                {
                if (!m_enable_z) continue;
                L_i = L.z;
                }

            vector<unsigned int> N_i;
            bool adjusted = false;
            bool active = reduce(N_i, dim, reduce_root);
            vector<Scalar> cum_frac = m_decomposition->getCumulativeFractions(dim);
            if (active)
                {
                adjusted = adjust(cum_frac, N_i, L_i);
                }
            bcast(adjusted, reduce_root, m_mpi_comm);

            if (adjusted)
                {
                m_decomposition->setCumulativeFractions(dim, cum_frac, reduce_root);
                m_pdata->setGlobalBox(box); // force a domain resizing to trigger
                signalResize();
                }
            }

        // force a particle migration
        if (m_needs_migrate)
            {
            m_comm->migrateParticles();
            resetNOwn(m_pdata->getN());
            m_needs_migrate = false;

            // increment the number of rebalances actually performed
            ++m_n_rebalances;
            }
        }

    if (m_prof) m_prof->pop();
    }

/*!
 * Computes the imbalance factor I = N / <N> for each rank, and computes the maximum among all ranks.
 */
Scalar LoadBalancer::getMaxImbalance()
    {
    if (m_recompute_max_imbalance)
        {
        Scalar cur_imb = Scalar(getNOwn()) / (Scalar(m_pdata->getNGlobal()) / Scalar(m_exec_conf->getNRanks()));
        Scalar max_imb(0.0);
        MPI_Allreduce(&cur_imb, &max_imb, 1, MPI_HOOMD_SCALAR, MPI_MAX, m_mpi_comm);

        m_max_imbalance = max_imb;
        m_recompute_max_imbalance = false;

        // save as a statistic if new max imbalance
        if (m_max_imbalance > m_max_max_imbalance)
            m_max_max_imbalance = m_max_imbalance;
        }
    return m_max_imbalance;
    }

/*!
 * \param N_i Vector holding the total number of particles in each slice (will be allocated on call)
 * \param dim The dimension of the slices (x=0, y=1, z=2)
 * \param reduce_root The rank to perform the reduction on
 * \returns true if the current rank holds the active \a N_i
 *
 * \post \a N_i holds the number of particles in each slice along \a dim
 *
 * \note reduce() relies on collective MPI calls, and so all ranks must call it. However, for efficiency the data will
 *       be active only on Cartesian rank 0 along \a dim, as indicated by the return value. As a result, only rank 0
 *       will actually allocate memory for \a N_i. The return value should be used to check the active buffer in
 *       case there is a situation where rank 0 is not Cartesian rank 0.
 */
bool LoadBalancer::reduce(std::vector<unsigned int>& N_i, unsigned int dim, unsigned int reduce_root)
    {
    // do nothing if there is only one rank
    if (N_i.size() == 1) return false;

    const Index3D& di = m_decomposition->getDomainIndexer();
    std::vector<unsigned int> N_per_rank(di.getNumElements());

    unsigned int N_own = getNOwn();

    if (m_prof) m_prof->push("reduce");
    MPI_Gather(&N_own, 1, MPI_UNSIGNED, &N_per_rank[0], 1, MPI_UNSIGNED, reduce_root, m_mpi_comm);

    // only the root rank performs the reduction
    if (m_exec_conf->getRank() != reduce_root)
        return false;

    // rearrange the data from ranks to cartesian order in case it is jumbled around
    ArrayHandle<unsigned int> h_cart_ranks_inv(m_decomposition->getInverseCartRanks(), access_location::host, access_mode::read);
    std::vector<unsigned int> N_per_cart_rank(di.getNumElements());
    for (unsigned int cur_rank=0; cur_rank < di.getNumElements(); ++cur_rank)
        {
        N_per_cart_rank[h_cart_ranks_inv.data[cur_rank]] = N_per_rank[cur_rank];
        }

    if (dim == 0) // to x
        {
        N_i.clear(); N_i.resize(di.getW());
        for (unsigned int i=0; i < di.getW(); ++i)
            {
            N_i[i] = 0;
            for (unsigned int k=0; k < di.getD(); ++k)
                {
                for (unsigned int j=0; j < di.getH(); ++j)
                    {
                    N_i[i] += N_per_cart_rank[di(i,j,k)];
                    }
                }
            }
        }
    else if (dim == 1) // to y
        {
        N_i.clear(); N_i.resize(di.getH());
        for (unsigned int j=0; j < di.getH(); ++j)
            {
            N_i[j] = 0;
            for (unsigned int k=0; k < di.getD(); ++k)
                {
                for (unsigned int i=0; i < di.getW(); ++i)
                    {
                    N_i[j] += N_per_cart_rank[di(i,j,k)];
                    }
                }
            }
        }
    else if (dim == 2) // to z
        {
        N_i.clear(); N_i.resize(di.getD());
        for (unsigned int k=0; k < di.getD(); ++k)
            {
            N_i[k] = 0;
            for (unsigned int j=0; j < di.getH(); ++j)
                {
                for (unsigned int i=0; i < di.getW(); ++i)
                    {
                    N_i[k] += N_per_cart_rank[di(i, j, k)];
                    }
                }
            }
        }
    else
        {
        m_exec_conf->msg->error() << "comm.balance: unknown dimension for particle reduction" << endl;
        throw runtime_error("Unknown dimension for particle reduction");
        }

    if (m_prof) m_prof->pop();

    return true;
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

    if (m_prof) m_prof->push("adjust");

    // target particles per rank is uniform distribution
    const Scalar target = Scalar(m_pdata->getNGlobal()) / Scalar(N_i.size());

    // make the minimum domain slightly bigger so that the optimization won't fail
    const Scalar min_domain_size = Scalar(2.0)*m_comm->getGhostLayerMaxWidth();

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
        if (m_prof) m_prof->pop();
        return false;
        }
    
    // if there's zero or one free variables, then the system is totally constrained and solved algebraically
    if (free_vars.size() == 0)
        {
        // either the system is overconstrained (an error), or the system is already perfectly balanced, so do nothing
        if (m_prof) m_prof->pop();
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
        if (m_prof) m_prof->pop();
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
                        if (m_prof) m_prof->pop();
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
                    if (m_prof) m_prof->pop();
                    return false;
                    }
                }
            for (unsigned int cur_div=0; cur_div < sorted_f.size(); ++cur_div)
                {
                cum_frac_i[cur_div+1] = sorted_f[cur_div];
                }
            if (m_prof) m_prof->pop();
            return true;
            }
        else
            {
            m_exec_conf->msg->warning() << "comm.balance: converged load balance not found" << endl;
            if (m_prof) m_prof->pop();
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

void LoadBalancer::computeOwnedParticles()
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
    if (m_prof) m_prof->push("count");
    countParticlesOffRank(cnts);
    if (m_prof) m_prof->pop();

    if (m_prof) m_prof->push("MPI send/recv");
    MPI_Request req[2*m_comm->getNUniqueNeighbors()];
    MPI_Status stat[2*m_comm->getNUniqueNeighbors()];
    unsigned int nreq = 0;

    unsigned int n_send_ptls[m_comm->getNUniqueNeighbors()];
    unsigned int n_recv_ptls[m_comm->getNUniqueNeighbors()];
    unsigned int send_bytes(0), recv_bytes(0);
    for (unsigned int cur_neigh=0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        unsigned int neigh_rank = h_unique_neigh.data[cur_neigh];
        n_send_ptls[cur_neigh] = cnts[neigh_rank];

        MPI_Isend(&n_send_ptls[cur_neigh], 1, MPI_UNSIGNED, neigh_rank, 0, m_mpi_comm, & req[nreq++]);
        MPI_Irecv(&n_recv_ptls[cur_neigh], 1, MPI_UNSIGNED, neigh_rank, 0, m_mpi_comm, & req[nreq++]);
        send_bytes += sizeof(unsigned int);
        recv_bytes += sizeof(unsigned int);
        }
    MPI_Waitall(nreq, req, stat);

    if (m_prof) m_prof->pop(0, send_bytes + recv_bytes);

    // reduce the particles sent to me
    int N_own = m_pdata->getN();
    for (unsigned int cur_neigh = 0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        N_own += n_recv_ptls[cur_neigh];
        N_own -= n_send_ptls[cur_neigh];
        }

    // set the count
    resetNOwn(N_own);
    }

/*!
 * Print statistics on the maximum and average load imbalance, and the number of times
 * load balancing was performed.
 */
void LoadBalancer::printStats()
    {
    if (m_exec_conf->msg->getNoticeLevel() < 1)
        return;

    double avg_imb = m_total_max_imbalance / ((double)m_n_calls);
    m_exec_conf->msg->notice(1) << "-- Load imbalance stats:" << endl;
    m_exec_conf->msg->notice(1) << "max imbalance: " << m_max_max_imbalance << " / avg. imbalance: " << avg_imb << endl;
    m_exec_conf->msg->notice(1) << "iterations: " << m_n_iterations << " / rebalances: " << m_n_rebalances << endl;
    }

void LoadBalancer::resetStats()
    {
    m_n_calls = m_n_iterations = m_n_rebalances = 0;
    m_total_max_imbalance = 0.0;
    m_max_max_imbalance = Scalar(1.0);
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
