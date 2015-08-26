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

using namespace std;

/*!
 * \param sysdef System definition
 */
LoadBalancer::LoadBalancer(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<BalancedDomainDecomposition> decomposition)
        : Updater(sysdef), m_decomposition(decomposition), m_mpi_comm(m_exec_conf->getMPICommunicator()),
          m_N_own(m_pdata->getN()), m_needs_recount(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing LoadBalancer" << endl;

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

    // copy the domain decomposition into GPUArrays (this could be avoided by turning the DD into GPUArrays
    // and just operating on this data, which is what we really want do do)
    const Index3D& di = m_decomposition->getDomainIndexer();
    const uint3 my_grid_pos = m_decomposition->getGridPos();

    // vectors to hold the reduced particle numbers along each dimension
    vector<unsigned int> N_x(di.getW()), N_y(di.getH()), N_z(di.getD());

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();

    bool active = reduce(N_x, 0);
    if (active)
        {
        vector<Scalar> frac_x = m_decomposition->getFractions(0);
        vector<Scalar> cum_frac_x = m_decomposition->getCumulativeFractions(0);
        if (adjust(cum_frac_x, frac_x, N_x, L.x))
            {
            cout << cum_frac_x[0] << " " << cum_frac_x[1] << " " << cum_frac_x[2] << endl;
//             m_decomposition->setCumulativeFractions(0, cum_frac_x);
//             m_pdata->updateLocalBox();
            }
        }

    active = reduce(N_y, 1);
    if (active)
        {
        vector<Scalar> frac_y = m_decomposition->getFractions(1);
        vector<Scalar> cum_frac_y = m_decomposition->getCumulativeFractions(1);
        if (adjust(cum_frac_y, frac_y, N_y, L.y))
            {
//             m_decomposition->setCumulativeFractions(1, cum_frac_y);
//             m_pdata->updateLocalBox();
            }
        }

    active = reduce(N_z, 2);
    if (active)
        {
        vector<Scalar> frac_z = m_decomposition->getFractions(2);
        vector<Scalar> cum_frac_z = m_decomposition->getCumulativeFractions(2);
        if (adjust(cum_frac_z, frac_z, N_z, L.z))
            {
//             m_decomposition->setCumulativeFractions(2, cum_frac_z);
//             m_pdata->updateLocalBox();
            }
        }

    // migrate particles to their final locations
    m_comm->migrateParticles();

    if (m_prof) m_prof->pop();
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
                          const vector<Scalar>& frac_i,
                          const vector<unsigned int>& N_i,
                          Scalar L_i)
    {
    if (N_i.size() == 1)
        return false;

    // target particles per rank is uniform distribution
    const Scalar target = Scalar(m_pdata->getNGlobal()) / Scalar(N_i.size());

    // imbalance factors for each rank
    vector<Scalar> scale_factor(N_i.size());
    Scalar max_imb(-1.0);
    for (unsigned int i=0; i < N_i.size(); ++i)
        {
        const Scalar imb_factor = Scalar(N_i[i]) / target;
        if (N_i[i] > 0)
            {
            scale_factor[i] = Scalar(0.5) / imb_factor; // as in gromacs, use half the imbalance factor to scale
            if (scale_factor[i] > Scalar(1.05))
                {
                scale_factor[i] = Scalar(1.05);
                }
            else if(scale_factor[i] < Scalar(0.95))
                {
                scale_factor[i] = Scalar(0.95);
                }
            
            }
        else
            {
            scale_factor[i] = Scalar(1.05); // apply rescaling limit to empty ranks otherwise they want to expand too much
            }
        
        if (imb_factor > max_imb)
            max_imb = imb_factor;
        }
    
    if (max_imb > Scalar(1.05))
        {
        // make the minimum domain slightly bigger so that the optimization won't fail
        const Scalar min_domain_size = Scalar(2.0)*m_comm->getGhostLayerWidth();
        const Scalar min_domain_target = Scalar(1.05) * min_domain_size;

        // define the optimization problem to solve to satisfy the constraints
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N_i.size(), N_i.size()-1);
        A(0,0) = 1.0;
        for (unsigned int i=1; i < N_i.size()-1; ++i)
            {
            A(i,i-1) = -1.0;
            A(i,i) = 1.0;
            }
        A(N_i.size()-1, N_i.size()-2) = -1.0;

        Eigen::VectorXd b(N_i.size());
        for (unsigned int cur_rank=0; cur_rank < N_i.size(); ++cur_rank)
            {
            Scalar new_width = scale_factor[cur_rank] * frac_i[cur_rank] * L_i;
            // enforce a soft (but really quite stiff) limit on the minimum domain size
            if (new_width < min_domain_target)
                {
                new_width = min_domain_target;
                // add a very steep penalty for violating this constraint
                if (cur_rank > 1)
                    A(cur_rank,cur_rank-1) *= Scalar(100.0);

                if (cur_rank < N_i.size() - 1)
                    A(cur_rank,cur_rank) *= Scalar(100.0);
                }
            b(cur_rank) = new_width;
            }
        b(N_i.size() - 1) -= L_i; // the last equation applies the length constraint
        
        Eigen::VectorXd l(N_i.size() - 1), u(N_i.size() - 1);
        for (unsigned int cur_rank=0; cur_rank < N_i.size()-1; ++cur_rank)
            {
            // multiply by a small scale factor so that we don't have <= problems
            l(cur_rank) = Scalar(1.001) * Scalar(0.5) * (cum_frac_i[cur_rank] + cum_frac_i[cur_rank+1]) * L_i;
            u(cur_rank) = Scalar(0.999) * Scalar(0.5) * (cum_frac_i[cur_rank+1] + cum_frac_i[cur_rank+2]) * L_i;
            }
        
        try
            {
            BVLSSolver solver(A, b, l, u);
            solver.solve();
            if (solver.converged())
                {
                Eigen::VectorXd x = solver.getSolution();
                // validate the converged solution by checking nobody is smaller than the minimum domain width
                // if somebody is, give up balancing this dimension
                vector<Scalar> sorted_f(x.size());
                bool need_sort = false;
                for (unsigned int cur_div=0; cur_div < x.size(); ++cur_div)
                    {
                    if (x(cur_div) < min_domain_size)
                        {
                        m_exec_conf->msg->warning() << "comm.balance: no convergence, domains too small" << endl;
                        return false;
                        }
                    sorted_f[cur_div] = x(cur_div) / L_i;
                    if (cur_div > 0 && sorted_f[cur_div] > sorted_f[cur_div-1])
                        {
                        need_sort = true;
                        }
                    }
                // sanity check: everybody needs to be in ascending order
                if (need_sort)
                    {
                    sort(sorted_f.begin(), sorted_f.end());
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
        }

    return false;
    }

void LoadBalancer::computeParticleChange()
    {
    // don't do anything if nobody has signaled a change
    if (!m_needs_recount) return;

    // compute the changes in each direction

    // we are done until someone needs us to recount again
    m_needs_recount = false;
    }

void export_LoadBalancer()
    {
    class_<LoadBalancer, boost::shared_ptr<LoadBalancer>, bases<Updater>, boost::noncopyable>
    ("LoadBalancer", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<BalancedDomainDecomposition> >())
    ;
    }
#endif // ENABLE_MPI
