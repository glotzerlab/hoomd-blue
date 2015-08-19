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

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

/*!
 * \param sysdef System definition
 */
LoadBalancer::LoadBalancer(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<BalancedDomainDecomposition> decomposition)
        : Updater(sysdef), m_decomposition(decomposition), m_mpi_comm(m_exec_conf->getMPICommunicator()),
          m_N_own(m_pdata->getN()), m_adjusted(false)
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
 *  - No length may change by more than half of the length of an adjacent cell (or particles could cross multiple cells)
 *  - No length may shrink smaller than the ghost width (or the system is over-decomposed)
 *  - The total volume must be conserved after adjustment
 */
void LoadBalancer::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("balance");

    // we need a communicator, but don't want to check for it in release builds
    assert(m_comm);

    // no adjustment has been made yet
    m_adjusted = false;

    // clear out the ghosts, they will be invalidated anyway
    m_comm->removeGhostParticles();

    // copy the domain decomposition into GPUArrays (this could be avoided by turning the DD into GPUArrays
    // and just operating on this data, which is what we really want do do)
    const Index3D& di = m_decomposition->getDomainIndexer();
    const uint3 my_grid_pos = m_decomposition->getGridPos();

    // vectors to hold the reduced particle numbers along each dimension
    vector<unsigned int> N_x(di.getW()), N_y(di.getH()), N_z(di.getD());

    vector<Scalar> cum_frac_x = m_decomposition->getCumulativeFractions(0);
    bool active = reduce(N_x, 0);
    if (active)
        {
        adjust(cum_frac_x, N_x);
        }
    // set the new domain decomposition along x
    // rescale the boxes (this is a trick to force the domain decomposition to act on the ParticleData)
    // should replace with a new function m_pdata->updateLocalBox() since this is hackish
    m_pdata->getGlobalBox(m_pdata->getGlobalBox());

    vector<Scalar> cum_frac_y = m_decomposition->getCumulativeFractions(1);
    active = reduce(N_y, 1);
    if (active)
        {
        adjust(cum_frac_y, N_y);
        }
    // broadcast the y adjustment
    // compute the change in particles per rank

    vector<Scalar> cum_frac_z = m_decomposition->getCumulativeFractions(2);
    active = reduce(N_z, 2);
    if (active)
        {
        adjust(cum_frac_z, N_z);
        }

    // notify a box size change
    // migrate particles to their final locations
    m_comm->migrateParticles();
    
    // compute the load imbalance and repeat if necessary

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
bool LoadBalancer::adjust(vector<Scalar>& cum_frac_i, const vector<unsigned int>& N_i)
    {
    // imbalance factors for each rank
    vector<Scalar> imb_factor(N_i.size());
    
    m_adjusted = true;
    return true;
    }

void LoadBalancer::computeParticleChange()
    {
    if (!m_adjusted)
        {
        m_N_own = m_pdata->getN();
        return;
        }
    }

void export_LoadBalancer()
    {
    class_<LoadBalancer, boost::shared_ptr<LoadBalancer>, bases<Updater>, boost::noncopyable>
    ("LoadBalancer", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<BalancedDomainDecomposition> >())
    ;
    }
#endif // ENABLE_MPI
