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
        : Updater(sysdef), m_decomposition(decomposition), m_mpi_comm(m_exec_conf->getMPICommunicator())
    {
    m_exec_conf->msg->notice(5) << "Constructing LoadBalancer" << endl;
    assert(m_pdata);

    // setup the MPI_Comm that this rank participates in for reduction
    // reduction happens down z first, so color according to the rank in the xy plane and key using the z position
    // all ranks participate in this reduction
    const Index3D& di = m_decomposition->getDomainIndexer();
    uint3 my_grid_pos = m_decomposition->getGridPos();
    MPI_Comm_split(m_mpi_comm, di(my_grid_pos.x, my_grid_pos.y, 0), my_grid_pos.z, &m_mpi_comm_z);
    
    // now, the reduction down y is a little trickier because only the ranks in z = 0 plane should participate in this step
    // we handle this by splitting and coloring each communicator, but then we will only do the reduction for the
    // ones we care about

    // get the world group and then select only those ranks that are in the xy plane
    MPI_Comm_group(m_mpi_comm, &m_mpi_comm_group);
    // we need one group and communicator for every slice in x
    m_mpi_group_y.resize(di.getW());
    m_mpi_comm_y.resize(di.getW());
    m_roots_y.resize(di.getW());

    // create an inclusion list along y for every x slice so that we know which ranks we need to include
    int *x_ranks = new int[di.getW()];
    int *y_ranks = new int[di.getH()];
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);
    for (unsigned int i=0; i < di.getW(); ++i)
        {
        // we are stuffing the ranks into the communicator in the Cartesian order, so the root should be 0
        for (unsigned int j=0; j < di.getH(); ++j)
            {
            y_ranks[j] = h_cart_ranks.data[di(i,j,0)];
            }
        MPI_Group_incl(m_mpi_comm_group, di.getH(), y_ranks, &m_mpi_group_y[i]);
        MPI_Comm_create(m_mpi_comm, m_mpi_group_y[i], &m_mpi_comm_y[i]);
        
        // save the ranks of the x roots for making a new communicator
        x_ranks[i] = h_cart_ranks.data[di(i,0,0)];
        }
    delete[] y_ranks;
    
    // create the x communicator
    MPI_Group_incl(m_mpi_comm_group, di.getW(), x_ranks, &m_mpi_group_x);
    MPI_Comm_create(m_mpi_comm, m_mpi_group_x, &m_mpi_comm_x);
    delete[] x_ranks;
    }

LoadBalancer::~LoadBalancer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LoadBalancer" << endl;
    
    // free the communicators and groups
    for (unsigned int i=0; i < m_mpi_comm_y.size(); ++i)
        {
        if (m_mpi_comm_y[i] != MPI_COMM_NULL)
            MPI_Comm_free(&m_mpi_comm_y[i]);
        if (m_mpi_group_y[i] != MPI_GROUP_NULL)
            MPI_Group_free(&m_mpi_group_y[i]);
        }
    if (m_mpi_comm_x != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_x);
    if (m_mpi_comm_z != MPI_COMM_NULL)
        MPI_Comm_free(&m_mpi_comm_z);
    }

/*!Perform the needed calculations to balance the system load between processors
 * \param timestep Current time step of the simulation
 *
 * This needs to do something better. What it should do is reduce down different columns of the decomposition.
 * For example, go down z and sum up the particles in each xy slice. Then, go down y and sum up the particles in
 * each x slice. Compute the imbalance for each slice and rescale in x. Then, compute the imbalance in each xy slice
 * and rescale y. Finally, compute the imbalance in each xyz slice and rescale z. This requires extra reduction and stuff
 * but it should be reasonably fast if you don't do it too often.
 */
void LoadBalancer::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("balance");

    // we need a communicator, but don't want to check for it in release builds
    assert(m_comm);

    // copy the domain decomposition into GPUArrays (this could be avoided by turning the DD into GPUArrays
    // and just operating on this data, which is what we really want do do)
    std::vector<Scalar> cum_frac_x = m_decomposition->getCumulativeFractions(0);
    std::vector<Scalar> cum_frac_y = m_decomposition->getCumulativeFractions(1);
    std::vector<Scalar> cum_frac_z = m_decomposition->getCumulativeFractions(2);

    const Index3D& di = m_decomposition->getDomainIndexer();
    const uint3 my_grid_pos = m_decomposition->getGridPos();

    // reduce down z to get the number in each xy bin
    unsigned int N = m_pdata->getN();
    unsigned int sum_N(0);
    MPI_Reduce(&N, &sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_z);
    
    if (my_grid_pos.z == 0)
        {
        // only down here do we need to do a reduction
        unsigned int sum_sum_N(0);
        MPI_Reduce(&sum_N, &sum_sum_N, 1, MPI_INT, MPI_SUM, 0, m_mpi_comm_y[my_grid_pos.x]);
        
        if (my_grid_pos.y == 0)
            {
            // okay, now we need to gather all of these to the root
            std::vector<unsigned int> N_x(di.getW());
            MPI_Gather(&sum_sum_N, 1, MPI_INT, &N_x.front(), 1, MPI_INT, 0, m_mpi_comm_x);
            if (my_grid_pos.x == 0)
                {
                // do the magic of adjusting
                adjust(N_x, cum_frac_x);
                }
            
            // now scatter that out
            
            // do the y adjustment
            }
        
        // scatter that out
        
        // do the z adjustment
        }
    // scatter that out

    if (m_prof) m_prof->pop();
    }

/*!
 * \returns true if an adjustment occurred
 */
bool LoadBalancer::adjust(const std::vector<unsigned int>& N_i, std::vector<Scalar>& cum_frac_i)
    {
    
    }

void export_LoadBalancer()
    {
    class_<LoadBalancer, boost::shared_ptr<LoadBalancer>, bases<Updater>, boost::noncopyable>
    ("LoadBalancer", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<DomainDecomposition> >())
    ;
    }
#endif // ENABLE_MPI
