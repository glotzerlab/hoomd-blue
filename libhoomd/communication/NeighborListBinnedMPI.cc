/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

/*! \file NeighborListBinnedMPI.cc
    \brief Implements the NeighborListBinnedMPI class (CPU version)
*/

#ifdef ENABLE_MPI
#include "NeighborListBinnedMPI.h"
#include "Communicator.h"

#include <boost/mpi.hpp>

//! Constructor
NeighborListBinnedMPI::NeighborListBinnedMPI(boost::shared_ptr<SystemDefinition> sysdef,
    Scalar r_cut,
    Scalar r_buff,
    boost::shared_ptr<Communicator> comm,
    boost::shared_ptr<CellList> cl)
    : NeighborListBinned(sysdef, r_cut, r_buff, cl), m_comm(comm), m_last_exchange_step(0), m_first_exchange(true)
    {
    for (unsigned int i = 0; i < 3; i++)
        {
        unsigned int dim = m_comm->getDimension(i);
        this->setGhostLayer(i, (dim > 1) ? true : false);
        this->setMinimumImage(i, (dim > 1) ? false : true); // do not use minium image convention, we use ghost particles
        }
    }

//! Evaluate global distance check criterium
bool NeighborListBinnedMPI::distanceCheck()
    {
    unsigned int local_flag = NeighborListBinned::distanceCheck() ? 1 : 0;

    unsigned int res;
    all_reduce(*m_comm->getMPICommunicator(), local_flag, res,  boost::mpi::maximum<unsigned int>());
    return (res > 0);
    }

//! Build the neighborlist
void NeighborListBinnedMPI::buildNlist(unsigned int timestep)
    {
    if (m_first_exchange || m_last_exchange_step != timestep)
    {
    m_first_exchange = false;
    m_last_exchange_step = timestep;

    // migrate atoms
    m_comm->migrateAtoms();

    // exchange ghosts
    Scalar rmax = m_r_cut + m_r_buff;
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);

    m_comm->exchangeGhosts(rmax);
    }

    // build the neighbor list
    NeighborListBinned::buildNlist(timestep);
    }
#endif // ENABLE_MPI
