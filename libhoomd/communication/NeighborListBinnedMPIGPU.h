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

/*! \file NeighborListBinnedMPIGPU.h
    \brief Defines the NeighborListBinnedMPIGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#ifndef __NEIGHBOR_LIST_BINNED_MPI_GPU_H
#define __NEIGHBOR_LIST_BINNED_MPI_GPU_H

#include "NeighborListGPUBinned.h"

//! forward declarations
class Communicator;

//! This class defines a neighbor list to be used in parallel MPI simulations
/*! This class extends the neighbor list construction by two communication steps,
    which are implemented in a separate communication class.

    Before the neighbor list is updated, particle data and ghost particles
    are exchanged with neighboring processors.

    To determine when the neighbor list needs to be updated, a global distance check
    criterium is applied. If a distance check indicates that the neighbor list needs to be rebuilt
    ony any processor, i .e. any particle on that processor has moved more than the \c r_cut + r_buff,
    the global criterium is fulfilled.
 */
class NeighborListBinnedMPIGPU : public NeighborListGPUBinned
    {
    public:
        //! Constructor
        /*! \param sysdef system definition to construct this neighbor list from
         * \param r_cut cutoff radius
         * \param r_buff skin length
         * \param comm the communicator this neighbor list is assoicated with
         * \param cl the cell list to use for binning the particles
         */
        NeighborListBinnedMPIGPU(boost::shared_ptr<SystemDefinition> sysdef,
                              Scalar r_cut,
                              Scalar r_buff,
                              boost::shared_ptr<Communicator> comm,
                              boost::shared_ptr<CellList> cl = boost::shared_ptr<CellList>());


    protected:
        //! Apply a global distance check criterium
        virtual bool distanceCheck();

        //! This method is called when the neighbor list needs to be rebuilt
        virtual void buildNlist(unsigned int timestep);

        boost::shared_ptr<Communicator> m_comm; //!< the communication class

    private:
        unsigned int m_last_exchange_step;      //!< timestep at which last exchange occured
        bool m_first_exchange;                  //!< indicates whether the particles have not been exchanged before
    };

#endif // __NEIGHBOR_LIST_BINNED_MPI_GPU_H
#endif // ENABLE_CUDA
#endif // ENABLE_MPI
