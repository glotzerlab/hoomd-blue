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

/*! \file CommunicatorGPU.h
    \brief Defines the CommunicatorGPU class
*/

#ifndef __COMMUNICATOR_GPU_H__
#define __COMMUNICATOR_GPU_H__

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

// in 3d, there are 27 neighbors max.
#define NEIGH_MAX 27

#include "Communicator.h"

#include "CommunicatorGPU.cuh"

#include "GPUFlags.h"
#include "GPUArray.h"

/*! \ingroup communication
*/

//! Class that handles MPI communication (GPU version)
/*! CommunicatorGPU uses a GPU optimized version of the basic Plimpton communication scheme implemented in the base
    class Communicator.

    Basically, particles are pre-sorted into face, edge and corner buffers depending whether they neighbor one, two or three
    boxes. The full algorithm will be documented in a forthcoming publication.

    This scheme guarantees that in between every of the six communication steps, no extra scanning of particle buffers needs
    to be done and only buffer copying on the host is involved. Since for MPI, data needs to reside on the host anyway,
    this avoids unnecessary copying of data between the GPU and the host.
*/
class CommunicatorGPU : public Communicator
    {
    public:
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param decomposition Information about the decomposition of the global simulation domain
         */
        CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                        boost::shared_ptr<DomainDecomposition> decomposition);
        virtual ~CommunicatorGPU();

        //! \name communication methods
        //@{

        /*! Perform ghosts update
         */
        virtual void updateGhosts(unsigned int timestep);

        //! Transfer particles between neighboring domains
        virtual void migrateParticles();

        //! Build a ghost particle list, exchange ghost particle data with neighboring processors
        virtual void exchangeGhosts();

        //@}

        //! Set maximum number of communication stages
        /*! \param max_stages Maximum number of communication stages
         */
        void setMaxStages(unsigned int max_stages)
            {
            m_max_stages = max_stages;
            initializeCommunicationStages();
            forceMigrate();
            }

    private:
        /* General communication */
        GPUArray<unsigned int> m_begin;                //!< Begin index for every neighbor in send buf
        GPUArray<unsigned int> m_end;                  //!< End index for every neighbor in send buf
        GPUArray<unsigned int> m_adj_mask;             //!< Adjacency mask for every neighbor
        GPUArray<unsigned int> m_neighbors;            //!< Neighbor ranks
        GPUArray<unsigned int> m_unique_neighbors;     //!< Neighbor ranks w/duplicates removed
        unsigned int m_nneigh;                         //!< Number of neighbors
        unsigned int m_n_unique_neigh;                 //!< Number of unique neighbors

        unsigned int m_max_stages;                     //!< Maximum number of (dependent) communication stages
        unsigned int m_num_stages;                     //!< Number of stages
        std::vector<unsigned int> m_comm_mask;         //!< Communication mask per stage
        std::vector<int> m_stages;                     //!< Communication stage per unique neighbor

        /* Particle migration */
        GPUVector<pdata_element> m_gpu_sendbuf;        //!< Send buffer for particle data
        GPUVector<pdata_element> m_gpu_recvbuf;        //!< Receive buffer for particle data

        GPUVector<unsigned int> m_send_keys;           //!< Destination rank for particles

        /* Ghost communication */
        GPUVector<unsigned int> m_tag_ghost_sendbuf;   //!< List of ghost particles tags per stage, ordered by neighbor
        GPUVector<unsigned int> m_tag_ghost_recvbuf;   //!< Buffer for recveiving particle tags
        GPUVector<Scalar4> m_pos_ghost_sendbuf;        //<! Buffer for sending ghost positions
        GPUVector<Scalar4> m_pos_ghost_recvbuf;        //<! Buffer for receiving ghost positions

        GPUVector<Scalar4> m_vel_ghost_sendbuf;        //<! Buffer for sending ghost velocities
        GPUVector<Scalar4> m_vel_ghost_recvbuf;        //<! Buffer for receiving ghost velocities

        GPUVector<Scalar> m_charge_ghost_sendbuf;      //!< Buffer for sending ghost charges
        GPUVector<Scalar> m_charge_ghost_recvbuf;      //!< Buffer for sending ghost charges

        GPUVector<Scalar> m_diameter_ghost_sendbuf;    //!< Buffer for sending ghost charges
        GPUVector<Scalar> m_diameter_ghost_recvbuf;    //!< Buffer for sending ghost charges

        GPUVector<Scalar4> m_orientation_ghost_sendbuf;//<! Buffer for sending ghost orientations
        GPUVector<Scalar4> m_orientation_ghost_recvbuf;//<! Buffer for receiving ghost orientations

        GPUVector<unsigned int> m_ghost_begin;          //!< Begin index for every stage and neighbor in send buf
        GPUVector<unsigned int> m_ghost_end;            //!< Begin index for every and neighbor in send buf

        GPUVector<unsigned int> m_ghost_idx;          //!< Indices of ghosts to send
        GPUVector<unsigned int> m_ghost_plan;         //!< Plans for every particle
        std::vector<unsigned int> m_idx_offs;         //!< Per-stage offset into ghost idx list

        GPUVector<unsigned int> m_neigh_counts;       //!< List of number of neighbors to send ghost to (temp array)

        std::vector<std::vector<unsigned int> > m_n_send_ghosts; //!< Number of ghosts to send per stage and neighbor
        std::vector<std::vector<unsigned int> > m_n_recv_ghosts; //!< Number of ghosts to receive per stage and neighbor
        std::vector<std::vector<unsigned int> > m_ghost_offs;    //!< Begin of offset in recv buf per stage and neighbor

        std::vector<unsigned int> m_n_send_ghosts_tot; //!< Total number of sent ghosts per stage
        std::vector<unsigned int> m_n_recv_ghosts_tot; //!< Total number of received ghosts per stage

        CommFlags m_last_flags;                       //! Flags of last ghost exchange
        cached_allocator m_cached_alloc;              //!< Cached memory allocator for internal thrust code
        mgpu::ContextPtr m_mgpu_context;                    //!< MGPU context

        //! Helper function to allocate various buffers
        void allocateBuffers();

        //! Helper function to initialize neighbor arrays
        void initializeNeighborArrays();

        //! Helper function to set up communication stages
        void initializeCommunicationStages();
    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU();

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
