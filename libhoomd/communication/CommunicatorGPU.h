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

// in 3d, there are 26 neighbors max.
#define NEIGH_MAX 26

#include "Communicator.h"

#include "GPUFlags.h"
#include "GPUArray.h"
#include "GPUBufferMapped.h"

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

    protected:
        //! Perform the first part of the communication (exchange of message sizes)
        void communicateStepOne(unsigned int dir,
                                unsigned int *n_send_ptls_corner,
                                unsigned int *n_send_ptls_edge,
                                unsigned int *n_send_ptls_face,
                                unsigned int *n_recv_ptls_face,
                                unsigned int *n_recv_ptls_edge,
                                unsigned int *n_recv_ptls_local,
                                bool unique_destination);

        //! Perform the first part of the communication (exchange of particle data)
        void communicateStepTwo(unsigned int face,
                                char *corner_send_buf,
                                char *edge_send_buf,
                                char *face_send_buf,
                                const unsigned int cpitch,
                                const unsigned int epitch,
                                const unsigned int fpitch,
                                char *local_recv_buf,
                                const unsigned int *n_send_ptls_corner,
                                const unsigned int *n_send_ptls_edge,
                                const unsigned int *n_send_ptls_face,
                                const unsigned int local_recv_buf_size,
                                const unsigned int n_tot_recv_ptls_local,
                                const unsigned int element_size,
                                bool unique_destination);

        //! Check that restrictions on bond lengths etc. are not violated
        void checkValid(unsigned int timestep);

    private:
        /* Particle migration */
        GPUVector<pdata_element> m_gpu_sendbuf;        //!< Send buffer for particle data
        GPUVector<pdata_element> m_gpu_recvbuf;        //!< Receive buffer for particle data

        GPUVector<unsigned int> m_send_keys;           //!< Destination rank for particles
        GPUArray<unsigned int> m_begin;                //!< Begin index for every neighbor in send buf
        GPUArray<unsigned int> m_end;                  //!< End index for every neighbor in send buf
        GPUArray<unsigned int> m_neighbors;            //!< Neighbor ranks
        GPUArray<unsigned int> m_unique_neighbors;     //!< Neighbor ranks w/duplicates removed
        unsigned int m_nneigh;                         //!< Number of neighbors
        unsigned int m_n_unique_neigh;                 //!< Number of unique neighbors

        GPUVector<bond_element> m_gpu_bond_sendbuf;    //!< Buffer for bonds that are sent
        GPUVector<bond_element> m_gpu_bond_recvbuf;    //!< Buffer for bonds that are received

        /* Ghost communication */

        GPUVector<unsigned int> m_tag_ghost_recvbuf;    //!< Buffer for recveiving particle tags

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

        GPUArray<unsigned int> m_ghost_begin;          //!< Begin index for every plan in send buf
        GPUArray<unsigned int> m_ghost_end;            //!< Begin index for every plan in send buf

        GPUArray<unsigned int> m_adj_mask;             //!< Adjacency mask for every neighbor
        GPUVector<unsigned int> m_ghost_plan;         //!< Plans for every particle
        GPUVector<unsigned int> m_ghost_tag;          //!< Ghost particles tags, ordered by neighbor
        GPUVector<unsigned int> m_neigh_counts;       //!< List of number of neighbors to send ghost to

        unsigned int m_n_send_ghosts[NEIGH_MAX];        //!< Number of ghosts to send per neighbor
        unsigned int m_n_recv_ghosts[NEIGH_MAX];        //!< Number of ghosts to receive per neighbor
        unsigned int m_ghost_offs[NEIGH_MAX];           //!< Begin of offset in recv buf per neighbor

        unsigned int m_n_send_ghosts_tot;              //!< Total number of sent ghosts
        unsigned int m_n_recv_ghosts_tot;              //!< Total number of received ghosts

        cached_allocator m_cached_alloc;              //!< Cached memory allocator for internal thrust code

        //! Helper function to allocate various buffers
        void allocateBuffers();

        //! Helper function to initialize neighbor arrays
        void initializeNeighborArrays();

    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU();

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
