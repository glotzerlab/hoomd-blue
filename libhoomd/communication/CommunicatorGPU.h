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

//#define MPI3 // define if the MPI implementation supports MPI3 one-sided communications

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
        GPUVector<pdata_element> m_gpu_sendbuf;        //!< Send buffer for particle data
        GPUVector<pdata_element> m_gpu_recvbuf;        //!< Receive buffer for particle data

        GPUVector<bond_element> m_gpu_bond_sendbuf;   //!< Buffer for bonds that are sent
        GPUVector<bond_element> m_gpu_bond_recvbuf;   //!< Buffer for bonds that are received

        unsigned int m_remote_send_corner[8*6];     //!< Remote corner particles, per direction
        unsigned int m_remote_send_edge[12*6];       //!< Remote edge particles, per direction
        unsigned int m_remote_send_face[6];         //!< Remote face particles, per direction

        GPUArray<char> m_corner_ghosts_buf;         //!< Copy buffer for ghosts lying at the edge
        GPUArray<char> m_edge_ghosts_buf;           //!< Copy buffer for ghosts lying in the corner
        GPUArray<char> m_face_ghosts_buf;           //!< Copy buffer for ghosts lying near a face
        GPUArray<char> m_ghosts_recv_buf;           //!< Receive buffer for particle data

        GPUArray<unsigned int> m_ghost_idx_corner;  //!< Indices of particles copied as ghosts via corner
        GPUArray<unsigned int> m_ghost_idx_edge;    //!< Indices of particles copied as ghosts via an edge
        GPUArray<unsigned int> m_ghost_idx_face;    //!< Indices of particles copied as ghosts via a face

        GPUArray<char> m_corner_update_buf;   //!< Copy buffer for 'corner' ghost positions
        GPUArray<char> m_edge_update_buf;     //!< Copy buffer for 'corner' ghost positions
        GPUArray<char> m_face_update_buf;     //!< Copy buffer for 'corner' ghost positions
        GPUArray<char> m_update_recv_buf;     //!< Receive buffer for ghost positions

        unsigned int m_max_copy_ghosts_face;        //!< Maximum number of ghosts 'face' particles
        unsigned int m_max_copy_ghosts_edge;        //!< Maximum number of ghosts 'edge' particles
        unsigned int m_max_copy_ghosts_corner;      //!< Maximum number of ghosts 'corner' particles
        unsigned int m_max_recv_ghosts;             //!< Maximum number of ghosts received for the local box

        GPUArray<unsigned int> m_n_local_ghosts_face;  //!< Number of local ghosts sent over a face
        GPUArray<unsigned int> m_n_local_ghosts_edge;  //!< Local ghosts sent over an edge
        GPUArray<unsigned int> m_n_local_ghosts_corner;//!< Local ghosts sent over a corner

        GPUArray<unsigned int> m_n_recv_ghosts_face; //!< Number of received ghosts for sending over a face, per direction
        GPUArray<unsigned int> m_n_recv_ghosts_edge; //!< Number of received ghosts for sending over an edge, per direction
        GPUArray<unsigned int> m_n_recv_ghosts_local;//!< Number of received ghosts that stay in the local box, per direction

        unsigned int m_n_tot_recv_ghosts;           //!< Total number of received ghots
        unsigned int m_n_tot_recv_ghosts_local;     //!< Total number of received ghosts for local box
        unsigned int m_n_forward_ghosts_face[6];    //!< Total number of received ghosts for the face send buffer
        unsigned int m_n_forward_ghosts_edge[12];   //!< Total number of received ghosts for the edge send buffer

        bool m_buffers_allocated;                   //!< True if buffers have been allocated
        bool m_mesh_buffers_allocated;              //!< True if buffers for ghost mesh exchange have been allocated

        uint3 m_mesh_dim;                           //!< Dimensions of mesh for which we allocated buffers
        uint3 m_inner_dim;                          //!< Number of non-ghost cells along every axis
        unsigned int m_n_ghost_cells;               //!< Number of ghost cells of mesh along non-periodic axes
        unsigned int m_n_corner_cells;              //!< Number of corner cells
        uint3 m_n_edge_cells;                       //!< Number of edge cells, for every four edges along every axis
        uint3 m_n_face_cells;                       //!< Number of face cells for every axis

        GPUFlags<unsigned int> m_condition;         //!< Condition variable set to a value unequal zero if send buffers need to be resized

#ifdef MPI3
        MPI_Group m_comm_group;                     //!< Group corresponding to MPI communicator
        MPI_Win m_win_edge[12];                     //!< Shared memory windows for every of the 12 edges
        MPI_Win m_win_face[6];                      //!< Shared memory windows for every of the 6 edges
        MPI_Win m_win_local;                        //!< Shared memory window for locally received particles
#endif

        //! Helper function to allocate various buffers
        void allocateBuffers();

        //! Helper function to allocate buffers for ghost mesh exchange
        void allocateMeshBuffers(const uint3 dim, const uint3 n_ghost_cells, size_t size_datatype);

        //! Compute size of ghost exchange element
        size_t ghost_exchange_element_size();

        //! Compute size of ghost update element
        size_t ghost_update_element_size();

    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU();

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
