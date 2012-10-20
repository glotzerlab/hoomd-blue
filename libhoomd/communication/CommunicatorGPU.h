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

#include "Communicator.h"

#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>

#include "WorkQueue.h"
#include "GPUFlags.h"

/*! \ingroup communication
*/

//! Structure for passing around parameters for worker thread invocation
struct ghost_gpu_thread_params
    {
    //! Constructor
    ghost_gpu_thread_params(const unsigned int *_ghost_idx_face_handle, 
                            const unsigned int _ghost_idx_face_pitch,   
                            const unsigned int *_ghost_idx_edge_handle, 
                            const unsigned int _ghost_idx_edge_pitch,   
                            const unsigned int *_ghost_idx_corner_handle, 
                            const unsigned int _ghost_idx_corner_pitch, 
                            char *_corner_update_buf_handle,            
                            const unsigned int _corner_update_buf_pitch, 
                            char *_edge_update_buf_handle,             
                            const unsigned int _edge_update_buf_pitch, 
                            char *_face_update_buf_handle,             
                            const unsigned int _face_update_buf_pitch, 
                            char *_update_recv_buf_handle,             
                            unsigned int *_d_ghost_plan,
                            const unsigned int _N,                     
                            const unsigned int _recv_ghosts_local_size,
                            const unsigned int *_n_recv_ghosts_edge,
                            const unsigned int *_n_recv_ghosts_face,
                            const unsigned int *_n_recv_ghosts_local,
                            const GPUArray<unsigned int>& _n_local_ghosts_corner,  
                            const GPUArray<unsigned int>& _n_local_ghosts_edge,  
                            const GPUArray<unsigned int>& _n_local_ghosts_face,  
                            Scalar4 *_pos_handle,                      
                            const BoxDim& _global_box)
        : ghost_idx_face_handle(_ghost_idx_face_handle),
          ghost_idx_face_pitch(_ghost_idx_face_pitch),
          ghost_idx_edge_handle(_ghost_idx_edge_handle),
          ghost_idx_edge_pitch(_ghost_idx_edge_pitch),
          ghost_idx_corner_handle(_ghost_idx_corner_handle),
          ghost_idx_corner_pitch(_ghost_idx_corner_pitch),
          corner_update_buf_handle(_corner_update_buf_handle),
          corner_update_buf_pitch(_corner_update_buf_pitch),
          edge_update_buf_handle(_edge_update_buf_handle),
          edge_update_buf_pitch(_edge_update_buf_pitch),
          face_update_buf_handle(_face_update_buf_handle),
          face_update_buf_pitch(_face_update_buf_pitch),
          update_recv_buf_handle(_update_recv_buf_handle),
          d_ghost_plan(_d_ghost_plan),
          N(_N),
          recv_ghosts_local_size(_recv_ghosts_local_size),
          n_recv_ghosts_edge(_n_recv_ghosts_edge),
          n_recv_ghosts_face(_n_recv_ghosts_face),
          n_recv_ghosts_local(_n_recv_ghosts_local),
          n_local_ghosts_corner(_n_local_ghosts_corner),
          n_local_ghosts_edge(_n_local_ghosts_edge),
          n_local_ghosts_face(_n_local_ghosts_face),
          pos_handle(_pos_handle),
          global_box(_global_box)
        { }

    const unsigned int *ghost_idx_face_handle; //!< Device pointer to 'face' ghost particle indices array
    const unsigned int ghost_idx_face_pitch;   //!< Pitch of 'face' ghost particle indices array
    const unsigned int *ghost_idx_edge_handle; //!< Device pointer to 'edge' ghost particle indices array
    const unsigned int ghost_idx_edge_pitch;   //!< Pitch of 'edge' ghost particle indices array
    const unsigned int *ghost_idx_corner_handle; //!< Device pointer to 'corner' ghost particle indices array
    const unsigned int ghost_idx_corner_pitch; //!< Pitch of 'corner' ghost particle indices array
    char *corner_update_buf_handle;            //!< Send/recv buffer for ghosts that are updated over a corner
    const unsigned int corner_update_buf_pitch; //!< Pitch of corner ghost update buffer
    char *edge_update_buf_handle;             //!< Send/recv buffer for ghosts that are updated over a edge
    const unsigned int edge_update_buf_pitch; //!< Pitch of edge ghost update buffer
    char *face_update_buf_handle;             //!< Send/recv buffer for ghosts that are updated over a face
    const unsigned int face_update_buf_pitch; //!< Pitch of face ghost update buffer
    char *update_recv_buf_handle;             //!< Buffer for ghosts received for the local box
    unsigned int *d_ghost_plan;               //!< Array of plans received ghosts
    const unsigned int N;                     //!< Number of local particles
    const unsigned int recv_ghosts_local_size; //!< Size of receive buffer for ghosts addressed to the local domain
    const unsigned int *n_recv_ghosts_edge;   //!< Number of ghosts received for updating over an edge
    const unsigned int *n_recv_ghosts_face;   //!< Number of ghosts received for updating over a face
    const unsigned int *n_recv_ghosts_local;  //!< Number of ghosts received for the local box
    const GPUArray<unsigned int>& n_local_ghosts_corner;//!< Number of local ghosts sent over a corner
    const GPUArray<unsigned int>& n_local_ghosts_edge;  //!< Number of local ghosts sent over an edge
    const GPUArray<unsigned int>& n_local_ghosts_face;  //!< Number of local ghosts sent over a face
    Scalar4 *pos_handle;                      //!< Device pointer to ghost positions array
    const BoxDim& global_box;                 //!< Dimensions of global box
    };

//! Forward declaration
class CommunicatorGPU;

//! Thread that handles update of ghost particles
class ghost_gpu_thread
    {
    public:
        //! Constructor
        ghost_gpu_thread(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                         CommunicatorGPU *communicator);
        virtual ~ghost_gpu_thread();

        //! The thread main routine
        void operator()(WorkQueue<ghost_gpu_thread_params>& queue, boost::barrier& barrier);

    private:
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< The execution configuration
        CommunicatorGPU *m_communicator;                              //!< Pointer to the communciator that called the thread

        unsigned int m_thread_id;                                     //!< GPU thread id

        char *h_recv_buf;                                             //!< Host receive buffer
        char *h_face_update_buf;                                      //!< Host buffer of particles that are sent through a face
        char *h_edge_update_buf;                                      //!< Host buffer of particles that are sent over an edge
        char *h_corner_update_buf;                                    //!< Host buffer of particles that are sent over a corner

        unsigned int m_recv_buf_size;                                 //!< Size of host receive buffer
        unsigned int m_face_update_buf_size;                          //!< Size of host send buffer for 'face' ptls
        unsigned int m_edge_update_buf_size;                          //!< Size of host send buffer for 'edge' ptls
        unsigned int m_corner_update_buf_size;                        //!< Size of host send buffer for 'corner' ptls

        bool m_buffers_allocated;                                     //!< True if host buffers have been allocated

        //! The routine that does the actual ghost update
        /*! \param params The parameters for this update
         */
        void update_ghosts(ghost_gpu_thread_params& params);
    };

//! Class that handles MPI communication (GPU version)
class CommunicatorGPU : public Communicator
    {
    public:
        //! Constructor
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param decomposition Information about the decomposition of the global simulation domain
         */
        CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                        boost::shared_ptr<DomainDecomposition> decomposition);
        virtual ~CommunicatorGPU();

        /*! Start ghost communication.
         * Ghost-communication can be multi-threaded, if so this method spawns the corresponding thread
         */
        virtual void startGhostsUpdate(unsigned int timestep);

        /*! Finish ghost communication.
         */
        virtual void finishGhostsUpdate(unsigned int timestep);

        //! \name communication methods
        //@{

        //! Transfer particles between neighboring domains
        virtual void migrateAtoms();

        //! Build a ghost particle list, exchange ghost particle data with neighboring processors
        /*! \param r_ghost Width of ghost layer
         */
        virtual void exchangeGhosts();

        //@}

    protected:
        //! Helper function to perform the first part of the communication (exchange of message sizes)
        void communicateStepOne(unsigned int dir,
                                unsigned int *n_send_ptls_corner,
                                unsigned int *n_send_ptls_edge,
                                unsigned int *n_send_ptls_face,
                                unsigned int *n_recv_ptls_face,
                                unsigned int *n_recv_ptls_edge,
                                unsigned int *n_recv_ptls_local,
                                bool unique_destination);

        //! Helper function to perform the first part of the communication (exchange of particle data)
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

    private:
        GPUVector<unsigned char> m_remove_mask;     //!< Per-particle flags to indicate whether particle has already been sent

        GPUArray<char> m_corner_send_buf;          //!< Send buffer for corner ptls
        GPUArray<char> m_edge_send_buf;            //!< Send buffer for edge ptls
        GPUArray<char> m_face_send_buf;            //!< Send buffer for edge ptls
        GPUArray<char> m_recv_buf;                 //!< Receive buffer for particle data

        unsigned int m_max_send_ptls_corner;        //!< Size of corner ptl send buffer
        unsigned int m_max_send_ptls_edge;          //!< Size of edge ptl send buffer
        unsigned int m_max_send_ptls_face;          //!< Size of face ptl send buffer

        GPUArray<char> m_corner_ghosts_buf;         //!< Copy buffer for ghosts lying at the edge
        GPUArray<char> m_edge_ghosts_buf;           //!< Copy buffer for ghosts lying in the corner
        GPUArray<char> m_face_ghosts_buf;           //!< Copy buffer for ghosts lying near a face
        GPUArray<char> m_ghosts_recv_buf;           //!< Receive buffer for particle data
        GPUArray<unsigned int> m_ghost_plan;        //!< Routing plans for received ghost particles

        GPUArray<unsigned int> m_ghost_idx_corner;  //!< Indices of particles copied as ghosts via corner
        GPUArray<unsigned int> m_ghost_idx_edge;    //!< Indices of particles copied as ghosts via an edge
        GPUArray<unsigned int> m_ghost_idx_face;    //!< Indices of particles copied as ghosts via a face

        GPUArray<char> m_corner_update_buf;         //!< Copy buffer for 'corner' ghost positions 
        GPUArray<char> m_edge_update_buf;           //!< Copy buffer for 'corner' ghost positions 
        GPUArray<char> m_face_update_buf;           //!< Copy buffer for 'corner' ghost positions 
        GPUArray<char> m_update_recv_buf;           //!< Receive buffer for ghost positions 

        char *h_ghosts_recv_buf;                    //!< Host receive buffer
        char *h_face_ghosts_buf;                    //!< Host buffer of particles that are sent through a face
        char *h_edge_ghosts_buf;                    //!< Host buffer of particles that are sent over an edge
        char *h_corner_ghosts_buf;                  //!< Host buffer of particles that are sent over a corner

        unsigned int m_max_copy_ghosts_corner;      //!< Maximum number of ghosts 'corner' particles
        unsigned int m_max_copy_ghosts_edge;        //!< Maximum number of ghosts 'edge' particles
        unsigned int m_max_copy_ghosts_face;        //!< Maximum number of ghosts 'face' particles
        unsigned int m_max_recv_ghosts;             //!< Maximum number of ghosts received for the local box

        GPUArray<unsigned int> m_n_local_ghosts_face;  //!< Number of local ghosts sent over a face
        GPUArray<unsigned int> m_n_local_ghosts_edge;  //!< Local ghosts sent over an edge
        GPUArray<unsigned int> m_n_local_ghosts_corner;//!< Local ghosts sent over a corner

        unsigned int m_n_recv_ghosts_face[6*6];     //!< Number of received ghosts for sending over a face, per direction
        unsigned int m_n_recv_ghosts_edge[12*6];    //!< Number of received ghosts for sending over an edge, per direction
        unsigned int m_n_recv_ghosts_local[6];      //!< Number of received ghosts that stay in the local box, per direction


        bool m_buffers_allocated;                   //!< True if buffers have been allocated

        const float m_resize_factor;                //!< Factor used for amortized array resizing
        GPUFlags<unsigned int> m_condition;         //!< Condition variable set to a value unequal zero if send buffers need to be resized

        boost::thread m_worker_thread;              //!< The worker thread for updating ghost positions
        bool m_thread_created;                      //!< True if the worker thread has been created
        WorkQueue<ghost_gpu_thread_params> m_work_queue; //!< The queue of parameters processed by the worker thread
        boost::barrier m_barrier;                   //!< Barrier to synchronize with worker thread

        MPI_Group m_comm_group;                     //!< Group corresponding to MPI communicator
        MPI_Win m_win_edge[12];                     //!< Shared memory windows for every of the 12 edges
        MPI_Win m_win_face[6];                      //!< Shared memory windows for every of the 6 edges
        MPI_Win m_win_local;                        //!< Shared memory window for locally received particles

        //! Helper function to allocate various buffers
        void allocateBuffers();

        //! Check and resize ghost buffers if necessary
        void checkReallocateGhostBuffers();

        //!< De-allocate host buffers
        void deallocateBuffers();

        friend class ghost_gpu_thread;
    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU();

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
