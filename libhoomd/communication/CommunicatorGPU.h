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

/*! \ingroup communication
*/

//! Structure for passing around parameters for worker thread invocation
struct ghost_gpu_thread_params
    {
    ghost_gpu_thread_params(const bool *_is_communicating,
                  const bool *_is_at_boundary,
                  const unsigned int *_neighbor_rank,
                  const unsigned int *_num_copy_ghosts,
                  const unsigned int *_num_recv_ghosts,
                  const unsigned int **_d_copy_ghosts,
                  Scalar4 *_d_pos_data,
                  Scalar4 *_d_pos_copybuf,
                  const BoxDim &_box,
                  const MPI_Comm& _mpi_comm)
        : is_communicating(_is_communicating),
          is_at_boundary(_is_at_boundary),
          neighbor_rank(_neighbor_rank),
          num_copy_ghosts(_num_copy_ghosts),
          num_recv_ghosts(_num_recv_ghosts),
          d_copy_ghosts(_d_copy_ghosts),
          d_pos_data(_d_pos_data),
          d_pos_copybuf(_d_pos_copybuf),
          box(_box),
          mpi_comm(_mpi_comm)
        { } 

    const bool *is_communicating;        //!< Per-direction flag if we are communicating in that direction
    const bool *is_at_boundary;          //!< Per-direction flag indicating whether our box lies at the boundary
    const unsigned int *neighbor_rank;   //!< Per-direction list of neighbor ranks
    const unsigned int *num_copy_ghosts; //!< Per-direction list of ghost particles to send
    const unsigned int *num_recv_ghosts; //!< Per-direction list of ghost particles to receive
    const unsigned int **d_copy_ghosts;  //!< Per-direction pointer to array of particle indicies to copy as ghosts
    Scalar4 *d_pos_data;                 //!< Device pointer to ghost positions array
    Scalar4 *d_pos_copybuf;              //!< Buffer pointer for copying positions
    const BoxDim& box;                   //!< Dimensions of local box
    const MPI_Comm& mpi_comm;            //!< MPI Communicator to use
    };

//! Thread that handles update of ghost particles
struct ghost_gpu_thread
    {
    //! The thread main routine
    void operator()(const ghost_gpu_thread_params& params);
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

#ifdef ENABLE_MPI_CUDA
        /*! Start ghost communication.
         * Ghost-communication can be multi-threaded, if so this method spawns the corresponding thread
         */
        virtual void startGhostsUpdate(unsigned int timestep);

        /*! Finish ghost communication.
         */
        virtual void finishGhostsUpdate(unsigned int timestep);
#endif

    protected:
        //! \name communication methods
        //@{

        //! Transfer particles between neighboring domains
        virtual void migrateAtoms();

        //! Build a ghost particle list, exchange ghost particle data with neighboring processors
        /*! \param r_ghost Width of ghost layer
         */
        virtual void exchangeGhosts();

        //! Update ghost particle positions
        virtual void copyGhosts();

        //@}

    private:
        GPUVector<unsigned char> m_remove_mask;     //!< Per-particle flags to indicate whether particle has already been sent

        GPUVector<Scalar4> m_pos_stage;             //!< Temporary storage of particle positions
        GPUVector<Scalar4> m_vel_stage;             //!< Temporary storage of particle velocities
        GPUVector<Scalar3> m_accel_stage;           //!< Temporary storage of particle accelerations
        GPUVector<int3> m_image_stage;              //!< Temporary storage of particle images
        GPUVector<Scalar> m_charge_stage;           //!< Temporary storage of particle charges
        GPUVector<Scalar> m_diameter_stage;         //!< Temporary storage of particle diameters
        GPUVector<unsigned int> m_body_stage;       //!< Temporary storage of particle body ids
        GPUVector<Scalar4> m_orientation_stage;     //!< Temporary storage of particle orientations
        GPUVector<unsigned int> m_tag_stage;        //!< Temporary storage of particle tags

        boost::thread m_worker_thread;              //!< The worker thread for updating ghost positions
        bool m_is_communicating[6];                 //!< Per-direction flag indicating whether we are communicating in that direction
        bool m_is_at_boundary[6];                  //!< Per-direction flag, true if we are at a global boundary in that direction
        unsigned int m_neighbor_rank[6];            //!< Rank of neighbor in every direction   
        const unsigned int *m_copy_ghosts_data[6];        //!< Per-direction pointers to ghost particle send list buffer
        
    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU();

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
