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

/*! \file CommunicatorGPU.cc
    \brief Implements the CommunicatorGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "CommunicatorGPU.h"
#include "CommunicatorGPU.cuh"
#include "Profiler.h"
#include "System.h"

#include <boost/mpi.hpp>
#include <boost/python.hpp>
using namespace boost::python;

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(int3)

BOOST_CLASS_TRACKING(Scalar4,track_never)

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_remove_mask(m_exec_conf),
      m_pos_stage(m_exec_conf),
      m_vel_stage(m_exec_conf),
      m_accel_stage(m_exec_conf),
      m_image_stage(m_exec_conf),
      m_charge_stage(m_exec_conf),
      m_diameter_stage(m_exec_conf),
      m_body_stage(m_exec_conf),
      m_orientation_stage(m_exec_conf),
      m_tag_stage(m_exec_conf)
    {
    // allocate temporary GPU buffers
    gpu_allocate_tmp_storage();
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    gpu_deallocate_tmp_storage();
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_atoms");

        {
        // Reset reverse lookup tags of old ghost atoms
        ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

        gpu_reset_rtags(m_pdata->getNGhosts(),
                        d_global_tag.data + m_pdata->getN(),
                        d_global_rtag.data);

        CHECK_CUDA_ERROR();
        }

    // reset ghost particle number
    m_pdata->removeAllGhostParticles();

    m_remove_mask.clear();

    for (unsigned int dir=0; dir < 6; dir++)
        {
        unsigned int n_send_ptls;

        if (! isCommunicating(dir) ) continue;

        if (m_prof)
            m_prof->push("remove ptls");

        // Reallocate send buffers
        m_pos_stage.resize(m_pdata->getN());
        m_vel_stage.resize(m_pdata->getN());
        m_accel_stage.resize(m_pdata->getN());
        m_charge_stage.resize(m_pdata->getN());
        m_diameter_stage.resize(m_pdata->getN());
        m_image_stage.resize(m_pdata->getN());
        m_body_stage.resize(m_pdata->getN());
        m_orientation_stage.resize(m_pdata->getN());
        m_tag_stage.resize(m_pdata->getN());

        // resize mask and set newly allocated flags to zero
        m_remove_mask.resize(m_pdata->getN());

            {
            // remove all particles from our domain that are going to be sent in the current direction

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::readwrite);

            // Stage particle data for sending
            gpu_migrate_select_particles(m_pdata->getN(),
                                   n_send_ptls,
                                   d_remove_mask.data,                                
                                   d_pos.data,
                                   d_pos_stage.data,
                                   d_vel.data,
                                   d_vel_stage.data,
                                   d_accel.data,
                                   d_accel_stage.data,
                                   d_image.data,
                                   d_image_stage.data,
                                   d_charge.data,
                                   d_charge_stage.data,
                                   d_diameter.data,
                                   d_diameter_stage.data,
                                   d_body.data,
                                   d_body_stage.data,
                                   d_orientation.data,
                                   d_orientation_stage.data,
                                   d_global_tag.data,
                                   d_tag_stage.data,
                                   m_pdata->getBox(),
                                   dir);
            CHECK_CUDA_ERROR();
            
            }

        if (m_prof)
	        m_prof->pop();

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        unsigned int n_recv_ptls;
        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[20];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,n_send_ptls);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,n_recv_ptls);
        boost::mpi::wait_all(reqs,reqs+2);

        // start index for atoms to be added
        unsigned int add_idx = m_pdata->getN();

        // allocate memory for particles that will be received
        m_pdata->addParticles(n_recv_ptls);

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::read);
            ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);


            // exchange actual particle data
            reqs[2] = m_mpi_comm->isend(send_neighbor,1,d_pos_stage.data,n_send_ptls);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,d_pos.data+add_idx,n_recv_ptls);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,d_vel_stage.data,n_send_ptls);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,d_vel.data+add_idx,n_recv_ptls);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,d_accel_stage.data,n_send_ptls);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,d_accel.data+add_idx,n_recv_ptls);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,d_image_stage.data,n_send_ptls);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,d_image.data+add_idx,n_recv_ptls);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,d_charge_stage.data,n_send_ptls);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,d_charge.data+add_idx,n_recv_ptls);

            reqs[12] = m_mpi_comm->isend(send_neighbor,6,d_diameter_stage.data,n_send_ptls);
            reqs[13] = m_mpi_comm->irecv(recv_neighbor,6,d_diameter.data+add_idx,n_recv_ptls);

            reqs[14] = m_mpi_comm->isend(send_neighbor,7,d_diameter_stage.data,n_send_ptls);
            reqs[15] = m_mpi_comm->irecv(recv_neighbor,7,d_diameter.data+add_idx,n_recv_ptls);

            reqs[16] = m_mpi_comm->isend(send_neighbor,8,d_body_stage.data,n_send_ptls);
            reqs[17] = m_mpi_comm->irecv(recv_neighbor,8,d_body.data+add_idx,n_recv_ptls);

            reqs[18] = m_mpi_comm->isend(send_neighbor,9,d_tag_stage.data,n_send_ptls);
            reqs[19] = m_mpi_comm->irecv(recv_neighbor,9,d_global_tag.data+add_idx,n_recv_ptls);
            boost::mpi::wait_all(reqs+2,reqs+20);
            }

#else
            {
            ArrayHandle<Scalar4> h_pos_stage(m_pos_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel_stage(m_vel_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_accel_stage(m_accel_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_stage(m_charge_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_stage(m_diameter_stage, access_location::host, access_mode::read);
            ArrayHandle<int3> h_image_stage(m_image_stage, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_stage(m_body_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_stage(m_orientation_stage, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag_stage(m_tag_stage, access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);



            // exchange actual particle data
            reqs[2] = m_mpi_comm->isend(send_neighbor,1,h_pos_stage.data,n_send_ptls);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,h_pos.data+adh_idx,n_recv_ptls);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,h_vel_stage.data,n_send_ptls);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,h_vel.data+adh_idx,n_recv_ptls);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,h_accel_stage.data,n_send_ptls);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,h_accel.data+adh_idx,n_recv_ptls);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,h_image_stage.data,n_send_ptls);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,h_image.data+adh_idx,n_recv_ptls);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,h_charge_stage.data,n_send_ptls);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,h_charge.data+adh_idx,n_recv_ptls);

            reqs[12] = m_mpi_comm->isend(send_neighbor,6,h_diameter_stage.data,n_send_ptls);
            reqs[13] = m_mpi_comm->irecv(recv_neighbor,6,h_diameter.data+adh_idx,n_recv_ptls);

            reqs[14] = m_mpi_comm->isend(send_neighbor,7,h_diameter_stage.data,n_send_ptls);
            reqs[15] = m_mpi_comm->irecv(recv_neighbor,7,h_diameter.data+adh_idx,n_recv_ptls);

            reqs[16] = m_mpi_comm->isend(send_neighbor,8,h_body_stage.data,n_send_ptls);
            reqs[17] = m_mpi_comm->irecv(recv_neighbor,8,h_body.data+adh_idx,n_recv_ptls);

            reqs[18] = m_mpi_comm->isend(send_neighbor,9,h_tag_stage.data,n_send_ptls);
            reqs[19] = m_mpi_comm->irecv(recv_neighbor,9,h_global_tag.data+adh_idx,n_recv_ptls);
            boost::mpi::wait_all(reqs+2,reqs+20);
            }
#endif

        if (m_prof)
            m_prof->pop();

            {
            ArrayHandle<float4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
            gpu_migrate_wrap_received_particles(d_pos.data + add_idx,
                                                d_image.data + add_idx,
                                                n_recv_ptls,
                                                m_pdata->getGlobalBox(),
                                                dir,
                                                m_is_at_boundary);
            CHECK_CUDA_ERROR();
            }

        } // end dir loop


    unsigned int n_remove_ptls;

    // Reallocate particle data buffers
    // it is important to use the actual size of the arrays as arguments,
    // which can be larger than the particle number
    m_pos_stage.resize(m_pdata->getPositions().getNumElements());
    m_vel_stage.resize(m_pdata->getVelocities().getNumElements());
    m_accel_stage.resize(m_pdata->getAccelerations().getNumElements());
    m_charge_stage.resize(m_pdata->getCharges().getNumElements());
    m_diameter_stage.resize(m_pdata->getDiameters().getNumElements());
    m_image_stage.resize(m_pdata->getImages().getNumElements());
    m_body_stage.resize(m_pdata->getBodies().getNumElements());
    m_orientation_stage.resize(m_pdata->getOrientationArray().getNumElements());
    m_tag_stage.resize(m_pdata->getGlobalTags().getNumElements());

    m_remove_mask.resize(m_pdata->getN());
    CHECK_CUDA_ERROR();

        {
        // reset rtag of deleted particles
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);
        gpu_reset_rtags_by_mask(m_pdata->getN(),
                               d_remove_mask.data,
                               d_global_tag.data,
                               d_global_rtag.data);
        CHECK_CUDA_ERROR();
        }


        {
        // Final array compaction
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);

        gpu_migrate_compact_particles(m_pdata->getN(),
                               d_remove_mask.data,
                               n_remove_ptls,
                               d_pos.data,
                               d_pos_stage.data,
                               d_vel.data,
                               d_vel_stage.data,
                               d_accel.data,
                               d_accel_stage.data,
                               d_image.data,
                               d_image_stage.data,
                               d_charge.data,
                               d_charge_stage.data,
                               d_diameter.data,
                               d_diameter_stage.data,
                               d_body.data,
                               d_body_stage.data,
                               d_orientation.data,
                               d_orientation_stage.data,
                               d_global_tag.data,
                               d_tag_stage.data);
        CHECK_CUDA_ERROR();

        }
   
    // Swap temporary arrays with particle data arrays
    m_pdata->getPositions().swap(m_pos_stage);
    m_pdata->getVelocities().swap(m_vel_stage);
    m_pdata->getAccelerations().swap(m_accel_stage);
    m_pdata->getImages().swap(m_image_stage);
    m_pdata->getCharges().swap(m_charge_stage);
    m_pdata->getDiameters().swap(m_diameter_stage);
    m_pdata->getBodies().swap(m_body_stage);
    m_pdata->getOrientationArray().swap(m_orientation_stage);
    m_pdata->getGlobalTags().swap(m_tag_stage);

    m_pdata->removeParticles(n_remove_ptls);

        {
        // update rtag information
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
        gpu_update_rtag(m_pdata->getN(),0, d_global_tag.data, d_global_rtag.data);
        CHECK_CUDA_ERROR();
        }

 
#ifndef NDEBUG
    // check that total particle number is conserved
    unsigned int N;
    reduce(*m_mpi_comm,m_pdata->getN(), N, std::plus<unsigned int>(), 0);
    if (m_mpi_comm->rank() == 0 && N != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Global number of particles has changed unexpectedly (" <<
                N << " != " << m_pdata->getNGlobal() << ")." << endl << endl;
        throw runtime_error("Error in MPI communication.");
        }
#endif

    // notify ParticleData that addition / removal of particles is complete
    m_pdata->notifyParticleSort();

    if (m_prof)
        m_prof->pop();
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    assert(m_r_ghost < (m_pdata->getBox().getL().x));
    assert(m_r_ghost < (m_pdata->getBox().getL().y));
    assert(m_r_ghost < (m_pdata->getBox().getL().z));

    // reset plans
    m_plan.clear();

    // resize plans
    m_plan.resize(m_pdata->getN());

    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    if (bdata->getNumBonds())
        {
        // Send incomplete bond member to the nearest plane in all directions
        const GPUVector<uint2>& btable = bdata->getBondTable();
        ArrayHandle<uint2> d_btable(btable, access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

        gpu_mark_particles_in_incomplete_bonds(d_btable.data,
                                               d_plan.data,
                                               d_pos.data,
                                               d_rtag.data,
                                               m_pdata->getN(),
                                               bdata->getNumBonds(),
                                               m_pdata->getBox());
        CHECK_CUDA_ERROR();
        }


    /*
     * Mark non-bonded atoms for sending
     */
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

        gpu_make_nonbonded_exchange_plan(d_plan.data,
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_pdata->getBox(),
                                         m_r_ghost);
        CHECK_CUDA_ERROR();
        }

    /*
     * Fill send buffers, exchange particles according to plans
     */

    // resize buffers
    m_plan_copybuf.resize(m_pdata->getN());
    m_pos_copybuf.resize(m_pdata->getN());
    m_charge_copybuf.resize(m_pdata->getN());
    m_diameter_copybuf.resize(m_pdata->getN());

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        m_num_copy_ghosts[dir] = 0;

        // resize array of ghost particle tags to copy 
        unsigned int max_copy_ghosts = m_pdata->getN() + m_pdata->getNGhosts();
        m_copy_ghosts[dir].resize(max_copy_ghosts);
        
        // resize buffers
        m_pos_copybuf.resize(max_copy_ghosts);
        m_charge_copybuf.resize(max_copy_ghosts);
        m_diameter_copybuf.resize(max_copy_ghosts);
        m_plan_copybuf.resize(max_copy_ghosts);
        m_tag_copybuf.resize(max_copy_ghosts);


            {
            // Fill send buffer
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tag_copybuf(m_tag_copybuf, access_location::device, access_mode::overwrite);

            gpu_make_exchange_ghost_list(m_pdata->getN()+m_pdata->getNGhosts() ,
                                         m_pdata->getN(),
                                         dir,
                                         d_plan.data,
                                         d_global_tag.data,
                                         d_copy_ghosts.data,
                                         d_tag_copybuf.data,
                                         m_num_copy_ghosts[dir]);
            CHECK_CUDA_ERROR();

            gpu_exchange_ghosts(m_num_copy_ghosts[dir],
                                d_copy_ghosts.data,
                                d_global_rtag.data,
                                d_pos.data,
                                d_pos_copybuf.data,
                                d_charge.data,
                                d_charge_copybuf.data,
                                d_diameter.data,
                                d_diameter_copybuf.data,
                                d_plan.data,
                                d_plan_copybuf.data);
            CHECK_CUDA_ERROR();
            }

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[12];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,m_num_copy_ghosts[dir]);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,m_num_recv_ghosts[dir]);
        boost::mpi::wait_all(reqs,reqs+2);

        if (m_prof)
            m_prof->pop();

        // append ghosts at the end of particle data array
        unsigned int start_idx = m_pdata->getN() + m_pdata->getNGhosts();

        // accommodate new ghost particles
        m_pdata->addGhostParticles(m_num_recv_ghosts[dir]);

        // resize plan array
        m_plan.resize(m_pdata->getN() + m_pdata->getNGhosts());

        // exchange particle data, write directly to the particle data arrays
        if (m_prof)
            m_prof->push("MPI send/recv");

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag_copybuf(m_tag_copybuf, access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,d_plan_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,d_plan.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,d_tag_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,d_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,d_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,d_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,d_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,d_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+12);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag_copybuf(m_tag_copybuf, access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,h_plan_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,h_plan.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,h_tag_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,h_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,h_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,h_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,h_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,h_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+12);
            }
#endif
        if (m_prof)
            m_prof->pop();

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            gpu_wrap_ghost_particles(dir,
                              m_num_recv_ghosts[dir],
                              d_pos.data + start_idx,
                              m_pdata->getGlobalBox(),
                              m_is_at_boundary);
            CHECK_CUDA_ERROR();

            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);

            gpu_update_rtag(m_num_recv_ghosts[dir], start_idx, d_global_tag.data + start_idx, d_global_rtag.data);
            CHECK_CUDA_ERROR();

            }
        }

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

    if (m_prof)
        m_prof->pop();
    }

//! Update ghost particle positions
void CommunicatorGPU::copyGhosts()
    {
    // we have a current m_copy_ghosts list which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("copy_ghosts");

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received


    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        if (! isCommunicating(dir) ) continue;

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);

            gpu_copy_ghosts(m_num_copy_ghosts[dir], d_pos.data, d_copy_ghosts.data, d_pos_copybuf.data);
            CHECK_CUDA_ERROR();
            }

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        unsigned int start_idx;

        if (m_prof)
            m_prof->push("MPI send/recv");


        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        boost::mpi::request reqs[2];
#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#endif

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sizeof(Scalar4));

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            gpu_wrap_ghost_particles(dir,
                           m_num_recv_ghosts[dir],
                           d_pos.data + start_idx,
                           m_pdata->getGlobalBox(),
                           m_is_at_boundary);
            CHECK_CUDA_ERROR();
            }

        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU()
    {
    class_<CommunicatorGPU, bases<Communicator>, boost::shared_ptr<CommunicatorGPU>, boost::noncopyable>("CommunicatorGPU",
           init<boost::shared_ptr<SystemDefinition>,
                boost::shared_ptr<DomainDecomposition> >())
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
