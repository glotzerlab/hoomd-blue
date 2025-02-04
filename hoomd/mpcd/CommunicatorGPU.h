// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CommunicatorGPU.h
 * \brief Defines the mpcd::CommunicatorGPU class
 */

#ifndef MPCD_COMMUNICATOR_GPU_H_
#define MPCD_COMMUNICATOR_GPU_H_

#ifdef ENABLE_MPI
#ifdef ENABLE_HIP

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "Communicator.h"

#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/GPUVector.h"
#include "hoomd/SystemDefinition.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! MPI communication of MPCD particle data on the GPU
/*!
 * This class implements the communication algorithms for mpcd::ParticleData that
 * are used in parallel simulations on the GPU. A domain decomposition communication pattern
 * is used so that every processor owns particles that are spatially local (\cite Plimpton 1995). So
 * far, the only communication needed for MPCD particles is migration, which is handled
 * using the same algorithms as for the standard hoomd::ParticleData
 * (::CommunicatorGPU::migrateParticles).
 *
 * There is unfortunately significant code duplication with ::CommunicatorGPU, but
 * there is little that can be done about this without creating an abstracted
 * communication base class for the GPU.
 *
 * \ingroup communication
 */
class PYBIND11_EXPORT CommunicatorGPU : public mpcd::Communicator
    {
    public:
    //! Constructor
    CommunicatorGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~CommunicatorGPU();

    //! \name communication methods
    //@{

    //! Migrate particle data to local domain
    virtual void migrateParticles(uint64_t timestep);
    //@}

    //! Set maximum number of communication stages
    /*! \param max_stages Maximum number of communication stages
     */
    void setMaxStages(unsigned int max_stages)
        {
        m_max_stages = max_stages;
        initializeCommunicationStages();
        }

    protected:
    //! Set the communication flags for the particle data on the GPU
    virtual void setCommFlags(const BoxDim& box);

    private:
    /* General communication */
    unsigned int m_max_stages;             //!< Maximum number of (dependent) communication stages
    unsigned int m_num_stages;             //!< Number of stages
    std::vector<unsigned int> m_comm_mask; //!< Communication mask per stage
    std::vector<int> m_stages;             //!< Communication stage per unique neighbor

    /* Particle migration */
    GPUArray<unsigned int> m_neigh_send;     //!< Neighbor rank indexes for sending
    GPUArray<unsigned int> m_num_send;       //!< Number of particles to send to each rank
    GPUVector<unsigned int> m_tmp_keys;      //!< Temporary keys for sorting particles
    std::vector<unsigned int> m_n_send_ptls; //!< Number of particles sent per neighbor
    std::vector<unsigned int> m_n_recv_ptls; //!< Number of particles received per neighbor
    std::vector<unsigned int> m_offsets;     //!< Offsets for particle send buffers

    //! Helper function to set up communication stages
    void initializeCommunicationStages();

    /* Autotuners */
    std::shared_ptr<Autotuner<1>> m_flags_tuner; //!< Tuner for marking communication flags
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // ENABLE_HIP
#endif // ENABLE_MPI
#endif // MPCD_COMMUNICATOR_GPU_H_
