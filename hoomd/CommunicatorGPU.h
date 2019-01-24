// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file CommunicatorGPU.h
    \brief Defines the CommunicatorGPU class
*/

#ifndef __COMMUNICATOR_GPU_H__
#define __COMMUNICATOR_GPU_H__

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "Communicator.h"
#include "Autotuner.h"

#include "CommunicatorGPU.cuh"

#include "GPUFlags.h"
#include "GPUVector.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif


/*! \ingroup communication
*/

//! Class that handles MPI communication (GPU version)
/*! CommunicatorGPU is the GPU implementation of the base communication class.
*/
class PYBIND11_EXPORT CommunicatorGPU : public Communicator
    {
    public:
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param decomposition Information about the decomposition of the global simulation domain
         */
        CommunicatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<DomainDecomposition> decomposition);
        virtual ~CommunicatorGPU();

        //! \name communication methods
        //@{

        /*! Perform ghosts update
         *
         * \param timestep The time step
         */
        virtual void beginUpdateGhosts(unsigned int timestep);

        /*! Finish ghost update
         *
         * \param timestep The time step
         */
        virtual void finishUpdateGhosts(unsigned int timestep);

        //! Transfer particles between neighboring domains
        virtual void migrateParticles();

        //! Build a ghost particle list, exchange ghost particle data with neighboring processors
        virtual void exchangeGhosts();

        /*! Communicate the net particle force
         * \parm timestep The time step
         */
        virtual void updateNetForce(unsigned int timestep);
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

    protected:
        //! Helper class to perform the communication tasks related to bonded groups
        template<class group_data>
        class GroupCommunicatorGPU
            {
            public:
                typedef struct rank_element<typename group_data::ranks_t> rank_element_t;
                typedef typename group_data::packed_t group_element_t;

                //! Constructor
                GroupCommunicatorGPU(CommunicatorGPU& gpu_comm, std::shared_ptr<group_data> gdata);

                //! Migrate groups
                /*! \param incomplete If true, mark all groups that have non-local members and update local
                 *         member rank information. Otherwise, mark only groups flagged for communication
                 *         in particle data
                 *  \param local_multiple If true, a group may be split across several ranks
                 * A group is marked for sending by setting its rtag to GROUP_NOT_LOCAL, and by updating
                 * the rank information with the destination ranks (or the local ranks if incomplete=true)
                 */
                void migrateGroups(bool incomplete, bool local_multiple);

                //! Mark ghost particles
                /* All particles that need to be sent as ghosts because they are members
                 * of incomplete groups are marked, and destination ranks are compute accordingly.
                 *
                 * \param plans Array of particle plans to write to
                 * \param mask Mask for allowed sending directions
                 */

                void markGhostParticles(const GlobalVector<unsigned int>& plans, unsigned int mask);

                //! Copy 'ghost groups' between domains
                /*! Both members of a ghost group are inside the ghost layer
                 *
                 * \param plans The ghost particle send directions determined by Communicator
                 */
                void exchangeGhostGroups(const GlobalVector<unsigned int>& plans);

            private:
                CommunicatorGPU& m_gpu_comm;                            //!< The outer class
                std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //< The execution configuration
                std::shared_ptr<group_data> m_gdata;                  //!< The group data

                GlobalVector<unsigned int> m_rank_mask;                    //!< Bitfield for every group to keep track of updated rank fields
                GlobalVector<unsigned int> m_scan;                         //!< Temporary array for exclusive scan of group membership information

                GlobalVector<rank_element_t> m_ranks_out;                  //!< Packed ranks data
                GlobalVector<rank_element_t> m_ranks_sendbuf;              //!< Send buffer for ranks information
                GlobalVector<rank_element_t> m_ranks_recvbuf;              //!< Recv buffer for ranks information

                GlobalVector<group_element_t> m_groups_out;                //!< Packed group data
                GlobalVector<unsigned int> m_rank_mask_out;                //!< Output buffer for rank update bitfields
                GlobalVector<group_element_t> m_groups_sendbuf;            //!< Send buffer for groups
                GlobalVector<group_element_t> m_groups_recvbuf;            //!< Recv buffer for groups
                GlobalVector<group_element_t> m_groups_in;                 //!< Input buffer of unique groups

                GlobalVector<unsigned int> m_ghost_group_begin;            //!< Begin index for every stage and neighbor in send buf
                GlobalVector<unsigned int> m_ghost_group_end;              //!< Begin index for every and neighbor in send buf

                GlobalVector<uint2> m_ghost_group_idx_adj;                 //!< Indices and adjacency relationships of ghosts to send
                GlobalVector<unsigned int> m_ghost_group_neigh;            //!< Neighbor ranks for every ghost group
                GlobalVector<unsigned int> m_ghost_group_plan;             //!< Plans for every particle
                GlobalVector<unsigned int> m_neigh_counts;                 //!< List of number of neighbors to send ghost to (temp array)
            };

        //! Remove tags of ghost particles
        virtual void removeGhostParticleTags();

    private:
        /* General communication */
        unsigned int m_max_stages;                     //!< Maximum number of (dependent) communication stages
        unsigned int m_num_stages;                     //!< Number of stages
        std::vector<unsigned int> m_comm_mask;         //!< Communication mask per stage
        std::vector<int> m_stages;                     //!< Communication stage per unique neighbor

        /* Particle migration */
        GlobalVector<pdata_element> m_gpu_sendbuf;        //!< Send buffer for particle data
        GlobalVector<pdata_element> m_gpu_recvbuf;        //!< Receive buffer for particle data
        GlobalVector<unsigned int> m_comm_flags;          //!< Output buffer for communication flags

        GlobalVector<unsigned int> m_send_keys;           //!< Destination rank for particles

        /* Communication of bonded groups */
        GroupCommunicatorGPU<BondData> m_bond_comm;    //!< Communication helper for bonds
        friend class GroupCommunicatorGPU<BondData>;

        GroupCommunicatorGPU<AngleData> m_angle_comm;  //!< Communication helper for angles
        friend class GroupCommunicatorGPU<AngleData>;

        GroupCommunicatorGPU<DihedralData> m_dihedral_comm;  //!< Communication helper for dihedrals
        friend class GroupCommunicatorGPU<DihedralData>;

        GroupCommunicatorGPU<ImproperData> m_improper_comm;  //!< Communication helper for impropers
        friend class GroupCommunicatorGPU<ImproperData>;

        GroupCommunicatorGPU<ConstraintData> m_constraint_comm;  //!< Communication helper for constraints
        friend class GroupCommunicatorGPU<ConstraintData>;

        GroupCommunicatorGPU<PairData> m_pair_comm;    //!< Communication helper for pairs
        friend class GroupCommunicatorGPU<PairData>;

        /* Ghost communication */
        GlobalVector<unsigned int> m_tag_ghost_sendbuf;   //!< Buffer for sending particle tags
        GlobalVector<unsigned int> m_tag_ghost_recvbuf;   //!< Buffer for receiving particle tags

        GlobalVector<Scalar4> m_pos_ghost_sendbuf;        //<! Buffer for sending ghost positions
        GlobalVector<Scalar4> m_pos_ghost_recvbuf;        //<! Buffer for receiving ghost positions

        GlobalVector<Scalar4> m_vel_ghost_sendbuf;        //<! Buffer for sending ghost velocities
        GlobalVector<Scalar4> m_vel_ghost_recvbuf;        //<! Buffer for receiving ghost velocities

        GlobalVector<Scalar> m_charge_ghost_sendbuf;      //!< Buffer for sending ghost charges
        GlobalVector<Scalar> m_charge_ghost_recvbuf;      //!< Buffer for sending ghost charges

        GlobalVector<Scalar> m_diameter_ghost_sendbuf;    //!< Buffer for sending ghost charges
        GlobalVector<Scalar> m_diameter_ghost_recvbuf;    //!< Buffer for sending ghost charges

        GlobalVector<unsigned int> m_body_ghost_sendbuf;      //!< Buffer for sending ghost bodys
        GlobalVector<unsigned int> m_body_ghost_recvbuf;      //!< Buffer for sending ghost bodys

        GlobalVector<int3> m_image_ghost_sendbuf;      //!< Buffer for sending ghost images
        GlobalVector<int3> m_image_ghost_recvbuf;      //!< Buffer for sending ghost images

        GlobalVector<Scalar4> m_orientation_ghost_sendbuf;//<! Buffer for sending ghost orientations
        GlobalVector<Scalar4> m_orientation_ghost_recvbuf;//<! Buffer for receiving ghost orientations

        GlobalVector<Scalar4> m_netforce_ghost_sendbuf;    //!< Send buffer for netforce
        GlobalVector<Scalar4> m_netforce_ghost_recvbuf;    //!< Recv buffer for netforce

        GlobalVector<Scalar4> m_nettorque_ghost_sendbuf;    //!< Send buffer for nettorque
        GlobalVector<Scalar4> m_nettorque_ghost_recvbuf;    //!< Recv buffer for nettorque

        GlobalVector<Scalar> m_netvirial_ghost_sendbuf;    //!< Send buffer for netvirial
        GlobalVector<Scalar> m_netvirial_ghost_recvbuf;    //!< Recv buffer for netvirial

        GlobalVector<unsigned int> m_ghost_begin;          //!< Begin index for every stage and neighbor in send buf
        GlobalVector<unsigned int> m_ghost_end;            //!< Begin index for every and neighbor in send buf

        GlobalVector<uint2> m_ghost_idx_adj;             //!< Indices and adjacency relationships of ghosts to send
        GlobalVector<unsigned int> m_ghost_neigh;        //!< Neighbor ranks for every ghost particle
        GlobalVector<unsigned int> m_ghost_plan;         //!< Plans for every particle
        std::vector<unsigned int> m_idx_offs;         //!< Per-stage offset into ghost idx list

        GlobalVector<unsigned int> m_neigh_counts;       //!< List of number of neighbors to send ghost to (temp array)

        std::vector<std::vector<unsigned int> > m_n_send_ghosts; //!< Number of ghosts to send per stage and neighbor
        std::vector<std::vector<unsigned int> > m_n_recv_ghosts; //!< Number of ghosts to receive per stage and neighbor
        std::vector<std::vector<unsigned int> > m_ghost_offs;    //!< Begin of offset in recv buf per stage and neighbor

        std::vector<unsigned int> m_n_send_ghosts_tot; //!< Total number of sent ghosts per stage
        std::vector<unsigned int> m_n_recv_ghosts_tot; //!< Total number of received ghosts per stage

        mgpu::ContextPtr m_mgpu_context;              //!< MGPU context
        cudaEvent_t m_event;                          //!< CUDA event for synchronization

        //! Helper function to allocate various buffers
        void allocateBuffers();

        //! Helper function to set up communication stages
        void initializeCommunicationStages();
    };

//! Export CommunicatorGPU class to python
void export_CommunicatorGPU(pybind11::module& m);

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GPU_H
