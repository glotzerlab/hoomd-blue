// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file CommunicatorGPU.cuh
    \brief Defines the GPU functions of the communication algorithms
*/

#ifdef ENABLE_MPI
#include "ParticleData.cuh"
#include "BondedGroupData.cuh"

#include "Index1D.h"

#include "hoomd/extern/util/mgpucontext.h"
#include "hoomd/CachedAllocator.h"

#ifdef NVCC
//! The flags used for indicating the itinerary of a particle
enum gpu_send_flags
    {
    send_east = 1,
    send_west = 2,
    send_north = 4,
    send_south = 8,
    send_up = 16,
    send_down = 32
    };

template<typename ranks_t>
struct rank_element
    {
    ranks_t ranks;
    unsigned int mask;
    unsigned int tag;
    };
#else
template<typename ranks_t>
struct rank_element;
#endif

//! Mark particles that have left the local box for sending
void gpu_stage_particles(const unsigned int n,
                         const Scalar4 *d_pos,
                         unsigned int *d_comm_flag,
                         const BoxDim& box,
                         const unsigned int comm_mask);

/*! \param nsend Number of particles in buffer
    \param d_in Send buf (in-place sort)
    \param di Domain indexer
    \param box Local box
    \param d_keys Output array (target domains)
    \param d_begin Output array (start indices per key in send buf)
    \param d_end Output array (end indices per key in send buf)
    \param d_neighbors List of neighbor ranks
    \param alloc Caching allocator
 */
void gpu_sort_migrating_particles(const unsigned int nsend,
                   pdata_element *d_in,
                   const unsigned int *d_comm_flags,
                   const Index3D& di,
                   const uint3 my_pos,
                   const unsigned int *d_cart_ranks,
                   unsigned int *d_keys,
                   unsigned int *d_begin,
                   unsigned int *d_end,
                   const unsigned int *d_neighbors,
                   const unsigned int nneigh,
                   const unsigned int mask,
                   mgpu::ContextPtr mgpu_context,
                   unsigned int *d_tmp,
                   pdata_element *d_in_copy,
                   CachedAllocator& alloc);

//! Apply boundary conditions
void gpu_wrap_particles(const unsigned int n_recv,
                        pdata_element *d_in,
                        const BoxDim& box);

//! Reset reverse lookup tags of particles we are removing
void gpu_reset_rtags(unsigned int n_delete_ptls,
                     unsigned int *d_delete_tags,
                     unsigned int *d_rtag);

//! Construct plans for sending non-bonded ghost particles
void gpu_make_ghost_exchange_plan(unsigned int *d_plan,
                                  unsigned int N,
                                  const Scalar4 *d_pos,
                                  const unsigned int *d_body,
                                  const BoxDim& box,
                                  const Scalar *d_r_ghost,
                                  const Scalar *d_r_ghost_body,
                                  Scalar r_ghost_max,
                                  unsigned int ntypes,
                                  unsigned int mask);

//! Get neighbor counts
unsigned int gpu_exchange_ghosts_count_neighbors(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_adj,
    unsigned int *d_counts,
    unsigned int nneigh,
    mgpu::ContextPtr mgpu_context);

//! Construct tag lists per ghost particle
void gpu_exchange_ghosts_make_indices(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_tag,
    const unsigned int *d_adj,
    const unsigned int *d_unique_neighbors,
    const unsigned int *d_counts,
    uint2 *d_ghost_idx,
    unsigned int *d_ghost_neigh,
    unsigned int *d_ghost_begin,
    unsigned int *d_ghost_end,
    unsigned int n_unique_neigh,
    unsigned int n_out,
    unsigned int mask,
    mgpu::ContextPtr mgpu_context,
    CachedAllocator& alloc);

//! Pack ghosts in output buffers
void gpu_exchange_ghosts_pack(
    unsigned int n_out,
    const uint2 *d_ghost_idx_adj,
    const unsigned int *d_tag,
    const Scalar4 *d_pos,
    const int3* d_img,
    const Scalar4 *d_vel,
    const Scalar *d_charge,
    const Scalar *d_diameter,
    const unsigned int *d_body,
    const Scalar4 *d_orientation,
    unsigned int *d_tag_sendbuf,
    Scalar4 *d_pos_sendbuf,
    Scalar4 *d_vel_sendbuf,
    Scalar *d_charge_sendbuf,
    Scalar *d_diameter_sendbuf,
    unsigned int *d_body_sendbuf,
    int3 *d_img_sendbuf,
    Scalar4 *d_orientation_sendbuf,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_body,
    bool send_image,
    bool send_orientation,
    const Index3D& di,
    uint3 my_pos,
    const BoxDim& box);

//! Initialize cache configuration
void gpu_communicator_initialize_cache_config();


//! Copy receive buffers into particle data
void gpu_exchange_ghosts_copy_buf(
    unsigned int n_recv,
    const unsigned int *d_tag_recvbuf,
    const Scalar4 *d_pos_recvbuf,
    const Scalar4 *d_vel_recvbuf,
    const Scalar *d_charge_recvbuf,
    const Scalar *d_diameter_recvbuf,
    const unsigned int *d_body_recvbuf,
    const int3 *d_image_recvbuf,
    const Scalar4 *d_orientation_recvbuf,
    unsigned int *d_tag,
    Scalar4 *d_pos,
    Scalar4 *d_vel,
    Scalar *d_charge,
    Scalar *d_diameter,
    unsigned int *d_body,
    int3 *d_image,
    Scalar4 *d_orientation,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_body,
    bool send_image,
    bool send_orientation);

//! Compute ghost rtags
void gpu_compute_ghost_rtags(unsigned int first_idx,
     unsigned int n_ghost,
     const unsigned int *d_tag,
     unsigned int *d_rtag);

//! Reset ghost plans
void gpu_reset_exchange_plan(
    unsigned int N,
    unsigned int *d_plan);

//! Mark groups for sending
template<unsigned int group_size, typename group_t, typename ranks_t>
void gpu_mark_groups(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_t *d_members,
    ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    unsigned int *d_scan,
    unsigned int &n_out,
    const Index3D di,
    uint3 my_pos,
    const unsigned int *d_cart_ranks,
    bool incomplete,
    mgpu::ContextPtr mgpu_context);

//! Compact rank information for groups that have been marked for sending
template<unsigned int group_size, typename group_t, typename ranks_t, typename rank_element_t>
void gpu_scatter_ranks_and_mark_send_groups(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_t *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    unsigned int &n_send,
    rank_element_t *d_out_ranks,
    mgpu::ContextPtr mgpu_context);

template<unsigned int group_size, typename ranks_t, typename rank_element_t>
void gpu_update_ranks_table(
    unsigned int n_groups,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element_t *d_ranks_recvbuf);

template<unsigned int group_size, typename group_t, typename ranks_t, typename packed_t>
void gpu_scatter_and_mark_groups_for_removal(
    unsigned int n_groups,
    const group_t *d_groups,
    const typeval_union *d_group_typeval,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_t *d_out_groups,
    unsigned int *d_out_rank_masks,
    bool local_multiple);

template<typename group_t, typename ranks_t>
void gpu_remove_groups(unsigned int n_groups,
    const group_t *d_groups,
    group_t *d_groups_alt,
    const typeval_union *d_group_typeval,
    typeval_union *d_group_typeval_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const ranks_t *d_group_ranks,
    ranks_t *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_scan,
    mgpu::ContextPtr mgpu_context);

template<typename packed_t, typename group_t, typename ranks_t>
void gpu_add_groups(unsigned int n_groups,
    unsigned int n_recv,
    const packed_t *d_groups_in,
    group_t *d_groups,
    typeval_union *d_group_typeval,
    unsigned int *d_group_tag,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    bool local_multiple,
    unsigned int myrank,
    mgpu::ContextPtr mgpu_context);

template<unsigned int group_size, typename members_t, typename ranks_t>
void gpu_mark_bonded_ghosts(
    unsigned int n_groups,
    members_t *d_groups,
    ranks_t *d_ranks,
    const Scalar4 *d_postype,
    const BoxDim& box,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    const unsigned int *d_cart_ranks_inv,
    unsigned int my_rank,
    unsigned int mask);

template<unsigned int group_size, typename members_t>
void gpu_make_ghost_group_exchange_plan(unsigned int *d_ghost_group_plan,
                                   const members_t *d_groups,
                                   unsigned int N,
                                   const unsigned int *d_rtag,
                                   const unsigned int *d_plans,
                                   unsigned int n_local);

template<class members_t, class ranks_t, class group_element_t>
void gpu_exchange_ghost_groups_pack(
    unsigned int n_out,
    const uint2 *d_ghost_idx_adj,
    const unsigned int *d_group_tag,
    const members_t *d_groups,
    const typeval_union *d_group_typeval,
    const ranks_t *d_group_ranks,
    group_element_t *d_groups_sendbuf);

template<unsigned int size, class members_t, class ranks_t, class group_element_t>
void gpu_exchange_ghost_groups_copy_buf(
    unsigned int nrecv,
    const group_element_t *d_groups_recvbuf,
    unsigned int *d_group_tag,
    members_t *d_groups,
    typeval_union *d_group_typeval,
    ranks_t *d_group_ranks,
    unsigned int *d_keep,
    unsigned int *d_scan,
    const unsigned int *d_group_rtag,
    const unsigned int *d_rtag,
    unsigned int max_n_local,
    unsigned int &n_keep,
    mgpu::ContextPtr mgpu_context);

void gpu_exchange_ghosts_pack_netforce(
    unsigned int n_out,
    const uint2 *d_ghost_idx_adj,
    const Scalar4 *d_netforce,
    Scalar4 *d_netforce_sendbuf);

void gpu_exchange_ghosts_copy_netforce_buf(
    unsigned int n_recv,
    const Scalar4 *d_netforce_recvbuf,
    Scalar4 *d_netforce);

void gpu_exchange_ghosts_pack_netvirial(
    unsigned int n_out,
    const uint2 *d_ghost_idx_adj,
    const Scalar *d_netvirial,
    Scalar *d_netvirial_sendbuf,
    unsigned int pitch_in);

void gpu_exchange_ghosts_copy_netvirial_buf(
    unsigned int n_recv,
    const Scalar *d_netvirial_recvbuf,
    Scalar *d_netvirial,
    unsigned int pitch_out);
#endif // ENABLE_MPI
