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

/*! \file CommunicatorGPU.cuh
    \brief Defines the GPU functions of the communication algorithms
*/

#ifdef ENABLE_MPI
#include "ParticleData.cuh"
#include "BondedGroupData.cuh"

#include "Index1D.h"

#include "moderngpu/util/mgpucontext.h"

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
                   unsigned int *d_keys,
                   unsigned int *d_begin,
                   unsigned int *d_end,
                   const unsigned int *d_neighbors,
                   const unsigned int nneigh,
                   const unsigned int mask,
                   mgpu::ContextPtr mgpu_context,
                   unsigned int *d_tmp,
                   pdata_element *d_in_copy);

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
                                  const BoxDim& box,
                                  Scalar3 ghost_fraction,
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
    unsigned int *d_ghost_idx,
    unsigned int *d_ghost_neigh,
    unsigned int *d_ghost_begin,
    unsigned int *d_ghost_end,
    unsigned int n_unique_neigh,
    unsigned int n_out,
    unsigned int mask,
    mgpu::ContextPtr mgpu_context);

//! Pack ghosts in output buffers
void gpu_exchange_ghosts_pack(
    unsigned int n_out,
    const unsigned int *d_ghost_idx,
    const unsigned int *d_tag,
    const Scalar4 *d_pos,
    const Scalar4 *d_vel,
    const Scalar *d_charge,
    const Scalar *d_diameter,
    const Scalar4 *d_orientation,
    unsigned int *d_tag_sendbuf,
    Scalar4 *d_pos_sendbuf,
    Scalar4 *d_vel_sendbuf,
    Scalar *d_charge_sendbuf,
    Scalar *d_diameter_sendbuf,
    Scalar4 *d_orientation_sendbuf,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_orientation);

//! Initialize cache configuration
void gpu_communicator_initialize_cache_config();

//! Wrap received ghost positions
void gpu_wrap_ghosts(const unsigned int n_recv,
                     Scalar4 *d_pos,
                     BoxDim box);

//! Copy receive buffers into particle data
void gpu_exchange_ghosts_copy_buf(
    unsigned int n_recv,
    const unsigned int *d_tag_recvbuf,
    const Scalar4 *d_pos_recvbuf,
    const Scalar4 *d_vel_recvbuf,
    const Scalar *d_charge_recvbuf,
    const Scalar *d_diameter_recvbuf,
    const Scalar4 *d_orientation_recvbuf,
    unsigned int *d_tag,
    Scalar4 *d_pos,
    Scalar4 *d_vel,
    Scalar *d_charge,
    Scalar *d_diameter,
    Scalar4 *d_orientation,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
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
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_t *d_out_groups,
    unsigned int *d_out_rank_masks);

template<typename group_t, typename ranks_t>
void gpu_remove_groups(unsigned int n_groups,
    const group_t *d_groups,
    group_t *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
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
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    mgpu::ContextPtr mgpu_context);

template<unsigned int group_size, typename members_t, typename ranks_t>
void gpu_mark_bonded_ghosts(
    unsigned int n_groups,
    members_t *d_groups,
    ranks_t *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    unsigned int mask);

#endif // ENABLE_MPI
