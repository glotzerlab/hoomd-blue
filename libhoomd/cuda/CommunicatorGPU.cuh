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
#include "BondData.cuh"

#include "cached_allocator.h"
#include "Index1D.h"

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

//! List of valid edges of the local simulation box
enum gpu_edge_flags
    {
    edge_east_north = 0 ,
    edge_east_south,
    edge_east_up,
    edge_east_down,
    edge_west_north,
    edge_west_south,
    edge_west_up,
    edge_west_down,
    edge_north_up,
    edge_north_down,
    edge_south_up,
    edge_south_down
    };

//! List of valid faces of the local simulation box
enum gpu_face_flags
    {
    face_east = 0,
    face_west,
    face_north,
    face_south,
    face_up,
    face_down
    };

//! List of valid corners of the local simulation box
enum gpu_corner_flags
    {
    corner_east_north_up = 0,
    corner_east_north_down,
    corner_east_south_up,
    corner_east_south_down,
    corner_west_north_up,
    corner_west_north_down,
    corner_west_south_up,
    corner_west_south_down
    };
#endif

//! Mark particles in incomplete bonds for sending
void gpu_mark_particles_in_incomplete_bonds(const uint2 *d_btable,
                                          unsigned char *d_plan,
                                          const Scalar4 *d_pos,
                                          const unsigned int *d_rtag,
                                          const unsigned int N,
                                          const unsigned int n_bonds,
                                          const BoxDim& box);

void gpu_send_bonds(const unsigned int n_bonds,
                    const unsigned int n_particles,
                    const uint2 *d_bonds,
                    const unsigned int *d_bond_type,
                    const unsigned int *d_bond_tag,
                    const unsigned int *d_rtag,
                    const unsigned int *d_ptl_plan,
                    unsigned int *d_bond_remove_mask,
                    bond_element *d_face_send_buf,
                    unsigned int face_send_buf_pitch,
                    bond_element *d_edge_send_buf,
                    unsigned int edge_send_buf_pitch,
                    bond_element *d_corner_send_buf,
                    unsigned int corner_send_buf_pitch,
                    unsigned int *d_n_send_bonds_face,
                    unsigned int *d_n_send_bonds_edge,
                    unsigned int *d_n_send_bonds_corner,
                    const unsigned int max_send_bonds_face,
                    const unsigned int max_send_bonds_edge,
                    const unsigned int max_send_bonds_corner,
                    unsigned int *d_n_remove_bonds,
                    unsigned int *d_condition);

//! Mark particles that have left the local box for sending
void gpu_stage_particles(const unsigned int n,
                         const Scalar4 *d_pos,
                         const unsigned int *d_tag,
                         unsigned int *d_rtag,
                         const BoxDim& box,
                         const unsigned int comm_mask,
                         cached_allocator& alloc);

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
                   const Index3D& di,
                   const uint3 my_pos,
                   const BoxDim& box,
                   unsigned int *d_keys,
                   unsigned int *d_begin,
                   unsigned int *d_end,
                   const unsigned int *d_neighbors,
                   const unsigned int nneigh,
                   cached_allocator& alloc);

//! Apply boundary conditions
void gpu_wrap_particles(const unsigned int n_recv,
                        pdata_element *d_in,
                        const BoxDim& box);

//! Select bonds for sending
void gpu_select_bonds(unsigned int n_bonds,
                      const uint2 *d_bonds,
                      const unsigned int *d_bond_tag,
                      unsigned int *d_bond_rtag,
                      const unsigned int *d_rtag,
                      cached_allocator& alloc);

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
                                  unsigned int mask,
                                  cached_allocator& alloc);
//! Get neighbor counts
unsigned int gpu_exchange_ghosts_count_neighbors(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_adj,
    unsigned int *d_counts,
    unsigned int nneigh,
    cached_allocator& alloc);

//! Construct tag lists per ghost particle
void gpu_exchange_ghosts_make_indices(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_tag,
    const unsigned int *d_adj,
    const unsigned int *d_neighbors,
    const unsigned int *d_unique_neighbors,
    const unsigned int *d_counts,
    unsigned int *d_ghost_tag,
    unsigned int *d_ghost_begin,
    unsigned int *d_ghost_end,
    unsigned int nneigh,
    unsigned int n_unique_neigh,
    unsigned int n_out,
    unsigned int mask,
    cached_allocator& alloc);

//! Pack ghosts in output buffers
void gpu_exchange_ghosts_pack(
    unsigned int n_out,
    const unsigned int *d_ghost_tag,
    const unsigned int *d_rtag,
    const Scalar4 *d_pos,
    const Scalar4 *d_vel,
    const Scalar *d_charge,
    const Scalar *d_diameter,
    const Scalar4 *d_orientation,
    Scalar4 *d_pos_sendbuf,
    Scalar4 *d_vel_sendbuf,
    Scalar *d_charge_sendbuf,
    Scalar *d_diameter_sendbuf,
    Scalar4 *d_orientation_sendbuf,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_orientation,
    cached_allocator &alloc);

//! Wrap received ghost positions
void gpu_wrap_ghosts(const unsigned int n_recv,
                     Scalar4 *d_pos,
                     const BoxDim& box);

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
    bool send_orientation,
    cached_allocator& alloc);

//! Compute ghost rtags
void gpu_compute_ghost_rtags(unsigned int first_idx,
     unsigned int n_ghost,
     const unsigned int *d_tag,
     unsigned int *d_rtag);
#endif // ENABLE_MPI
