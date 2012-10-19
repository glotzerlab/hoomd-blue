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

//! The flags used for indicating the itinerary of a ghost particle
enum gpu_send_flags
    {
    send_east = 1,
    send_west = 2,
    send_north = 4,
    send_south = 8,
    send_up = 16,
    send_down = 32
    };

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

enum gpu_face_flags
    {
    face_east = 0,
    face_west,
    face_north,
    face_south,
    face_up,
    face_down
    };

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

struct pdata_element_gpu
    {
    Scalar4 pos;               //!< Position
    Scalar4 vel;               //!< Velocity
    Scalar3 accel;             //!< Acceleration
    Scalar charge;             //!< Charge
    Scalar diameter;           //!< Diameter
    int3 image;                //!< Image
    unsigned int body;         //!< Body id
    Scalar4 orientation;       //!< Orientation
    unsigned int tag;          //!< global tag
    };

unsigned int gpu_pdata_element_size();

struct ghost_element_gpu
    {
    Scalar4 pos;               //!< Position
    Scalar4 vel;               //!< Velocity
    Scalar charge;             //!< Charge
    Scalar diameter;           //!< Diameter
    unsigned int tag;          //!< global tag
    unsigned int plan;         //!< Ghost sending plan
    };

unsigned int gpu_ghost_element_size();

struct update_element_gpu
    {
    Scalar4 pos;               //!< Position
    };

unsigned int gpu_update_element_size();

//! Allocate temporary device memory for reordering particles
void gpu_allocate_tmp_storage();

//! Dellocate temporary memory
void gpu_deallocate_tmp_storage();

//! Mark particles in incomplete bonds for sending
void gpu_mark_particles_in_incomplete_bonds(const uint2 *d_btable,
                                          unsigned char *d_plan,
                                          const float4 *d_pos,
                                          const unsigned int *d_rtag,
                                          const unsigned int N,
                                          const unsigned int n_bonds,
                                          const BoxDim box);

//! Reorder the particle data
void gpu_migrate_select_particles(unsigned int N,
                                  const Scalar4 *d_pos,
                                  const Scalar4 *d_vel,
                                  const Scalar3 *d_accel,
                                  const int3 *d_image,
                                  const Scalar *d_charge,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_body,
                                  const Scalar4 *d_orientation,
                                  const unsigned int *d_tag,
                                  unsigned int *d_rtag,
                                  unsigned int *n_send_ptls_corner,
                                  unsigned int *n_send_ptls_edge,
                                  unsigned int *n_send_ptls_face,
                                  unsigned int &n_remove_ptls,
                                  unsigned n_max_send_ptls_corner,
                                  unsigned n_max_send_ptls_edge,
                                  unsigned n_max_send_ptls_face,
                                  unsigned char *d_remove_mask,
                                  char *d_corner_buf,
                                  unsigned int corner_buf_pitch,
                                  char *d_edge_buf,
                                  unsigned int edge_buf_pitch,
                                  char *d_face_buf,
                                  unsigned int face_buf_pitch,
                                  const BoxDim& box,
                                  const BoxDim& global_box,
                                  const unsigned int *is_at_boundary,
                                  unsigned int *d_condition);
 
void gpu_migrate_fill_particle_arrays(unsigned int old_nparticles,
                        unsigned int n_recv_ptls,
                        unsigned int n_remove_ptls,
                        unsigned char *d_remove_mask,
                        char *d_recv_buf,
                        float4 *d_pos,
                        float4 *d_vel,
                        float3 *d_accel,
                        int3 *d_image,
                        float *d_charge,
                        float *d_diameter,
                        unsigned int *d_body,
                        float4 *d_orientation,
                        unsigned int *d_tag,
                        unsigned int *d_rtag);
 
//! Reset reverse lookup tags of particles we are removing
void gpu_reset_rtags(unsigned int n_delete_ptls,
                     unsigned int *d_delete_tags,
                     unsigned int *d_rtag);


//! Construct plans for sending non-bonded ghost particles
void gpu_make_nonbonded_exchange_plan(unsigned char *d_plan,
                                      unsigned int N,
                                      float4 *d_pos,
                                      const BoxDim& box,
                                      float r_ghost);

//! Construct a list of particle tags to send as ghost particles
void gpu_exchange_ghosts(const unsigned int N,
                         const unsigned char *d_plan,
                         const unsigned int *d_tag,
                         unsigned int *d_ghost_idx_face,
                         unsigned int ghost_idx_face_pitch,
                         unsigned int *d_ghost_idx_edge,
                         unsigned int ghost_idx_edge_pitch,
                         unsigned int *d_ghost_idx_corner,
                         unsigned int ghost_idx_corner_pitch,
                         float4 *d_pos,
                         float *d_charge,
                         float *d_diameter,
                         char *d_ghost_corner_buf,
                         unsigned int corner_buf_pitch,
                         char *d_ghost_edge_buf,
                         unsigned int edge_buf_pitch,
                         char *d_ghost_face_buf,
                         unsigned int face_buf_pitch,
                         unsigned int *n_copy_ghosts_corner,
                         unsigned int *n_copy_ghosts_edge,
                         unsigned int *n_copy_ghosts_face,
                         unsigned int max_copy_ghosts_corner,
                         unsigned int max_copy_ghosts_edge,
                         unsigned int max_copy_ghosts_face,
                         const unsigned int *is_at_boundary,
                         unsigned int *d_condition);

void gpu_update_ghosts_pack(const unsigned int n_copy_ghosts,
                                     const unsigned int *d_ghost_idx_face,
                                     const unsigned int ghost_idx_face_pitch,
                                     const unsigned int *d_ghost_idx_edge,
                                     const unsigned int ghost_idx_edge_pitch,
                                     const unsigned int *d_ghost_idx_corner,
                                     const unsigned int ghost_idx_corner_pitch,
                                     const float4 *d_pos,
                                     char *d_update_corner_buf,
                                     unsigned int corner_buf_pitch,
                                     char *d_update_edge_buf,
                                     unsigned int edge_buf_pitch,
                                     char *d_update_face_buf,
                                     unsigned int face_buf_pitch,
                                     const unsigned int *n_copy_ghosts_corner,
                                     const unsigned int *n_copy_ghosts_edge,
                                     const unsigned int *n_copy_ghosts_face,
                                     cudaStream_t stream);

void gpu_exchange_ghosts_unpack(unsigned int N,
                                unsigned int n_tot_recv_ghosts,
                                const unsigned int *n_local_ghosts_face,
                                const unsigned int *n_local_ghosts_edge,
                                const unsigned int *n_tot_forward_ghosts_face,
                                const unsigned int *n_tot_forward_ghosts_edge,
                                const unsigned int n_tot_recv_ghosts_local,
                                const unsigned int *n_recv_ghosts_local,
                                const unsigned int *n_forward_ghosts_face,
                                const unsigned int *n_forward_ghosts_edge,
                                const char *d_face_ghosts,
                                const unsigned int face_pitch,
                                const char *d_edge_ghosts,
                                const unsigned int edge_pitch,
                                const char *d_recv_ghosts,
                                Scalar4 *d_pos,
                                Scalar *d_charge,
                                Scalar *d_diameter,
                                unsigned int *d_tag,
                                unsigned int *d_rtag,
                                unsigned int *d_ghost_plan,
                                unsigned int *is_at_boundary,
                                const BoxDim& global_box);

void gpu_update_ghosts_unpack(unsigned int N,
                                unsigned int n_tot_recv_ghosts,
                                const unsigned int *n_local_ghosts_face,
                                const unsigned int *n_local_ghosts_edge,
                                const unsigned int *n_tot_forward_ghosts_face,
                                const unsigned int *n_tot_forward_ghosts_edge,
                                const unsigned int n_tot_recv_ghosts_local,
                                const unsigned int *n_recv_ghosts_local,
                                const unsigned int *n_forward_ghosts_face,
                                const unsigned int *n_forward_ghosts_edge,
                                const char *d_face_ghosts,
                                const unsigned int face_pitch,
                                const char *d_edge_ghosts,
                                const unsigned int edge_pitch,
                                const char *d_recv_ghosts,
                                Scalar4 *d_pos,
                                unsigned int *d_ghost_plan,
                                const unsigned int *is_at_boundary,
                                const BoxDim& global_box,
                                cudaStream_t stream);
 
#endif // ENABLE_MPI
