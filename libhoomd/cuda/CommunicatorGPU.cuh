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

//! Get the size of packed data element
/*! \return the size of the data element (in bytes)
 */
unsigned int gpu_pdata_element_size();

//! Allocate temporary device memory for reordering particles
void gpu_allocate_tmp_storage();

//! Dellocate temporary memory
void gpu_deallocate_tmp_storage();

//! Mark particles in incomplete bonds for sending
void gpu_mark_particles_in_incomplete_bonds(const uint2 *d_btable,
                                          unsigned char *d_plan,
                                          const unsigned int *d_rtag,
                                          const unsigned int N,
                                          const unsigned int n_bonds);

//! Reorder the particle data
void gpu_migrate_select_particles(unsigned int N,
                        unsigned int &n_send_ptls,
                        float4 *d_pos,
                        float4 *d_pos_tmp,
                        float4 *d_vel,
                        float4 *d_vel_tmp,
                        float3 *d_accel,
                        float3 *d_accel_tmp,
                        int3 *d_image,
                        int3 *d_image_tmp,
                        float *d_charge,
                        float *d_charge_tmp,
                        float *d_diameter,
                        float *d_diameter_tmp,
                        unsigned int *d_body,
                        unsigned int *d_body_tmp,
                        float4 *d_orientation,
                        float4 *d_orientation_tmp,
                        unsigned int *d_tag,
                        unsigned int *d_tag_tmp,
                        const BoxDim& box,
                        unsigned int dir);

//! Reset reverse lookup tags of particles we are removing
void gpu_reset_rtags(unsigned int n_delete_ptls,
                     unsigned int *d_delete_tags,
                     unsigned int *d_rtag);


//! Pack particle data into send buffer
void gpu_migrate_pack_send_buffer(unsigned int N,
                           float4 *d_pos,
                           float4 *d_vel,
                           float3 *d_accel,
                           int3 *d_image,
                           float *d_charge,
                           float *d_diameter,
                           unsigned int *d_body,
                           float4  *d_orientation,
                           unsigned int *d_tag,
                           char *d_send_buf,
                           char *&d_send_buf_end);

//! Wrap received particles across global box boundaries
void gpu_migrate_wrap_received_particles(char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 unsigned int &n_recv_ptl,
                                 const BoxDim& global_box,
                                 const unsigned int dir,
                                 const bool is_at_boundary[]);

//! Add received particles to local box if their positions are inside the local boundaries
void gpu_migrate_add_particles(  char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 float4 *d_pos,
                                 float4 *d_vel,
                                 float3 *d_accel,
                                 int3 *d_image,
                                 float *d_charge,
                                 float *d_diameter,
                                 unsigned int *d_body,
                                 float4  *d_orientation,
                                 unsigned int *d_tag);

//! Filter received ghost particles (only accept particles that are not local or ghosts) (step one)
void gpu_filter_ghost_particles_step_one(unsigned int *d_tag_recvbuf,
                                         unsigned int *d_rtag,
                                         unsigned char *d_add_ghost,
                                         const unsigned int n_recv_ghosts,
                                         unsigned int& n_marked_particles);

//! Filter received ghost particles (only accept particles that are not local or ghosts) (step two)
void gpu_filter_ghost_particles_step_two(unsigned char *d_plan,
                                Scalar4 *d_pos,
                                Scalar *d_charge,
                                Scalar *d_diameter,
                                unsigned int *d_tag,
                                unsigned char *d_plan_recvbuf,
                                Scalar4 *d_pos_recvbuf,
                                Scalar *d_charge_recvbuf,
                                Scalar *d_diameter_recvbuf,
                                unsigned int *d_tag_recvbuf,
                                unsigned char *d_add_ghost,
                                unsigned int n_recv_ghosts
                                );

//! Filter received ghost particles when copying
void gpu_filter_ghost_particles_copy(Scalar4 *d_pos,
                                Scalar4 *d_pos_recvbuf,
                                unsigned char *d_add_ghost,
                                unsigned int n_recv_ghosts,
                                unsigned int& n_added_ptls
                                );
 
//! Wrap received ghost particles across global box
void gpu_wrap_ghost_particles(unsigned int dir,
                              unsigned int n,
                              float4 *d_pos,
                              const BoxDim& global_box,
                              const bool is_at_boundary[]);

//! Construct plans for sending non-bonded ghost particles
void gpu_make_nonbonded_exchange_plan(unsigned char *d_plan,
                                      unsigned int N,
                                      float4 *d_pos,
                                      const BoxDim& box,
                                      float r_ghost);

//! Construct a list of particle tags to send as ghost particles
void gpu_make_exchange_ghost_list(unsigned int n_total,
                                  unsigned int N,
                                  unsigned int dir,
                                  unsigned char *d_plan,
                                  unsigned int *d_global_tag,
                                  unsigned int* d_copy_ghosts,
                                  unsigned int &n_copy_ghosts);

//! Fill send buffers of particles we are sending as ghost particles with partial particle data
void gpu_exchange_ghosts(unsigned int nghosts,
                         unsigned int *d_copy_ghosts,
                         unsigned int *d_rtag,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_diameter,
                         float *d_diameter_copybuf,
                         unsigned char *d_plan,
                         unsigned char *d_plan_copybuf);

//! Update global tag <-> local particle index reverse lookup array
void gpu_update_rtag(unsigned int nptl,
                     unsigned int start_idx,
                     unsigned int *d_tag,
                     unsigned int *d_rtag);


//! Copy ghost particle positions into send buffer
void gpu_copy_ghosts(unsigned int nghost,
                     float4 *d_pos,
                     unsigned int *d_copy_ghosts,
                     float4 *d_pos_copybuf,
                     unsigned int *rtag);

#endif // ENABLE_MPI
