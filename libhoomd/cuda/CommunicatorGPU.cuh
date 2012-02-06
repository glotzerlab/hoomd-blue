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

/*! \file Communicator.cuh
    \brief Defines the GPU functions of the communication algorithms
*/

#ifdef ENABLE_MPI
#include "ParticleData.cuh"

//! Get the size of packed data element
/*! \return the size of the data element (in bytes)
 */
unsigned int gpu_pdata_element_size();

//! Reorder the particle data
void gpu_migrate_compact_pdata(unsigned int N,
                           unsigned int &n_delete_ptls,
                           float4 *d_pos,
                           float4 *d_vel,
                           float3 *d_accel,
                           int3 *d_image,
                           float *d_charge,
                           float *d_diameter,
                           unsigned int *d_body,
                           float4  *d_orientation,
                           unsigned int *d_tag,
                           gpu_boxsize box,
                           bool send_x,
                           bool send_y,
                           bool send_z);

//! Determine particles to be sent in a given direction and fill send buffer
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
                           char *&d_send_buf_end,
                           gpu_boxsize box,
                           unsigned int dir);

//! Select particles to forward in a given direction and pack them into a send buffer
void gpu_migrate_forward_particles(char *d_recv_buf,
                                   char *d_recv_buf_end,
                                   char *d_send_buf,
                                   char *&d_send_buf_end,
                                   gpu_boxsize box,
                                   unsigned int dir);

//! Wrap received particles across global box boundaries
void gpu_migrate_wrap_received_particles(char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 const gpu_boxsize& global_box,
                                 unsigned int dir);

//! Count received particles that are to be added to the local simulation box
void gpu_migrate_count_particles_in_box(unsigned int &num_ptls_in_box,
                                char *d_recv_buf,
                                char *d_recv_buf_end,
                                const gpu_boxsize& box);

//! Add received particles to local box if their positions are inside the local boundaries
void gpu_migrate_move_particles_into_box(unsigned int &num_added_ptls,
                                 char *d_recv_buf,
                                 char *d_recv_buf_end,
                                 float4 *d_pos,
                                 float4 *d_vel,
                                 float3 *d_accel,
                                 int3 *d_image,
                                 float *d_charge,
                                 float *d_diameter,
                                 unsigned int *d_body,
                                 float4  *d_orientation,
                                 unsigned int *d_tag,
                                 const gpu_boxsize &box);


//! Wrap received ghost particles across global box
void gpu_wrap_ghost_particles(unsigned int dir,
                              unsigned int n,
                              float4 *d_pos,
                              gpu_boxsize global_box,
                              float rghost);

//! Construct a list of particle tags to send as ghost particles
void gpu_make_exchange_ghost_list(unsigned int N,
                                  unsigned int dir,
                                  float4 *d_pos,
                                  unsigned int *d_global_tag,
                                  unsigned int* d_copy_ghosts,
                                  unsigned int &n_copy_ghosts,
                                  gpu_boxsize box,
                                  float r_ghost);

//! Fill send buffers of particles we are sending as ghost particles with partial particle data
void gpu_exchange_ghosts(unsigned int nghosts,
                         unsigned int *d_copy_ghosts,
                         unsigned int *d_rtag,
                         float4 *d_pos,
                         float4 *d_pos_copybuf,
                         float *d_charge,
                         float *d_charge_copybuf,
                         float *d_diameter,
                         float *d_diameter_copybuf);

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
