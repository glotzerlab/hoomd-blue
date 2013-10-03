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


// Maintainer: ndtrung

#include "RigidData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file RigidData.cu

*/


#ifdef NVCC
//! Kernel for seting R and V of rigid body particles
/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param pdata_orientation Particle orientation
    \param d_pgroup_idx Particle index
    \param n_pgroup Number of particles in the group
    \param d_particle_offset Local index of a particle in the body
    \param d_particle_body Body index of a particle
    \param d_rigid_orientation Body orientation (quaternion)
    \param d_rigid_com Body center of mass
    \param d_rigid_vel Body velocity
    \param d_rigid_angvel Body angular velocity
    \param d_rigid_image Body image 
    \param d_rigid_particle_dis Position of a particle in the body frame
    \param d_rigid_particle_orientation Orientation of a particle in the body frame
    \param nmax Maximum number of particles per body
    \param box Box dimensions for periodic boundary condition handling
*/
template<bool set_x>
__global__ void gpu_rigid_setRV_kernel(Scalar4* pdata_pos,
                                       Scalar4* pdata_vel,
                                       int3* pdata_image,
                                       Scalar4* pdata_orientation,
                                       unsigned int *d_pgroup_idx,
                                       unsigned int n_pgroup,
                                       unsigned int *d_particle_offset,
                                       unsigned int *d_particle_body,
                                       Scalar4* d_rigid_orientation,
                                       Scalar4* d_rigid_com,
                                       Scalar4* d_rigid_vel,
                                       Scalar4* d_rigid_angvel,
                                       int3* d_rigid_image,
                                       Scalar4* d_rigid_particle_dis,
                                       Scalar4* d_rigid_particle_orientation,
                                       unsigned int nmax,
                                       BoxDim box)
    {
    Scalar4 com, vel, angvel, ex_space, ey_space, ez_space;
    int3 body_image = make_int3(0, 0, 0);

    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_pgroup)
        return;

    unsigned int pidx = d_pgroup_idx[group_idx];
    unsigned int idx_body = d_particle_body[pidx];
    unsigned int particle_offset = d_particle_offset[pidx];
    Scalar4 body_orientation = d_rigid_orientation[idx_body];

    com = d_rigid_com[idx_body];
    vel = d_rigid_vel[idx_body];
    angvel = d_rigid_angvel[idx_body];
    if (set_x)
        {
        body_image = d_rigid_image[idx_body];
        }

    exyzFromQuaternion(body_orientation, ex_space, ey_space, ez_space);

    int localidx = idx_body * nmax + particle_offset;
    Scalar4 particle_pos = d_rigid_particle_dis[localidx];
    Scalar4 constituent_orientation = d_rigid_particle_orientation[localidx];

    // compute ri with new orientation
    Scalar3 ri;
    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;

    Scalar3 ppos;
    int3 image;
    Scalar4 porientation;
    if (set_x)
        {
        // x_particle = com + ri
        ppos.x = com.x + ri.x;
        ppos.y = com.y + ri.y;
        ppos.z = com.z + ri.z;

        // time to fix the periodic boundary conditions
        image = body_image;
        box.wrap(ppos, image);

        // update particle orientation
        quatquat(body_orientation,
                 constituent_orientation,
                 porientation);
        }

    // v_particle = vel + angvel x ri
    Scalar4 pvel = pdata_vel[pidx];
    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;

    // write out the results
    if (set_x)
        {
        pdata_pos[pidx] = make_scalar4(ppos.x, ppos.y, ppos.z, pdata_pos[pidx].w);
        pdata_image[pidx] = image;
        pdata_orientation[pidx] = porientation;
        }
    pdata_vel[pidx] = pvel;
    }
#endif

// Sets R and v of particles of the rigid body on the GPU
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_image array of particle images
    \param d_body array of particle body ids
    \param rigid_data Rigid body data
    \param d_pdata_orientation Particle orientations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate (all particles in rigid bodies)
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param set_x boolean indicating whether the positions are changed or not (first or second step of integration)
*/
cudaError_t gpu_rigid_setRV(Scalar4 *d_pos,
                            Scalar4 *d_vel,
                            int3 *d_image,
                            unsigned int *d_body,
                                   const gpu_rigid_data_arrays& rigid_data,
                                   Scalar4 *d_pdata_orientation,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const BoxDim& box, 
                                   bool set_x)
    {
    
    assert(d_pos);
    assert(d_vel);
    assert(d_pdata_orientation);
    assert(d_image);
    assert(d_group_members);
    
    assert(rigid_data.particle_offset);    
    assert(d_body);
    assert(rigid_data.orientation);    
    assert(rigid_data.com);
    assert(rigid_data.vel);
    assert(rigid_data.angvel);    
    assert(rigid_data.body_image);
    assert(rigid_data.particle_pos);
    assert(rigid_data.particle_orientation); 
    
    unsigned int nmax = rigid_data.nmax;

    unsigned int block_size = 192;
    dim3 particle_grid(group_size/block_size+1, 1, 1);
    dim3 particle_threads(block_size, 1, 1);
    
    if (set_x)
        gpu_rigid_setRV_kernel<true><<< particle_grid, particle_threads >>>(d_pos,
                                                                        d_vel,
                                                                        d_image,
                                                                        d_pdata_orientation,
                                                                        d_group_members,
                                                                        group_size,
                                                                        rigid_data.particle_offset,
                                                                        d_body,
                                                                        rigid_data.orientation,
                                                                        rigid_data.com,
                                                                        rigid_data.vel,
                                                                        rigid_data.angvel,
                                                                        rigid_data.body_image,
                                                                        rigid_data.particle_pos,
                                                                        rigid_data.particle_orientation,
                                                                        nmax,
                                                                        box);
     else
        gpu_rigid_setRV_kernel<false><<< particle_grid, particle_threads >>>(d_pos,
                                                                        d_vel,
                                                                        d_image,
                                                                        d_pdata_orientation,
                                                                        d_group_members,
                                                                        group_size,
                                                                        rigid_data.particle_offset,
                                                                        d_body,
                                                                        rigid_data.orientation,
                                                                        rigid_data.com,
                                                                        rigid_data.vel,
                                                                        rigid_data.angvel,
                                                                        rigid_data.body_image,
                                                                        rigid_data.particle_pos,
                                                                        rigid_data.particle_orientation,
                                                                        nmax,
                                                                        box);                                                                   
        return cudaSuccess;
}

//! Kernel driven by gpu_compute_virial_correction_end()
__global__ void gpu_compute_virial_correction_end_kernel(Scalar *d_net_virial,
                                                         unsigned int virial_pitch,
                                                         const Scalar4 *d_net_force,
                                                         const Scalar4 *d_oldpos,
                                                         const Scalar4 *d_oldvel,
                                                         const Scalar4 *d_vel,
                                                         const unsigned int *d_body,
                                                         Scalar deltaT,
                                                         unsigned int N)
    {
    unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= N)
        return;
    
    if (d_body[pidx] != NO_BODY)
        {
        // calculate the virial from the position and velocity from the previous step
        Scalar4 old_vel = d_oldvel[pidx];
        Scalar4 old_pos = d_oldpos[pidx];
        Scalar4 vel = d_vel[pidx];
        Scalar mass = vel.w;
        Scalar4 net_force = d_net_force[pidx];
        Scalar3 fc;
        fc.x = mass * (vel.x - old_vel.x) / deltaT - net_force.x;
        fc.y = mass * (vel.y - old_vel.y) / deltaT - net_force.y;
        fc.z = mass * (vel.z - old_vel.z) / deltaT - net_force.z;

        d_net_virial[0*virial_pitch+pidx] += old_pos.x * fc.x;
        d_net_virial[1*virial_pitch+pidx] += old_pos.x * fc.y;
        d_net_virial[2*virial_pitch+pidx] += old_pos.x * fc.z;
        d_net_virial[3*virial_pitch+pidx] += old_pos.y * fc.y;
        d_net_virial[4*virial_pitch+pidx] += old_pos.y * fc.z;
        d_net_virial[5*virial_pitch+pidx] += old_pos.z * fc.z;
        }
    }

/*! \param d_net_virial Net virial data to update with correction terms
    \param virial_pitch Pitch of d_net_virial
    \param d_net_force Net force on each particle
    \param d_oldpos Old position of particles saved at the start of the step
    \param d_oldvel Old velocity of particles saved at the start of the step
    \param d_vel Current velocity of particles at the end of the step
    \param d_body Body index of each particle
    \param deltaT Step size
    \param N number of particles in the box
*/
cudaError_t gpu_compute_virial_correction_end(Scalar *d_net_virial,
                                              const unsigned int virial_pitch,
                                              const Scalar4 *d_net_force,
                                              const Scalar4 *d_oldpos,
                                              const Scalar4 *d_oldvel,
                                              const Scalar4 *d_vel,
                                              const unsigned int *d_body,
                                              Scalar deltaT,
                                              unsigned int N)
    {
    assert(d_net_virial);
    assert(d_net_force);
    assert(d_oldpos);
    assert(d_oldvel);
    assert(d_vel);
    
    unsigned int block_size = 192;
    dim3 particle_grid(N/block_size+1, 1, 1);
    dim3 particle_threads(block_size, 1, 1);
    
    gpu_compute_virial_correction_end_kernel<<<particle_grid, particle_threads>>>(d_net_virial,
                                                                                  virial_pitch,
                                                                                  d_net_force,
                                                                                  d_oldpos,
                                                                                  d_oldvel,
                                                                                  d_vel,
                                                                                  d_body,
                                                                                  deltaT,
                                                                                  N);

    return cudaSuccess;
    }

