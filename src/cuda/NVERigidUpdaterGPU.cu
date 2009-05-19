/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: NVEUpdaterGPU.cu 1676 2009-02-13 19:15:18Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/cuda/NVEUpdaterGPU.cu $

#include "Integrator.cuh"
#include "NVERigidUpdaterGPU.cuh"
#include "gpu_settings.h"
#include <stdio.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file NVERigidUpdaterGPU.cu
	\brief Defines GPU kernel code for NVE integration on the GPU. Used by NVEUpdaterGPU.
*/

#define INVALID_INDEX 0xffffffff // identical to the sentinel value NO_INDEX in RigidData.h

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;
//! The texture for reading in the pdata mass array
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;

//! The texture for reading the rigid data body mass array
texture<float, 1, cudaReadModeElementType> rigid_data_body_mass_tex;
//! The texture for reading the rigid data moment of inertia array
texture<float4, 1, cudaReadModeElementType> rigid_data_moment_inertia_tex;
//! The texture for reading the rigid data com array
texture<float4, 1, cudaReadModeElementType> rigid_data_com_tex;
//! The texture for reading the rigid data vel array
texture<float4, 1, cudaReadModeElementType> rigid_data_vel_tex;
//! The texture for reading the rigid data angualr momentum array
texture<float4, 1, cudaReadModeElementType> rigid_data_angmom_tex;
//! The texture for reading the rigid data angular velocity array
texture<float4, 1, cudaReadModeElementType> rigid_data_angvel_tex;
//! The texture for reading the rigid data orientation array
texture<float4, 1, cudaReadModeElementType> rigid_data_orientation_tex;
//! The texture for reading the rigid data ex space array
texture<float4, 1, cudaReadModeElementType> rigid_data_exspace_tex;
//! The texture for reading the rigid data ey space array
texture<float4, 1, cudaReadModeElementType> rigid_data_eyspace_tex;
//! The texture for reading the rigid data ez space array
texture<float4, 1, cudaReadModeElementType> rigid_data_ezspace_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagex_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagey_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagez_tex;
//! The texture for reading the rigid data particle position array
texture<float4, 1, cudaReadModeElementType> rigid_data_particle_pos_tex;
//! The texture for reading the rigid data particle indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_particle_indices_tex;
//! The texture for reading the rigid data force array
texture<float4, 1, cudaReadModeElementType> rigid_data_force_tex;
//! The texture for reading the rigid data torque array
texture<float4, 1, cudaReadModeElementType> rigid_data_torque_tex;

#pragma mark HELPER
//! Helper functions for rigid body quaternion update

__device__ void exyzFromQuaternion(float4& quat, float4& ex_space, float4& ey_space, float4& ez_space)
	{
	// ex_space
	ex_space.x = quat.x * quat.x + quat.y * quat.y - quat.z * quat.z - quat.w * quat.w;
	ex_space.y = 2.0 * (quat.y * quat.z + quat.x * quat.w);
	ex_space.z = 2.0 * (quat.y * quat.w - quat.x * quat.z);
	
	// ey_space
	ey_space.x = 2.0 * (quat.y * quat.z - quat.x * quat.w);
	ey_space.y = quat.x * quat.x - quat.y * quat.y + quat.z * quat.z - quat.w * quat.w;
	ey_space.z = 2.0 * (quat.z * quat.w + quat.x * quat.y);
	
	// ez_space
	ez_space.x = 2.0 * (quat.y * quat.w + quat.x * quat.z);
	ez_space.y = 2.0 * (quat.z * quat.w - quat.x * quat.y);
	ez_space.z = quat.x * quat.x - quat.y * quat.y - quat.z * quat.z + quat.w * quat.w;
	}

__device__ void computeAngularVelocity(float4& angmom, float4& moment_inertia, float4& ex_space, float4& ey_space, float4& ez_space, float4& angvel)
	{
	//! Angular velocity in the body frame
	float4 angbody;
	
	//! angbody = angmom_body / moment_inertia = transpose(rotation_matrix) * angmom / moment_inertia  
	if (moment_inertia.x == 0.0) angbody.x = 0.0;
	else angbody.x = (ex_space.x * angmom.x + ex_space.y * angmom.y + ex_space.z * angmom.z) / moment_inertia.x;
	
	if (moment_inertia.y == 0.0) angbody.y = 0.0;
	else angbody.y = (ey_space.x * angmom.x + ey_space.y * angmom.y + ey_space.z * angmom.z) / moment_inertia.y;
	
	if (moment_inertia.z == 0.0) angbody.z = 0.0;
	else angbody.z = (ez_space.x * angmom.x + ez_space.y * angmom.y + ez_space.z * angmom.z) / moment_inertia.z;
	
	//! Convert to angbody to the space frame: angvel = rotation_matrix * angbody
	angvel.x = angbody.x * ex_space.x + angbody.y * ey_space.x + angbody.z * ez_space.x;
	angvel.y = angbody.x * ex_space.y + angbody.y * ey_space.y + angbody.z * ez_space.y;
	angvel.z = angbody.x * ex_space.z + angbody.y * ey_space.z + angbody.z * ez_space.z;
	}

/*! Quaternion multiply: c = a * b where a = (0, a)
 */

__device__ void multiply(float4& a, float4& b, float4& c)
	{
	c.x = -(a.x * b.y + a.y * b.z + a.z * b.w);
	c.y =   b.x * a.x + a.y * b.w - a.z * b.z;
	c.z =   b.x * a.y + a.z * b.y - a.x * b.w;
	c.w =   b.x * a.z + a.x * b.z - a.y * b.y;
	}

/*! Normalize a quaternion
 */

__device__ void normalize(float4 &q)
	{
	float norm = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
	q.x *= norm;
	q.y *= norm;
	q.z *= norm;
	q.w *= norm;
	}

/*! Advance the quaternion using angular momentum and angular velocity
 */
__device__ void advanceQuaternion(float4& angmom, float4& moment_inertia, float4& angvel, float4& ex_space, float4& ey_space, float4& ez_space, float4& quat, float deltaT)
	{
	float4 qhalf, qfull, omegaq;
	float dtq = 0.5 * deltaT;
	
	computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
	
	// Compute (w q)
	multiply(angvel, quat, omegaq);
	
	// Full update q from dq/dt = 1/2 w q
	qfull.x = quat.x + dtq * omegaq.x;
	qfull.y = quat.y + dtq * omegaq.y;
	qfull.z = quat.z + dtq * omegaq.z;
	qfull.w = quat.w + dtq * omegaq.w;
	normalize(qfull);
	
	// 1st half update from dq/dt = 1/2 w q
	qhalf.x = quat.x + 0.5 * dtq * omegaq.x;
	qhalf.y = quat.y + 0.5 * dtq * omegaq.y;
	qhalf.z = quat.z + 0.5 * dtq * omegaq.z;
	qhalf.w = quat.w + 0.5 * dtq * omegaq.w;
	normalize(qhalf);
	
	// Udpate ex, ey, ez from qhalf = update A
	exyzFromQuaternion(qhalf, ex_space, ey_space, ez_space);
	
	// Compute angular velocity from new ex_space, ey_space and ex_space
	computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
	
	// Compute (w qhalf)
	multiply(angvel, qhalf, omegaq);
	
	// 2nd half update from dq/dt = 1/2 w q
	qhalf.x += 0.5 * dtq * omegaq.x;
	qhalf.y += 0.5 * dtq * omegaq.y;
	qhalf.z += 0.5 * dtq * omegaq.z;
	qhalf.w += 0.5 * dtq * omegaq.w;
	normalize(qhalf);
	
	// Corrected Richardson update
	quat.x = 2.0 * qhalf.x - qfull.x;
	quat.y = 2.0 * qhalf.y - qfull.y;
	quat.z = 2.0 * qhalf.z - qfull.z;
	quat.w = 2.0 * qhalf.w - qfull.w;
	normalize(quat);
	
	exyzFromQuaternion(quat, ex_space, ey_space, ez_space);
	}

#pragma mark BODY_PRE_STEP_KERNEL
//! Takes the first half-step forward for rigid bodies in the velocity-verlet NVE integration
/*! \param rigid_data rigid data to step forward 1/2 step
	\param deltaT timestep
	\param limit If \a limit is true, then the dynamics will be limited so that particles do not move 
		a distance further than \a limit_val in one step.
	\param limit_val Length to limit particle distance movement to
	\param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_body_pre_step_kernel(gpu_pdata_arrays pdata, gpu_rigid_data_arrays rigid_data, gpu_boxsize box, float deltaT, bool limit, float limit_val)
	{
	unsigned int idx_particle = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int idx_body = blockIdx.x + rigid_data.local_beg; 
		
	// do velocity verlet update
	// v(t+deltaT/2) = v(t) + (1/2)a*deltaT
	// r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
	float body_mass;
	float4 moment_inertia, com, vel, angmom, angvel, orientation, ex_space, ey_space, ez_space, particle_pos, force, torque;
	int body_imagex, body_imagey, body_imagez;
	float4 ri = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pos2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 particle_accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	// ri, body_mass and moment_inertia is used throughout the kernel
	if (idx_body < rigid_data.n_bodies)
		{
		body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
		moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
		com = tex1Dfetch(rigid_data_com_tex, idx_body);
		vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
		angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
		angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
		orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
		ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
		ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
		ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
		body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
		body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
		body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);	
		particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
		force = tex1Dfetch(rigid_data_force_tex, idx_body);
		torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
		
		// update velocity
		float dtfm = (1.0f/2.0f) * deltaT / body_mass;
		
		vel2.x = vel.x + dtfm * force.x;
		vel2.y = vel.y + dtfm * force.y;
		vel2.z = vel.z + dtfm * force.z;
		vel2.w = vel.w;
			
		// update position
		pos2.x = com.x + vel2.x * deltaT;
		pos2.y = com.y + vel2.y * deltaT;
		pos2.z = com.z + vel2.z * deltaT;
		pos2.w = com.w;
						
		// read in body's image
		// time to fix the periodic boundary conditions
		float x_shift = rintf(pos2.x * box.Lxinv);
		pos2.x -= box.Lx * x_shift;
		body_imagex += (int)x_shift;
		
		float y_shift = rintf(pos2.y * box.Lyinv);
		pos2.y -= box.Ly * y_shift;
		body_imagey += (int)y_shift;
		
		float z_shift = rintf(pos2.z * box.Lzinv);
		pos2.z -= box.Lz * z_shift;
		body_imagez += (int)z_shift;
	
		// update the angular momentum
		float4 angmom2;
		angmom2.x = angmom.x + (1.0f/2.0f) * deltaT * torque.x;
		angmom2.y = angmom.y + (1.0f/2.0f) * deltaT * torque.y;
		angmom2.z = angmom.z + (1.0f/2.0f) * deltaT * torque.z;
			
		advanceQuaternion(angmom2, moment_inertia, angvel, ex_space, ey_space, ez_space, orientation, deltaT); 
			
		// write out the results (MEM_TRANSFER: ? bytes)
		if (threadIdx.x == 0)
			{
			rigid_data.com[idx_body] = pos2;
			rigid_data.vel[idx_body] = vel2;
			rigid_data.angmom[idx_body] = angmom2;
			rigid_data.angvel[idx_body] = angvel;
			rigid_data.orientation[idx_body] = orientation;
			rigid_data.ex_space[idx_body] = ex_space;
			rigid_data.ey_space[idx_body] = ey_space;
			rigid_data.ez_space[idx_body] = ez_space;
			rigid_data.body_imagex[idx_body] = body_imagex;
			rigid_data.body_imagey[idx_body] = body_imagey;
			rigid_data.body_imagez[idx_body] = body_imagez;
			}
		}
		
	unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
	// Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
	// the particle index of these empty slots is set to be INVALID_INDEX.
	if (idx_body < rigid_data.n_bodies && idx_particle_index != INVALID_INDEX) 
		{
		// time to fix the periodic boundary conditions (FLOPS: 15)
		int4 image = tex1Dfetch(pdata_image_tex, idx_particle_index);
		float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
		
		// compute ri with new orientation
		ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
		ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
		ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
	
		// x_particle = com + ri
		float4 ppos;
		ppos.x = pos2.x + ri.x;
		ppos.y = pos2.y + ri.y;
		ppos.z = pos2.z + ri.z;
		ppos.w = pos.w;
		
		float x_shift = rintf(ppos.x * box.Lxinv);
		ppos.x -= box.Lx * x_shift;
		image.x += (int)x_shift;
		
		float y_shift = rintf(ppos.y * box.Lyinv);
		ppos.y -= box.Ly * y_shift;
		image.y += (int)y_shift;
		
		float z_shift = rintf(ppos.z * box.Lzinv);
		ppos.z -= box.Lz * z_shift;
		image.z += (int)z_shift;
		
		// v_particle = vel + angvel x ri
		float4 pvel;
		pvel.x = vel2.x + angvel.y * ri.z - angvel.z * ri.y;
		pvel.y = vel2.y + angvel.z * ri.x - angvel.x * ri.z;
		pvel.z = vel2.z + angvel.x * ri.y - angvel.y * ri.x;
		pvel.w = 0.0;
		
		// write out the results (MEM_TRANSFER: ? bytes)
		pdata.pos[idx_particle_index] = ppos;
		pdata.vel[idx_particle_index] = pvel;
		pdata.image[idx_particle_index] = image;
		}
	}


/*! \param pdata Particle data to step forward 1/2 step
	\param box Box dimensions for periodic boundary condition handling
	\param deltaT Amount of real time to step forward in one time step
	\param limit If \a limit is true, then the dynamics will be limited so that particles do not move 
		a distance further than \a limit_val in one step.
	\param limit_val Length to limit particle distance movement to
*/
cudaError_t gpu_nve_rigid_body_pre_step(const gpu_pdata_arrays& pdata, const gpu_rigid_data_arrays& rigid_data, const gpu_boxsize &box, float deltaT, bool limit, float limit_val)
{
	unsigned int n_bodies = rigid_data.n_bodies;
	unsigned int nmax = rigid_data.nmax;
		
	// setup the grid to run the kernel 
	int block_size = nmax; // maximum number of particles in a rigid body: each thread in a block takes care of a particle in a rigid body
	dim3 grid(n_bodies, 1, 1);	
	dim3 threads(block_size, 1, 1);

	// bind the textures for rigid bodies: 
	// body mass, com, vel, angmom, angvel, orientation, ex_space, ey_space, ez_space, body images, particle pos, particle indices, force and torque
	cudaError_t error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_orientation_tex, rigid_data.orientation, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_body_imagex_tex, rigid_data.body_imagex, sizeof(int) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_body_imagey_tex, rigid_data.body_imagey, sizeof(int) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_body_imagez_tex, rigid_data.body_imagez, sizeof(int) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_force_tex, rigid_data.force, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, rigid_data_torque_tex, rigid_data.torque, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	// bind the textures for particles: pos, vel, accel and image of ALL particles
	error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, pdata_image_tex, pdata.image, sizeof(int4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	// run the kernel for bodies
    gpu_nve_rigid_body_pre_step_kernel<<< grid, threads, nmax * sizeof(float4)  >>>(pdata, rigid_data, box, deltaT, limit, limit_val);

		
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
			
	}

#pragma mark BODY_STEP_KERNEL

//! Takes the 2nd 1/2 step forward in the velocity-verlet NVE integration scheme
/*! \param pdata Particle data to step forward in time
	\param force_data_ptrs List of pointers to forces on each particle
	\param num_forces Number of forces listed in \a force_data_ptrs
	\param deltaT Amount of real time to step forward in one time step
	\param limit If \a limit is true, then the dynamics will be limited so that particles do not move 
		a distance further than \a limit_val in one step.
	\param limit_val Length to limit particle distance movement to
*/

extern __shared__ float4 sum[];

extern "C" __global__ void gpu_nve_rigid_body_step_kernel(gpu_pdata_arrays pdata, gpu_rigid_data_arrays rigid_data, float4 **force_data_ptrs, int num_forces, gpu_boxsize box, float deltaT, bool limit, float limit_val)
	{
	int idx_particle = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_body = blockIdx.x + rigid_data.local_beg; 

	float4 *body_force = sum;
	float4 *body_torque = &sum[blockDim.x];
		
	body_force[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	body_torque[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
	// Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
	// the particle index of these empty slots is set to be 0xffffffff.
	float body_mass;
	float4 moment_inertia, com, vel, angmom, angvel, ex_space, ey_space, ez_space, particle_pos, ri;
	float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 particle_accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		
	particle_accel = gpu_integrator_sum_forces_inline(idx_particle_index, pdata.local_num, force_data_ptrs, num_forces);		
	if (idx_body < rigid_data.n_bodies && idx_particle_index != INVALID_INDEX)
		{
		// calculate body force and torques
		float particle_mass = tex1Dfetch(pdata_mass_tex, idx_particle_index);
		float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
		
		// project the position in the body frame to the space frame: ri = rotation_matrix * particle_pos
		float4 fi, torquei;
		
		fi.x = particle_mass * particle_accel.x;
		fi.y = particle_mass * particle_accel.y;
		fi.z = particle_mass * particle_accel.z;
		fi.w = 0.0;
	
		body_force[threadIdx.x].x = fi.x;
		body_force[threadIdx.x].y = fi.y;
		body_force[threadIdx.x].z = fi.z;
		body_force[threadIdx.x].w = fi.w;
		
		com = tex1Dfetch(rigid_data_com_tex, idx_body);
		ri.x = pos.x - com.x;
		ri.y = pos.y - com.y;
		ri.z = pos.z - com.z;
		
		float x_shift = rintf(ri.x * box.Lxinv);
		ri.x -= box.Lx * x_shift;
		float y_shift = rintf(ri.y * box.Lyinv);
		ri.y -= box.Ly * y_shift;
		float z_shift = rintf(ri.z * box.Lzinv);
		ri.z -= box.Lz * z_shift;

		torquei.x = ri.y * fi.z - ri.z * fi.y;
		torquei.y = ri.z * fi.x - ri.x * fi.z;
		torquei.z = ri.x * fi.y - ri.y * fi.x;
		torquei.w = 0.0;
		
		body_torque[threadIdx.x].x = torquei.x;
		body_torque[threadIdx.x].y = torquei.y;
		body_torque[threadIdx.x].z = torquei.z;
		body_torque[threadIdx.x].w = torquei.w;		
		}
	
	__syncthreads();
			
	unsigned int offset = blockDim.x >> 1;	
	while (offset > 0)
		{
		if (threadIdx.x < offset)
			{
			body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
			body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
			body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
			body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;
								
			body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
			body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
			body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
			body_torque[threadIdx.x].w += body_torque[threadIdx.x + offset].w;
			}
				
		offset >>= 1;
				
		__syncthreads();
		}	
	
	// Update body velocity and angmom
	if (idx_body < rigid_data.n_bodies)
		{
		// Every thread now has its own copy of body force and torque
		float4 force2 = body_force[0];
		float4 torque2 = body_torque[0];
				
		// update the velocity
		body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
		vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
		float dtfm = (1.0f/2.0f) * deltaT / body_mass;
		vel2.x = vel.x + dtfm * force2.x;
		vel2.y = vel.y + dtfm * force2.y;
		vel2.z = vel.z + dtfm * force2.z;
		vel2.w = 0.0;
		
		// update angular momentum
		angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
		float4 angmom2;
		angmom2.x = angmom.x + (1.0f/2.0f) * deltaT * torque2.x;
		angmom2.y = angmom.y + (1.0f/2.0f) * deltaT * torque2.y;
		angmom2.z = angmom.z + (1.0f/2.0f) * deltaT * torque2.z;
		angmom2.w = 0.0;
		
		// update angular velocity
		moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
		ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
		ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
		ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);

		computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel);
		
		// thread 0 writes out the results
		if (threadIdx.x == 0)
			{
			rigid_data.force[idx_body] = force2;
			rigid_data.torque[idx_body] = torque2;
			rigid_data.vel[idx_body] = vel2;
			rigid_data.angmom[idx_body] = angmom2;
			rigid_data.angvel[idx_body] = angvel;
			}	
		
				
		if (idx_particle_index != INVALID_INDEX)
			{
			particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
		
			ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
			ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
			ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;

			// v_particle = v_com + angvel x xr
			float4 pvel;
			pvel.x = vel2.x + angvel.y * ri.z - angvel.z * ri.y;
			pvel.y = vel2.y + angvel.z * ri.x - angvel.x * ri.z;
			pvel.z = vel2.z + angvel.x * ri.y - angvel.y * ri.x;
			pvel.w = 0.0;
			
			// write out the results 
			pdata.vel[idx_particle_index] = pvel;
			}
		}	
	}


/*! \param pdata Particle data to step forward in time
	\param force_data_ptrs List of pointers to forces on each particle
	\param num_forces Number of forces listed in \a force_data_ptrs
	\param deltaT Amount of real time to step forward in one time step
	\param limit If \a limit is true, then the dynamics will be limited so that particles do not move 
		a distance further than \a limit_val in one step.
	\param limit_val Length to limit particle distance movement to
*/
cudaError_t gpu_nve_rigid_body_step(const gpu_pdata_arrays &pdata, const gpu_rigid_data_arrays& rigid_data, float4 **force_data_ptrs, int num_forces, const gpu_boxsize &box, float deltaT, bool limit, float limit_val)
	{	
	unsigned int n_bodies = rigid_data.n_bodies;
	unsigned int nmax = rigid_data.nmax;
	
	// setup the grid to run the particle kernel 
	int block_size = nmax; // each thread in a block takes care of a particle in a rigid body
	dim3 grid(n_bodies, 1, 1);	
	dim3 threads(block_size, 1, 1);
	
	// bind the textures for ALL rigid bodies
	cudaError_t error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
			
	error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
	if (error != cudaSuccess)
		return error;
	
	error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
	if (error != cudaSuccess)
		return error;
		
	// bind the textures for particles
	
	error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
	if (error != cudaSuccess)
		return error;
	
	
	// run the kernel: the shared memory size is used for dynamic memory allocation of extern __shared__ sum
	// tested with separate extern __shared__ body_force and body_torque each with size of nmax * sizeof(float4) but it did not work
    gpu_nve_rigid_body_step_kernel<<< grid, threads, 2 * nmax * sizeof(float4) >>>(pdata, rigid_data, force_data_ptrs, num_forces, box, deltaT, limit, limit_val);
	
	
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
	}

