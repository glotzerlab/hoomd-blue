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

// $Id: fenebondforce_kernel.cu 1158 2008-09-01 15:41:21Z phillicl $
// $URL: https://svn2.assembla.com/svn/hoomd/tags/hoomd-0.7.0/src/cuda/fenebondforce_kernel.cu $

#include "gpu_forces.h"
#include "gpu_pdata.h"
#include <iostream>
#include <stdexcept>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file fenebondforce_kernel.cu
	\brief Contains code that implements the fene bond force sum on the GPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading bond parameters
texture<float2, 1, cudaReadModeElementType> bond_params_tex;

extern "C" __global__ void calcFENEBondForces_kernel(float4 *d_forces, gpu_pdata_arrays pdata, gpu_bondtable_array blist, gpu_boxsize box)
	{
	// start by identifying which particle we are to handle
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pidx >= pdata.N)
		return;
	
	// load in the length of the list for this thread
	int n_bonds = blist.n_bonds[pidx];

	// read in the position of our particle. 
	float4 pos = tex1Dfetch(pdata_pos_tex, pidx);

	// initialize the force to 0
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	// loop over neighbors
	for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
		{
		// the volatile fails to compile in device emulation mode
		#ifdef _DEVICEEMU
		uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + pidx];
		#else
		// the volatile is needed to force the compiler to load the uint2 coalesced
		volatile uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + pidx];
		#endif
		
		int cur_bond_idx = cur_bond.x;
		int cur_bond_type = cur_bond.y;
		
		// get the bonded particle's position
		float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_bond_idx);
	
		// calculate dr
		float dx = pos.x - neigh_pos.x;
		float dy = pos.y - neigh_pos.y;
		float dz = pos.z - neigh_pos.z;
		
		// apply periodic boundary conditions
		dx -= box.Lx * rintf(dx * box.Lxinv);
		dy -= box.Ly * rintf(dy * box.Lyinv);
		dz -= box.Lz * rintf(dz * box.Lzinv);
		
		// get the bond parameters
		float2 params = tex1Dfetch(bond_params_tex, cur_bond_type);
		float K = params.x;
		float r_0 = params.y;
		
		float rsq = dx*dx + dy*dy + dz*dz;
		float r = sqrtf(rsq);
		
		//TODO - need a way to check if r >= r_0 and terminate the simulation if it is) 
		 
		float fforce = -K *r/ (1.0f - rsq/(r_0*r_0));
		float tmp_eng = -0.5f * K * r_0*r_0*logf(1.0f - rsq/(r_0*r_0));
				
		// add up the forces
		force.x += dx * fforce;
		force.y += dy * fforce;
		force.z += dz * fforce;
		force.w += tmp_eng;
		
		// Checking to see if bond length restriction is violated.
	//	if (r >= r_0) *blist.checkr = 1;
		
		}
		
	// energy is double counted: multiply by 0.5
	force.w *= 0.5f;
	
	// now that the force calculation is complete, write out the result if we are a valid particle
	d_forces[pidx] = force;
	}


/*! \param d_forces Device memory to write forces to
	\param pdata Particle data on the GPU to perform the calculation on
	\param box Box dimensions (in GPU format) to use for periodic boundary conditions
	\param btable List of bonds stored on the GPU
	\param d_params K and r_0 params packed as float2 variables
	\param n_bond_types Number of bond types in d_params
	\param block_size Block size to use when performing calculations
	
	\returns Any error code resulting from the kernel launch
	\note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
	
	\a d_params should include one float2 element per bond type. The x component contains K the spring constant
	and the y component contains r_0 the equilibrium length.
*/
cudaError_t gpu_fenebondforce_sum(float4 *d_forces, gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bondtable_array *btable, float2 *d_params, unsigned int n_bond_types, int block_size)
	{
	assert(pdata);
	assert(btable);
	assert(d_params);
	// check that block_size is valid
	assert(block_size != 0);

	// setup the grid to run the kernel
	dim3 grid( (int)ceil((double)pdata->N/ (double)block_size), 1, 1);
	dim3 threads(block_size, 1, 1);

	// bind the textures
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, bond_params_tex, d_params, sizeof(float2) * n_bond_types);
	if (error != cudaSuccess)
		return error;

	// start by zeroing check value on the device
/*	int *exceedsR0 = 0;
	

	error = cudaMemcpy(btable->checkr, exceedsR0,
			sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		return error;
*/
			
	// run the kernel
	calcFENEBondForces_kernel<<< grid, threads>>>(d_forces, *pdata, *btable, *box);
	
/*
	error = cudaMemcpy(exceedsR0, btable->checkr,
			sizeof(int), cudaMemcpyDeviceToHost);	
	if (error != cudaSuccess)
		return error;

    //check the fene bondlength violation condition
	if (*exceedsR0)
	 {
		std::cerr << std::endl << "***Error! FENE bond length exceeds maximum permitted" << std::endl << std::endl;
		throw std::runtime_error("Error in fene bond calculation");
	  }
*/										
	
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}
