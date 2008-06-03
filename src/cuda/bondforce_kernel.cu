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

// $Id$
// $URL$

#include "gpu_forces.h"
#include "gpu_pdata.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file bondforce_kernel.cu
	\brief Contains code that implements the bond force sum on the GPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

// this is a copied and pasted ljforces kernel modified to do bond forces
extern "C" __global__ void calcBondForces_kernel(float4 *d_forces, gpu_pdata_arrays pdata, gpu_bondtable_array blist, float K, float r_0, gpu_boxsize box)
	{
	// start by identifying which particle we are to handle
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pidx >= pdata.N)
		return;
	
	// load in the length of the list (each thread loads it individually) 
	int n_bonds = blist.list[pidx];

	// read in the position of our particle. Sure, this COULD be done as a fully coalesced global mem read
	// but reading it from the texture gives a slightly better performance, possibly because it "warms up" the
	// texture cache for the next read
	float4 pos = tex1Dfetch(pdata_pos_tex, pidx);

	// initialize the force to 0
	float fx = 0.0f;
	float fy = 0.0f;
	float fz = 0.0f;
	
	// loop over neighbors
	for (int neigh_idx = 0; neigh_idx < n_bonds; neigh_idx++)
		{
		int cur_neigh = blist.list[blist.pitch*(neigh_idx+1) + pidx];
			
		// get the neighbor's position
		float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
		float nx = neigh_pos.x;
		float ny = neigh_pos.y;
		float nz = neigh_pos.z;
	
		// calculate dr (with periodic boundary conditions)
		float dx = pos.x - nx;
		dx -= box.Lx * rintf(dx * box.Lxinv);
				
		float dy = pos.y - ny;
		dy -= box.Ly * rintf(dy * box.Lyinv);
			
		float dz = pos.z - nz;
		dz -= box.Lz * rintf(dz * box.Lzinv);
				
		float rsq = dx*dx + dy*dy + dz*dz;
		float rinv = rsqrtf(rsq);
		float fforce = 2.0f * K * (r_0 * rinv - 1.0f);
				
		// add up the forces
		fx += dx * fforce;
		fy += dy * fforce;
		fz += dz * fforce;
		}
		
	// now that the force calculation is complete, write out the result if we are a valid particle
	float4 force;
	force.x = fx;
	force.y = fy;
	force.z = fz;
	force.w = 0.0f;
	d_forces[pidx] = force;
	}


/*! \param d_forces Device memory to write forces to
	\param pdata Particle data on the GPU to perform the calculation on
	\param box Box dimensions (in GPU format) to use for periodic boundary conditions
	\param btable List of bonds stored on the GPU
	\param K Stiffness parameter of the bond
	\param r_0 Equilibrium bond length
	\param block_size Block size to use when performing calculations
	
	\returns Any error code resulting from the kernel launch
	\note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_bondforce_sum(float4 *d_forces, gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bondtable_array *btable, float K, float r_0, int block_size)
	{
	assert(pdata);
	assert(btable);
	// check that block_size is valid
	assert(block_size != 0);
	assert((block_size & 31) == 0);
	assert(block_size <= 512);

	// setup the grid to run the kernel
	dim3 grid( (int)ceil((double)pdata->N/ (double)block_size), 1, 1);
	dim3 threads(block_size, 1, 1);

	// bind the textures
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

	// run the kernel
	calcBondForces_kernel<<< grid, threads>>>(d_forces, *pdata, *btable, K, r_0, *box);
	
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}
