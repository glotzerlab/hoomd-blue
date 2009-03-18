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

// $Id: AngleData.cu 1514 2008-12-02 18:59:15Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/branches/hoomd-0.8/src/cuda/AngleData.cu $

#include "AngleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file AngleData.cu
 	\brief Defines the data structures for storing angles on the GPU.
*/

/*! \pre no allocations have been performed or deallocate() has been called after a previous allocate()
	\post Memory for \a n_angles and \a angles is allocated on the device
	\param num_local Number of particles local to the GPU on which this is being called
	\param alloc_height Number of angles to allocate for each particle
	\note allocate() \b must be called on the GPU it is to allocate data on
*/
cudaError_t gpu_angletable_array::allocate(unsigned int num_local, unsigned int alloc_height)
	{
	// sanity checks
	assert(n_angles == NULL);
	assert(angles == NULL);
	
		
	// allocate n_bonds and check for errors
	cudaError_t error = cudaMalloc((void**)((void*)&n_angles), num_local*sizeof(unsigned int));
	if (error != cudaSuccess)
		return error;
	
	error = cudaMemset((void*)n_angles, 0, num_local*sizeof(unsigned int));
	if (error != cudaSuccess)
		return error;

        
	// cudaMallocPitch fails to work for coalesced reads here (dunno why), need to calculate pitch ourselves
	// round up to the nearest multiple of 32
	height = alloc_height;
	pitch = (num_local + (32 - num_local & 31));
	error = cudaMalloc((void**)((void*)&angles), pitch * height * sizeof(uint4));
	if (error != cudaSuccess)
		return error;	
	
	error = cudaMemset((void*)angles, 0, pitch * height * sizeof(uint4));
	if (error != cudaSuccess)
		return error;
		
	// all done, return success
	return cudaSuccess;	
	}
	
/*! \pre allocate() has been called
	\post Memory for \a n_angles and \a angles is freed on the device
	\note deallocate() \b must be called on the same GPU as allocate()
*/
cudaError_t gpu_angletable_array::deallocate()
	{
	// sanity checks
	assert(n_angles != NULL);
	assert(angles != NULL);
	
	// free the memory
	cudaError_t error = cudaFree((void*)n_angles);
	n_angles = NULL;
	if (error != cudaSuccess)
		return error;
		
	error = cudaFree((void*)angles);
	angles = NULL;
	if (error != cudaSuccess)
		return error;


	// all done, return success
	return cudaSuccess;
	}
