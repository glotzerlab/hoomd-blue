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

#ifndef _CUDA_FORCES_H_
#define _CUDA_FORCES_H_

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "gpu_pdata.h"
#include "gpu_nlist.h"

/*! \file gpu_forces.h
 	\brief Declares functions and data structures for calculating forces on the GPU
 	\details Functions in this file are NOT to be called by anyone not knowing 
		exactly what they are doing. They are designed to be used solely by 
		LJForceComputeGPU and BondForceComputeGPU.
*/

extern "C" {

//! The maximum number of particle types allowed
#define MAX_NTYPES 32
//! Precalculate the maximum number of type pairs
#define MAX_NTYPE_PAIRS	MAX_NTYPES*MAX_NTYPES

//! Stores parameters for the lennard jones potential
/*! The lj1 and lj2 params for the lennard jones potential are stored on the device in constant memory.
	This structure is what holds them. The number of type pairs are determined at compile time by a 
	define. Since only one constant memory area exists on the device, 
	the lj parameters are allocated with an id and selected into the device. Only when the selected id
	differs from the id last selected (or it is forced) will the data actually be copied over to the GPU

	The type pair in the params array is to be indexed by i*NLIST_MAX_NTYPES + j, and the symmetric 
	entry MUST be filled out.
*/
struct gpu_ljparam_data
	{
	//! coefficients
	float lj1[MAX_NTYPE_PAIRS];
	float lj2[MAX_NTYPE_PAIRS];

	//! identifier for this structure
	unsigned int id;
	};
	
	
//////////////////////////////////////////////////////////////////////////
// bond force data structures
//! This data structure is modeled after the neighbor list
struct gpu_bondtable_array
	{
	unsigned int *list;
	unsigned int height;
	unsigned int pitch;
	};

///////////////////////////// LJ params

//! Allocate ljparams
gpu_ljparam_data *gpu_alloc_ljparam_data();
//! Free ljparams
void gpu_free_ljparam_data(gpu_ljparam_data *ljparams);
//! Make the parameters active on the device
cudaError_t gpu_select_ljparam_data(gpu_ljparam_data *ljparams, bool force);
//! Perform the lj force calculation
cudaError_t gpu_ljforce_sum(float4 *d_forces, gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_array *nlist, float r_cutsq, int M);

//////////////////////////// Bond table stuff

//! Sum bond forces
cudaError_t gpu_bondforce_sum(float4 *d_forces, gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bondtable_array *btable, float K, float r_0, int block_size);
}

#endif

