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
// Maintainer: joaander

#include "ForceCompute.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ForceCompute.cu
    \brief Defines data structures for calculating forces on the GPU. Used by ForceCompute and descendants.
*/

/*! \pre allocate() has not previously been called
    \post Memory for \a force and \a virial is allocated on the device
    \param num_local Number of particles local to the GPU on which this is being called
    \note allocate() \b must be called on the GPU it is to allocate data on
*/
cudaError_t gpu_force_data_arrays::allocate(unsigned int num_local)
    {
    // sanity checks
    assert(force == NULL);
    assert(virial == NULL);
    
    // allocate force and check for errors
    cudaError_t error = cudaMalloc((void **)((void *)&force), sizeof(float4)*num_local);
    if (error != cudaSuccess)
        return error;
        
    // allocate virial and check for errors
    error = cudaMalloc((void **)((void *)&virial), sizeof(float)*num_local);
    if (error != cudaSuccess)
        return error;
        
    // all done, return success
    return cudaSuccess;
    }

/*! \pre allocate() has been called
    \post Memory for \a force and \a virial is freed on the device
    \note deallocate() \b must be called on the same GPU as allocate()
*/
cudaError_t gpu_force_data_arrays::deallocate()
    {
    // sanity checks
    assert(force != NULL);
    assert(virial != NULL);
    
    // free force and check for errors
    cudaError_t error = cudaFree((void*)force);
    if (error != cudaSuccess)
        return error;
        
    // free virial and check for errors
    error = cudaFree((void*)virial);
    if (error != cudaSuccess)
        return error;
        
    // all done, return success
    return cudaSuccess;
    }
