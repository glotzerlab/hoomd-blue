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

#ifndef _FORCECOMPUTE_H_
#define _FORCECOMPUTE_H_

#include <stdio.h>
#include <cuda_runtime.h>

/*! \file ForceCompute.cuh
    \brief Declares data structures for calculating forces on the GPU. Used by ForceCompute and descendants.
*/

//! Force data stored on the GPU
/*! Stores device pointers to allocated force data on the GPU. \a force[local_idx] holds the
    x,y,z componets of the force and the potential energy in w for particle \em local_idx.
    \a virial[local_idx] holds the single particle virial value for particle \em local_idx. See
     ForceDataArrays for a definition of what the single particle virial and potential energy
     mean.

    Only forces for particles belonging to a GPU are stored there, thus each array
    is allocated to be of length \a local_num (see gpu_pdata_arrays)

    \ingroup gpu_data_structs
*/
struct gpu_force_data_arrays
    {
    float4 *force;  //!< Force in \a x, \a y, \a z and the single particle potential energy in \a w.
    float *virial;  //!< Single particle virial
    
    //! Allocates memory
    cudaError_t allocate(unsigned int num_local);
    
    //! Frees memory
    cudaError_t deallocate();
    };

#endif
