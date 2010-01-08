/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifndef _FORCECOMPUTE_H_
#define _FORCECOMPUTE_H_

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

