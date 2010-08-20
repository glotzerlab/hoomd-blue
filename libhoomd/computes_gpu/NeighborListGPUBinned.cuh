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

#ifndef __NEIGHBORLOSTGPUBINNED_CUH__
#define __NEIGHBORLOSTGPUBINNED_CUH__

#include <cuda_runtime.h>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "ParticleData.cuh"

/*! \file NeighborListGPUBinned.cuh
    \brief Declares GPU kernel code for neighbor list generation on the GPU
*/

//! Kernel driver for the the first step of the computation called by CellListGPU
cudaError_t gpu_compute_nlist_binned(unsigned int *d_nlist,
                                     unsigned int *d_n_neigh,
                                     const Index2D& nli,
                                     const float4 *d_pos,
                                     const unsigned int N,
                                     const unsigned int *d_cell_size,
                                     const float4 *d_cell_xyzf,
                                     const unsigned int *d_cell_adj,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Index2D& cadji,
                                     const float3& cell_scale,
                                     const uint3& cell_dim,
                                     const gpu_boxsize& box,
                                     const float r_maxsq,
                                     const unsigned int block_size);

#endif
