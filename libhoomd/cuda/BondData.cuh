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

// Maintainer: joaander

#ifndef _BONDDATA_CUH_
#define _BONDDATA_CUH_

#include <cuda_runtime.h>

/*! \file BondData.cuh
    \brief GPU helper functions used in BondData
*/

#ifdef NVCC
//! Structure to keep a bond, used for bond buffers
struct bond_element
    {
    uint2 bond;                //!< Member tags of the bond
    unsigned int type;         //!< Type of the bond
    unsigned int tag;          //!< Unique bond identifier
    };

//! Sentinal value in \a bond_r_tag to signify that this bond is not currently present on the local processor
const unsigned int BOND_NOT_LOCAL = 0xffffffff;
#else
//! Forward declaration
class bond_element;
#endif

//! Find the maximum number of bonds per particle
cudaError_t gpu_find_max_bond_number(unsigned int *d_n_bonds,
                                     const uint2 *d_bonds,
                                     const unsigned int num_bonds,
                                     const unsigned int N,
                                     const unsigned int n_ghosts,
                                     const unsigned int *d_rtag,
                                     const unsigned int cur_max,
                                     unsigned int *d_condition);

//! Construct the GPU bond table
cudaError_t gpu_create_bondtable(uint2 *d_gpu_bondtable,
                                 unsigned int *d_n_bonds,
                                 const uint2 *d_bonds,
                                 const unsigned int *d_bond_type,
                                 const unsigned int *d_rtag,
                                 const unsigned int num_bonds,
                                 unsigned int pitch,
                                 unsigned int N);

void gpu_mark_recv_bond_duplicates(const unsigned int n_bonds,
                                   const bond_element *d_recv_bonds,
                                   unsigned int *d_bond_remove_mask,
                                   const unsigned int n_recv_bonds,
                                   unsigned int *d_bond_rtag,
                                   unsigned char *d_recv_bond_active,
                                   unsigned int *d_n_duplicate_recv_bonds);

void gpu_fill_bond_bondtable(const unsigned int old_n_bonds,
                             const unsigned int n_recv_bonds,
                             const unsigned int n_unique_recv_bonds,
                             const unsigned int n_remove_bonds,
                             const unsigned int *d_remove_mask,
                             const unsigned char *d_recv_bond_active,
                             const bond_element *d_recv_buf,
                             uint2 *d_bonds,
                             unsigned int *d_bond_type,
                             unsigned int *d_bond_tag,
                             unsigned int *d_bond_rtag,
                             unsigned int *d_n_fetch_ptl);

void gpu_update_bond_rtags(unsigned int *d_bond_rtag,
                           const unsigned int *d_bond_tag,
                           const unsigned int num_bonds);
#endif
