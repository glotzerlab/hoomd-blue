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

// Maintainer: dnlebard

#ifndef _DIHEDRALDATA_CUH_
#define _DIHEDRALDATA_CUH_

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

/*! \file DihedralData.cuh
    \brief GPU helper functions used in DihedralData
*/

class TransformDihedralDataGPU
    {
    public:
        //! Find the maximum number of dihedrals per particle
        cudaError_t gpu_find_max_dihedral_number(unsigned int& max_dihedral_num,
                                             uint4 *d_dihedrals,
                                             unsigned int *d_dihedral_type,
                                             unsigned int num_dihedrals,
                                             unsigned int N,
                                             unsigned int *d_rtag,
                                             unsigned int *d_n_dihedrals);

        //! Construct the GPU dihedral table
        cudaError_t gpu_create_dihedraltable(unsigned int num_dihedrals,
                                             uint4 *d_gpu_dihedraltable,
                                             uint1 *d_gpu_dihedral_ABCD,
                                             unsigned int pitch);

    private:
        //! Sorted array of the first dihedral member as key
        thrust::device_vector<unsigned int> dihedral_sort_keys;

        //! Sorted array of three dihedral members and the dihedral type as value, for every particle part of a dihedral
        thrust::device_vector<uint4> dihedral_sort_values;

        //! Sorted array of position in the dihedral for every particle part of a dihedral
        thrust::device_vector<uint1> dihedral_sort_ABCD;

        //! Map of indices in the 2D GPU dihedral table for every first member of a dihedral
        thrust::device_vector<unsigned int> dihedral_map;

        //! Sorted list of number of dihedrals for each particle index
        thrust::device_vector<unsigned int> num_dihedrals_sorted;

        //! Sorted list of particle indices that are part of at least one dihedral
        thrust::device_vector<unsigned int> dihedral_indices;
    };
#endif

