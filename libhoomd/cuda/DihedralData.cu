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

#include "DihedralData.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file DihedralData.cu
    \brief Implements the helper functions for updating the GPU dihedral table
*/

//! Helper structure to get particle tag a, b, c or d from a dihedral
struct dihedral_get_tag : thrust::unary_function<uint4, unsigned int>
    {
    //! Constructor
    dihedral_get_tag(unsigned int _member_idx)
        : member_idx(_member_idx)
        { }
        
    //! Get particle tag
    __host__ __device__
    unsigned int operator ()(uint4 dihedral)
        {
        switch(member_idx)
            {
            case 0:
               return dihedral.x;
            case 1:
               return dihedral.y;
            case 2:
               return dihedral.z;
            case 3:
               return dihedral.w;
            default:
               // we should never get here
               return 0;
            }
        }

    private:
        unsigned int member_idx; //!< Index of particle tag to get
    };

//! Helper kernel to get particle tag a, b, c, d the dihedral type and the particle location (a b or c) from a dihedral
__global__ void gpu_kernel_dihedral_fill_values(const uint4 *dihedrals,
                            const unsigned int *dihedral_types,
                            const unsigned int *d_rtag,
                            uint4 *values,
                            uint1 *values_ABCD,
                            unsigned int member_idx,
                            unsigned int num_dihedrals)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_dihedrals)
        return;

    uint4 dihedral = dihedrals[idx];
    unsigned int tag1, tag2, tag3;
    switch(member_idx)
        {
        case 0:
            tag1 = dihedral.y;
            tag2 = dihedral.z;
            tag3 = dihedral.w;
            break;
        case 1:
            tag1 = dihedral.x;
            tag2 = dihedral.z;
            tag3 = dihedral.w;
            break;
        case 2:
            tag1 = dihedral.x;
            tag2 = dihedral.y;
            tag3 = dihedral.w;
            break;
        case 3:
            tag1 = dihedral.x;
            tag2 = dihedral.y;
            tag3 = dihedral.z;
            break;
        default:
           // we should never get here
           tag1 = tag2 = tag3 = 0;
        }

    values[idx] = make_uint4(d_rtag[tag1], d_rtag[tag2], d_rtag[tag3], dihedral_types[idx]);
    values_ABCD[idx] = make_uint1(member_idx);
    };


//! Find the maximum number of dihedrals per particle
/*! \param d_dihedrals Array of dihedrals
    \param d_dihedral_type Array of dihedral types
    \param num_dihedrals Size of dihedral array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag -> particle index
    \param d_n_dihedrals Number of dihedrals per particle
    \param max_dihedral_num Maximum number of dihedrals (return value)
    \param d_sort_keys Pointer to a temporary sorted list of first dihedral member indices (return value)
    \param d_sort_values Pointer to a temporary list of other dihedral member indices and dihedral types (sorted)
    \param d_sort_ABCD Pointer to a temporary list of relative atom positions in the dihedral (sorted)
 */
cudaError_t TransformDihedralDataGPU::gpu_find_max_dihedral_number(unsigned int& max_dihedral_num,
                                     uint4 *d_dihedrals,
                                     unsigned int *d_dihedral_type,
                                     unsigned int num_dihedrals,
                                     unsigned int N,
                                     unsigned int *d_rtag,
                                     unsigned int *d_n_dihedrals)
    {
    assert(d_dihedrals);
    assert(d_dihedral_type);
    assert(d_rtag);
    assert(d_n_dihedrals);

    thrust::device_ptr<uint4> dihedrals_ptr(d_dihedrals);
    thrust::device_ptr<unsigned int> dihedral_type_ptr(d_dihedral_type);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<unsigned int> n_dihedrals_ptr(d_n_dihedrals);

    if (dihedral_sort_keys.size() < 4*num_dihedrals)
        {
        dihedral_sort_keys.resize(4*num_dihedrals);
        dihedral_sort_values.resize(4*num_dihedrals);
        dihedral_sort_ABCD.resize(4*num_dihedrals);
        dihedral_indices.resize(4*num_dihedrals);
        num_dihedrals_sorted.resize(4*num_dihedrals);
        }

    // fill sort key and value arrays
    unsigned int block_size = 512;
    uint4 *d_dihedral_sort_values =  thrust::raw_pointer_cast(&* dihedral_sort_values.begin());
    uint1 *d_dihedral_sort_ABCD =  thrust::raw_pointer_cast(&* dihedral_sort_ABCD .begin());

    for (unsigned int i = 0; i < 4; i++)
        {
        thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(dihedrals_ptr, dihedral_get_tag(i))
                     ),
                     thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(dihedrals_ptr, dihedral_get_tag(i))
                     ) + num_dihedrals,
                 dihedral_sort_keys.begin() + i * num_dihedrals);

        // fill sort values
        gpu_kernel_dihedral_fill_values<<<num_dihedrals/block_size + 1, block_size>>>(d_dihedrals,
                                                           d_dihedral_type,
                                                           d_rtag,
                                                           d_dihedral_sort_values + i * num_dihedrals,
                                                           d_dihedral_sort_ABCD + i * num_dihedrals,
                                                           i,
                                                           num_dihedrals);
        }

    // sort first dihedral members as keys with other dihedral members, dihedral types and particle indicies in in dihedral as values
    thrust::sort_by_key(dihedral_sort_keys.begin(),
                        dihedral_sort_keys.begin() + 4 * num_dihedrals,
                        make_zip_iterator(thrust::make_tuple(dihedral_sort_values.begin(), dihedral_sort_ABCD.begin())));

    // count multiplicity of each key
    unsigned int n_unique_indices = thrust::reduce_by_key(dihedral_sort_keys.begin(),
                          dihedral_sort_keys.begin() + 4 * num_dihedrals,
                          thrust::constant_iterator<unsigned int>(1),
                          dihedral_indices.begin(),
                          num_dihedrals_sorted.begin() ).second - num_dihedrals_sorted.begin();

    // find the maximum
    max_dihedral_num = thrust::reduce(num_dihedrals_sorted.begin(),
                                  num_dihedrals_sorted.begin() + n_unique_indices,
                                  0,
                                  thrust::maximum<unsigned int>());

    // fill n_dihedrals array with zeros
    thrust::fill(n_dihedrals_ptr,
                 n_dihedrals_ptr + N,
                 0);

    // scatter dihedral numbers in n_dihedrals array
    thrust::scatter(num_dihedrals_sorted.begin(),
                    num_dihedrals_sorted.begin() + n_unique_indices,
                    dihedral_indices.begin(),
                    n_dihedrals_ptr);

    return cudaSuccess;
    }

//! Construct the GPU dihedral table
/*! \param num_dihedrals Size of dihedral array
    \param d_gpu_dihedraltable Pointer to the dihedral table on the GPU
    \param d_gpu_dihedral_ABCD Pointer to ABCD table on the GPU
    \param pitch Pitch of 2D dihedraltable array
    \param d_sort_keys First dihedral members as keys (sorted)
    \param d_sort_values Other dihedral members plus type as values (sorted)
    \param d_sort_ABCD Relative atom position in the dihedral (sorted)

    \pre Prior to calling this method, the internal dihedral_sort_keys, dihedral_sort_values
         and dihedral_sort_ABCD need to be initialized by a call to gpu_find_max_dihedral_number

 */
cudaError_t TransformDihedralDataGPU::gpu_create_dihedraltable(unsigned int num_dihedrals,
                                     uint4 *d_gpu_dihedraltable,
                                     uint1 *d_gpu_dihedral_ABCD,
                                     unsigned int pitch)
    {

    thrust::device_ptr<uint4> gpu_dihedraltable_ptr(d_gpu_dihedraltable);
    thrust::device_ptr<uint1> gpu_dihedral_ABCD_ptr(d_gpu_dihedral_ABCD);

    if (dihedral_map.size() < 4*num_dihedrals)
        {
        dihedral_map.resize(4*num_dihedrals);
        }

    // create the dihedral_map of 2D dihedral table indices for all first dihedral members
    thrust::exclusive_scan_by_key(dihedral_sort_keys.begin(),
                                  dihedral_sort_keys.begin() + 4 * num_dihedrals,
                                  thrust::make_constant_iterator(pitch),
                                  dihedral_map.begin());

    thrust::transform(dihedral_map.begin(),
                      dihedral_map.begin() + 4 * num_dihedrals,
                      dihedral_sort_keys.begin(),
                      dihedral_map.begin(),
                      thrust::plus<unsigned int>());

    // scatter the other dihedral members and the type into the 2D matrix according to the dihedral_map
    thrust::scatter(dihedral_sort_values.begin(),
                    dihedral_sort_values.begin() + 4* num_dihedrals,
                    dihedral_map.begin(),
                    gpu_dihedraltable_ptr);

    // scatter the relative position of the atoms in the dihedral into the 2D matrix d_gpu_dihedral_ABCD
    thrust::scatter(dihedral_sort_ABCD.begin(),
                   dihedral_sort_ABCD.begin() + 4 * num_dihedrals,
                   dihedral_map.begin(),
                   gpu_dihedral_ABCD_ptr);

    return cudaSuccess;
    }
