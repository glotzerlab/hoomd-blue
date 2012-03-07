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

#include "AngleData.cuh"

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

/*! \file AngleData.cu
    \brief Implements the helper functions for updating the GPU angle table
*/

//! Helper structure to get particle tag a, b or c from an angle
struct get_tag : thrust::unary_function<uint3, unsigned int>
    {
    //! Constructor
    get_tag(unsigned int _member_idx)
        : member_idx(_member_idx)
        { }
        
    //! Get particle tag
    __host__ __device__
    unsigned int operator ()(uint3 angle)
        {
        switch(member_idx)
            {
            case 0:
               return angle.x;
            case 1:
               return angle.y;
            case 2:
               return angle.z;
            default:
               // we should never get here
               return 0;
            }
        }

    private:
        unsigned int member_idx; //!< Index of particle tag to get (0: a, 1: b)
    };

//! Helper kernel to get particle tag a, b or c, the angle type and the particle location (a b or c) from an angle
__global__ void gpu_kernel_angle_fill_values(const uint3 *angles,
                            const unsigned int *angle_types,
                            const unsigned int *d_rtag,
                            uint4 *values,
                            unsigned int member_idx,
                            unsigned int num_angles)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_angles)
        return;

    uint3 angle = angles[idx];
    unsigned int tag1, tag2;
    switch(member_idx)
        {
        case 0:
           tag1 = angle.y;
           tag2 = angle.z;
           break;
        case 1:
           tag1 = angle.x;
           tag2 = angle.z;
           break;
        case 2:
           tag1 = angle.x;
           tag2 = angle.y;
           break;
        default:
           // we should never get here
           tag1 = 0;
           tag2 = 0;
        }

    values[idx] = make_uint4(d_rtag[tag1], d_rtag[tag2], angle_types[idx], member_idx);
    };


//! Find the maximum number of angles per particle
/*! \param d_angles Array of angles
    \param d_angle_type Array of angle types
    \param num_angles Size of angle array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag -> particle index
    \param d_n_angles Number of angles per particle
    \param max_angle_num Maximum number of angles (return value)
    \param d_sort_keys Pointer to a temporary sorted list of first angle member indices (return value)
    \param d_sort_values Pointer to a temporary list of second angle member indices and angle types

    \pre Prior to calling this method, the internal angle_sort_keys and angle_sort_values
         need to be initialized by a call to gpu_find_max_angle_number
 */
cudaError_t TransformAngleDataGPU::gpu_find_max_angle_number(unsigned int& max_angle_num,
                                     uint3 *d_angles,
                                     unsigned int *d_angle_type,
                                     unsigned int num_angles,
                                     unsigned int N,
                                     unsigned int *d_rtag,
                                     unsigned int *d_n_angles)
    {
    assert(d_angles);
    assert(d_angle_type);
    assert(d_rtag);
    assert(d_n_angles);

    thrust::device_ptr<uint3> angles_ptr(d_angles);
    thrust::device_ptr<unsigned int> angle_type_ptr(d_angle_type);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<unsigned int> n_angles_ptr(d_n_angles);

    if (angle_sort_keys.size() < 3*num_angles)
        {
        angle_sort_keys.resize(3*num_angles);
        angle_sort_values.resize(3*num_angles);
        angle_indices.resize(3*num_angles);
        num_angles_sorted.resize(3*num_angles);
        }

    // idx a goes into angle_sort_keys
    thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(0))
                     ),
                 thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(0))
                     ) + num_angles,
                 angle_sort_keys.begin());

    // fill sort values
    unsigned int block_size = 512;
    uint4 *d_angle_sort_values =  thrust::raw_pointer_cast(&* angle_sort_values.begin());
    gpu_kernel_angle_fill_values<<<num_angles/block_size + 1, block_size>>>(d_angles,
                                                           d_angle_type,
                                                           d_rtag,
                                                           d_angle_sort_values,
                                                           0,
                                                           num_angles);

    // append idx b values to angle_sort_keys
    thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(1))
                     ),
                 thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(1))
                     ) + num_angles,
                 angle_sort_keys.begin() + num_angles);

    // fill sort values
    gpu_kernel_angle_fill_values<<<num_angles/block_size + 1, block_size>>>(d_angles,
                                                           d_angle_type,
                                                           d_rtag,
                                                           d_angle_sort_values + num_angles,
                                                           1,
                                                           num_angles);


    // append idx c values to angle_sort_keys
    thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(2))
                     ),
                 thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(angles_ptr, get_tag(2))
                     ) + num_angles,
                 angle_sort_keys.begin() + 2*num_angles);

    // fill sort values
    gpu_kernel_angle_fill_values<<<num_angles/block_size + 1, block_size>>>(d_angles,
                                                           d_angle_type,
                                                           d_rtag,
                                                           d_angle_sort_values + 2 *num_angles,
                                                           2,
                                                           num_angles);

    // sort first angle members as keys with second angle members and angle types as values
    thrust::sort_by_key(angle_sort_keys.begin(),
                 angle_sort_keys.begin() + 3 * num_angles,
                 angle_sort_values.begin());

    // count multiplicity of each key
    unsigned int n_unique_indices = thrust::reduce_by_key(angle_sort_keys.begin(),
                          angle_sort_keys.begin() + 3 * num_angles,
                          thrust::constant_iterator<unsigned int>(1),
                          angle_indices.begin(),
                          num_angles_sorted.begin() ).second - num_angles_sorted.begin();

    // find the maximum
    max_angle_num = thrust::reduce(num_angles_sorted.begin(),
                                  num_angles_sorted.begin() + n_unique_indices,
                                  0,
                                  thrust::maximum<unsigned int>());

    // fill n_angles array with zeros
    thrust::fill(n_angles_ptr,
                 n_angles_ptr + N,
                 0);

    // scatter angle numbers in n_angles array
    thrust::scatter(num_angles_sorted.begin(),
                    num_angles_sorted.begin() + n_unique_indices,
                    angle_indices.begin(),
                    n_angles_ptr);

    return cudaSuccess;
    }

//! Construct the GPU angle table
/*! \param num_angles Size of angle array
    \param d_gpu_angletable Pointer to the angle table on the GPU
    \param pitch Pitch of 2D angletable array
    \param d_sort_keys First angle members as keys (sorted)
    \param d_sort_values Second angle members as values (sorted)
 */
cudaError_t TransformAngleDataGPU::gpu_create_angletable(unsigned int num_angles,
                                     uint4 *d_gpu_angletable,
                                     unsigned int pitch)

    {

    thrust::device_ptr<uint4> gpu_angletable_ptr(d_gpu_angletable);

    if (angle_map.size() < 3*num_angles)
        {
        angle_map.resize(3*num_angles);
        }

    // create the angle_map of 2D angle table indices for all first angle members
    thrust::exclusive_scan_by_key(angle_sort_keys.begin(),
                                  angle_sort_keys.begin() + 3 * num_angles,
                                  thrust::make_constant_iterator(pitch),
                                  angle_map.begin());

    thrust::transform(angle_map.begin(),
                      angle_map.begin() + 3 * num_angles,
                      angle_sort_keys.begin(),
                      angle_map.begin(),
                      thrust::plus<unsigned int>());

    // scatter the second angle member into the 2D matrix according to the angle_map
    thrust::scatter(angle_sort_values.begin(),
                    angle_sort_values.begin() + 3* num_angles,
                    angle_map.begin(),
                    gpu_angletable_ptr);

    return cudaSuccess;
    }
