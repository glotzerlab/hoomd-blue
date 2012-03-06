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

#include "BondData.cuh"

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

/*! \file BondData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bond table
*/

//! Sorted array of the first bond member as key
thrust::device_vector<unsigned int> *sort_keys = NULL;
//! Sorted array of the second bond member and the bond type as value
thrust::device_vector<uint2> *sort_values = NULL;

//! Map of indices in the 2D GPU bond table for every first member of a bond
thrust::device_vector<unsigned int> *map = NULL;

//! Sorted list of number of bonds for each particle index
thrust::device_vector<unsigned int> *num_bonds_sorted = NULL;

//! Sorted list of particle indices that have at least one bond
thrust::device_vector<unsigned int> *bonded_indices = NULL;

void gpu_bonddata_allocate_scratch()
    {
    assert(sort_keys == NULL);
    assert(sort_values == NULL);
    assert(map == NULL);
    assert(bonded_indices == NULL);
    assert(num_bonds_sorted == NULL);

    sort_keys = new thrust::device_vector<unsigned int>();
    sort_values = new thrust::device_vector<uint2>();

    map = new thrust::device_vector<unsigned int>();

    bonded_indices= new thrust::device_vector<unsigned int>();
    num_bonds_sorted = new thrust::device_vector<unsigned int>();
    }

void gpu_bonddata_deallocate_scratch()
    {
    assert(sort_keys);
    assert(sort_values);
    assert(bonded_indices);
    assert(num_bonds_sorted);
    assert(map);



    delete sort_keys;
    delete sort_values;
    delete map;
    delete bonded_indices;
    delete num_bonds_sorted;

    sort_keys = NULL;
    sort_values = NULL;
    map = NULL;
    bonded_indices = NULL;
    num_bonds_sorted = NULL;
    }

//! Helper structure to get particle tag a or b from a bond
struct get_tag : thrust::unary_function<uint2, unsigned int>
    {
    //! Constructor
    get_tag(unsigned int _member_idx)
        : member_idx(_member_idx)
        { }

    //! Get particle tag
    __host__ __device__
    unsigned int operator ()(uint2 bond)
        {
        return (member_idx == 0) ? bond.x : bond.y;
        }

    private:
        unsigned int member_idx; //!< Index of particle tag to get (0: a, 1: b)
    };

//! Helper structure to get particle tag a or b and the bond type from a bond
struct get_idx_and_type : thrust::unary_function<thrust::tuple<uint2, unsigned int>, uint2>
    {
    //! Constructor
    get_idx_and_type(unsigned int *_d_rtag,
                       unsigned int _member_idx)
        : d_rtag(_d_rtag), member_idx(_member_idx) {}

    //! Get tag b and the bond type
    __device__
    uint2 operator ()(thrust::tuple<uint2, unsigned int> t)
        {
        uint2 b = thrust::get<0>(t);
        return make_uint2((member_idx == 0) ? d_rtag[b.x]: d_rtag[b.y],thrust::get<1>(t));
        }

    private:
        unsigned int *d_rtag; //!< Device pointer to rtag array
        unsigned int member_idx; //!< Index of particle tag to get (0: a, 1: b)

    };


//! Find the maximum number of bonds per particle
/*! \param d_bonds Array of bonds
    \param d_bond_type Array of bond types
    \param num_bonds Size of bond array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag -> particle index
    \param d_n_bonds Number of bonds per particle
    \param max_bond_num Maximum number of bonds (return value)
    \param d_sort_keys Pointer to a temporary sorted list of first bond member indices (return value)
    \param d_sort_values Pointer to a temporary list of second bond member indices and bond types
 */
cudaError_t gpu_find_max_bond_number(uint2 *d_bonds,
                                     unsigned int *d_bond_type,
                                     unsigned int num_bonds,
                                     unsigned int N,
                                     unsigned int *d_rtag,
                                     unsigned int *d_n_bonds,
                                     unsigned int& max_bond_num,
                                     unsigned int *& d_sort_keys,
                                     uint2 *& d_sort_values)
    {
    assert(d_bonds);
    assert(d_bond_type);
    assert(d_rtag);
    assert(d_n_bonds);

    thrust::device_ptr<uint2> bonds_ptr(d_bonds);
    thrust::device_ptr<unsigned int> bond_type_ptr(d_bond_type);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);
    thrust::device_ptr<unsigned int> n_bonds_ptr(d_n_bonds);

    if (sort_keys->size() < 2*num_bonds)
        {
        sort_keys->resize(2*num_bonds);
        sort_values->resize(2*num_bonds);
        bonded_indices->resize(2*num_bonds);
        num_bonds_sorted->resize(2*num_bonds);
        }

    // idx a goes into sort_keys
    thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(bonds_ptr, get_tag(0))
                     ),
                 thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(bonds_ptr, get_tag(0))
                     ) + num_bonds,
                 sort_keys->begin());

    // idx b and bond type goes into sort_values
    thrust::copy(
        thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(bonds_ptr, bond_type_ptr)),
            get_idx_and_type(d_rtag,1)),
        thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(bonds_ptr, bond_type_ptr)),
            get_idx_and_type(d_rtag,1)) + num_bonds,
        sort_values->begin());

    // append idx b values to sort_keys
    thrust::copy(thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(bonds_ptr, get_tag(1))
                     ),
                 thrust::make_permutation_iterator(
                     rtag_ptr,
                     thrust::make_transform_iterator(bonds_ptr, get_tag(1))
                     ) + num_bonds,
                 sort_keys->begin() + num_bonds);

    // append idx a and bond type to sort_values
    thrust::copy(
        thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(bonds_ptr, bond_type_ptr)),
            get_idx_and_type(d_rtag,0)),
        thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(bonds_ptr, bond_type_ptr)),
            get_idx_and_type(d_rtag,0)) + num_bonds,
        sort_values->begin() + num_bonds);


    // sort first bond members as keys with second bond members and bond types as values
    thrust::sort_by_key(sort_keys->begin(),
                 sort_keys->begin() + 2 * num_bonds,
                 sort_values->begin());

    // count multiplicity of each key
    unsigned int n_unique_indices = thrust::reduce_by_key(sort_keys->begin(),
                          sort_keys->begin() + 2 * num_bonds,
                          thrust::constant_iterator<unsigned int>(1),
                          bonded_indices->begin(),
                          num_bonds_sorted->begin() ).second - num_bonds_sorted->begin();

    // find the maximum
    max_bond_num = thrust::reduce(num_bonds_sorted->begin(),
                                  num_bonds_sorted->begin() + n_unique_indices,
                                  0,
                                  thrust::maximum<unsigned int>());

    // fill n_bonds array with zeros
    thrust::fill(n_bonds_ptr,
                 n_bonds_ptr + N,
                 0);

    // scatter bond numbers in n_bonds array
    thrust::scatter(num_bonds_sorted->begin(),
                    num_bonds_sorted->begin() + n_unique_indices,
                    bonded_indices->begin(),
                    n_bonds_ptr);

    d_sort_keys = thrust::raw_pointer_cast(&*sort_keys->begin());
    d_sort_values = thrust::raw_pointer_cast(&*sort_values->begin());
    return cudaSuccess;
    }

//! Construct the GPU bond table
/*! \param num_bonds Size of bond array
    \param d_gpu_bondtable Pointer to the bond table on the GPU
    \param pitch Pitch of 2D bondtable array
    \param d_sort_keys First bond members as keys (sorted)
    \param d_sort_values Second bond members as values (sorted)
 */
cudaError_t gpu_create_bondtable(unsigned int num_bonds,
                                     uint2 *d_gpu_bondtable,
                                     unsigned int pitch,
                                     unsigned int * d_sort_keys,
                                     uint2 *d_sort_values )

    {

    thrust::device_ptr<uint2> gpu_bondtable_ptr(d_gpu_bondtable);

    thrust::device_ptr<unsigned int> sort_keys_ptr(d_sort_keys);
    thrust::device_ptr<uint2> sort_values_ptr(d_sort_values);

    if (map->size() < 2*num_bonds)
        {
        map->resize(2*num_bonds);
        }

    // create the map of 2D bond table indices for all first bond members
    thrust::exclusive_scan_by_key(sort_keys_ptr,
                                  sort_keys_ptr + 2 * num_bonds,
                                  thrust::make_constant_iterator(pitch),
                                  map->begin());

    thrust::transform(map->begin(),
                      map->begin() + 2 * num_bonds,
                      sort_keys_ptr,
                      map->begin(),
                      thrust::plus<unsigned int>());

    // scatter the second bond member into the 2D matrix according to the map
    thrust::scatter(sort_values_ptr,
                    sort_values_ptr + 2* num_bonds,
                    map->begin(),
                    gpu_bondtable_ptr);

    return cudaSuccess;
    }
