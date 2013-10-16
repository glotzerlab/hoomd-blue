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
#include "ParticleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef ENABLE_MPI
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#endif

/*! \file BondData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bond table
*/

//! Kernel to find the maximum number of angles per particle
__global__ void gpu_find_max_bond_number_kernel(const uint2 *bonds,
                                             const unsigned int *d_rtag,
                                             unsigned int *d_n_bonds,
                                             unsigned int num_bonds,
                                             unsigned int N,
                                             unsigned int n_ghosts,
                                             const unsigned int cur_max,
                                             unsigned int *condition)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    bool bond_needed = false;
    bool bond_valid = true;
    if (idx1 < N)
        {
        unsigned int n = atomicInc(&d_n_bonds[idx1], 0xffffffff);
        bond_valid &= (idx2 < N + n_ghosts);
        if (n >= cur_max) bond_needed = true;
        }
    if (idx2 < N)
        {
        unsigned int n = atomicInc(&d_n_bonds[idx2], 0xffffffff);
        bond_valid &= (idx1 < N + n_ghosts);
        if (n >= cur_max) bond_needed = true;
        }

    if (bond_needed)
        atomicOr(condition, 1);
    if (!bond_valid)
        atomicOr(condition, 2);
    }

//! Kernel to fill the GPU bond table
__global__ void gpu_fill_gpu_bond_table(const uint2 *bonds,
                                        const unsigned int *bond_type,
                                        uint2 *gpu_btable,
                                        const unsigned int pitch,
                                        const unsigned int *d_rtag,
                                        unsigned int *d_n_bonds,
                                        unsigned int num_bonds,
                                        unsigned int N)
    {
    int bond_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bond_idx >= num_bonds)
        return;

    uint2 bond = bonds[bond_idx];
    unsigned int tag1 = bond.x;
    unsigned int tag2 = bond.y;
    unsigned int type = bond_type[bond_idx];
    unsigned int idx1 = d_rtag[tag1];
    unsigned int idx2 = d_rtag[tag2];

    if (idx1 < N)
        {
        unsigned int num1 = atomicInc(&d_n_bonds[idx1],0xffffffff);
        gpu_btable[num1*pitch+idx1] = make_uint2(idx2,type);
        }
    if (idx2 < N)
        {
        unsigned int num2 = atomicInc(&d_n_bonds[idx2],0xffffffff);
        gpu_btable[num2*pitch+idx2] = make_uint2(idx1,type);
        }
    }


//! Find the maximum number of bonds per particle
/*! \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Array of bonds
    \param num_bonds Size of bond array
    \param N Number of particles in the system
    \param d_rtag Array of reverse-lookup particle tag . particle index
    \param cur_max Current maximum bonded particle number
    \param d_condition Condition variable, set to unequal zero if we exceed the maximum numbers
 */
cudaError_t gpu_find_max_bond_number(unsigned int *d_n_bonds,
                                     const uint2 *d_bonds,
                                     const unsigned int num_bonds,
                                     const unsigned int N,
                                     const unsigned int n_ghosts,
                                     const unsigned int *d_rtag,
                                     const unsigned int cur_max,
                                     unsigned int *d_condition)
    {
    assert(d_bonds);
    assert(d_rtag);
    assert(d_n_bonds);

    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_find_max_bond_number_kernel<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                              d_rtag,
                                                                              d_n_bonds,
                                                                              num_bonds,
                                                                              N,
                                                                              n_ghosts,
                                                                              cur_max,
                                                                              d_condition);

    return cudaSuccess;
    }

//! Construct the GPU bond table
/*! \param d_gpu_bondtable Pointer to the bond table on the GPU
    \param d_n_bonds Number of bonds per particle (return array)
    \param d_bonds Bonds array
    \param d_bond_type Array of bond types
    \param d_rtag Reverse-lookup tag->index
    \param num_bonds Number of bonds in bond list
    \param pitch Pitch of 2D bondtable array
    \param N Number of particles
 */
cudaError_t gpu_create_bondtable(uint2 *d_gpu_bondtable,
                                 unsigned int *d_n_bonds,
                                 const uint2 *d_bonds,
                                 const unsigned int *d_bond_type,
                                 const unsigned int *d_rtag,
                                 const unsigned int num_bonds,
                                 unsigned int pitch,
                                 unsigned int N)
    {
    unsigned int block_size = 512;

    // clear n_bonds array
    cudaMemset(d_n_bonds, 0, sizeof(unsigned int) * N);

    gpu_fill_gpu_bond_table<<<num_bonds/block_size + 1, block_size>>>(d_bonds,
                                                                      d_bond_type,
                                                                      d_gpu_bondtable,
                                                                      pitch,
                                                                      d_rtag,
                                                                      d_n_bonds,
                                                                      num_bonds,
                                                                      N);
    return cudaSuccess;
    }

#ifdef ENABLE_MPI
//! A predicate to select bond rtags by value (BOND_STAGED or BOND_SPLIT)
struct bond_rtag_select_send_gpu
    {
    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const unsigned int bond_rtag) const
        {
        return (bond_rtag == BOND_STAGED || bond_rtag == BOND_SPLIT);
        }
    };

//! A predicate to select bond rtags by value
struct bond_rtag_compare_gpu
    {
    unsigned int compare;  //!< Value to compare to

    //! Constructor
    bond_rtag_compare_gpu(unsigned int _compare)
        : compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const unsigned int bond_rtag) const
        {
        return (bond_rtag == compare);
        }
    };

/*! \param num_bonds Number of local bonds
    \param d_bond_tag Device array of bond tag
    \param d_bond_rtag Device array for the reverse-lookup of bond tags
 */
unsigned int gpu_bdata_count_rtag_staged(const unsigned int num_bonds,
    const unsigned int *d_bond_tag,
    const unsigned int *d_bond_rtag,
    cached_allocator& alloc)
    {
    thrust::device_ptr<const unsigned int> bond_tag_ptr(d_bond_tag);
    thrust::device_ptr<const unsigned int> bond_rtag_ptr(d_bond_rtag);

    // set up permutation iterator to point into rtags
    thrust::permutation_iterator<
        thrust::device_ptr<const unsigned int>, thrust::device_ptr<const unsigned int> >
        bond_rtag_prm(bond_rtag_ptr, bond_tag_ptr);

    return thrust::count_if(thrust::cuda::par(alloc),
        bond_rtag_prm, bond_rtag_prm + num_bonds, bond_rtag_select_send_gpu());
    }

//! A tuple of bond data pointers
typedef thrust::tuple <
    thrust::device_ptr<unsigned int>,  // tag
    thrust::device_ptr<uint2>,         // bond
    thrust::device_ptr<unsigned int>   // type
    > bdata_it_tuple_gpu;

//! A tuple of bond data pointers (const version)
typedef thrust::tuple <
    thrust::device_ptr<const unsigned int>,  // tag
    thrust::device_ptr<const uint2>,         // bond
    thrust::device_ptr<const unsigned int>   // type
    > bdata_it_tuple_gpu_const;

//! A zip iterator for filtering particle data
typedef thrust::zip_iterator<bdata_it_tuple_gpu> bdata_zip_gpu;

//! A zip iterator for filtering particle data (const version)
typedef thrust::zip_iterator<bdata_it_tuple_gpu_const> bdata_zip_gpu_const;

//! A tuple of bond data fields
typedef thrust::tuple <
    const unsigned int,  // tag
    const uint2,         // bond
    const unsigned int   // type
    > bdata_tuple_gpu;

//! A predicate to select bonds by rtag (BOND_STAGED or BOND_SPLIT)
struct bond_element_select_gpu : public thrust::unary_function<bond_element, bool>
    {
    //! Constructor
    bond_element_select_gpu(const unsigned int *_d_bond_rtag)
        : d_bond_rtag(_d_bond_rtag)
        { }

    //! Returns true if the send flag is set for a particle
    __device__ bool operator() (bond_element const b) const
        {
        unsigned int rtag = d_bond_rtag[b.tag];

        return (rtag == BOND_STAGED || rtag == BOND_SPLIT);
        }

    const unsigned int *d_bond_rtag; //!< The reverse-lookup tag array
    };

//! A converter from bond_element to a tuple of bond data entries
struct to_bdata_tuple_gpu : public thrust::unary_function<const bond_element, const bdata_tuple_gpu>
    {
    __device__ const bdata_tuple_gpu operator() (const bond_element b)
        {
        return thrust::make_tuple(
            b.tag,
            b.bond,
            b.type
            );
        }
    };

//! A converter from a tuple of bond entries to a bond_element
struct to_bond_element_gpu : public thrust::unary_function<const bdata_tuple_gpu,const bond_element>
    {
    __device__ const bond_element operator() (const bdata_tuple_gpu t)
        {
        bond_element b;

        b.tag = thrust::get<0>(t);
        b.bond = thrust::get<1>(t);
        b.type = thrust::get<2>(t);

        return b;
        }
    };

//! A predicate to select bond rtags by value
struct bdata_tuple_rtag_compare_gpu
    {
    unsigned int *d_bond_rtag; //!< Bond reverse-lookup table
    unsigned int compare;      //!< Value to compare to

    //! Constructor
    bdata_tuple_rtag_compare_gpu(unsigned int *_d_bond_rtag, unsigned int _compare)
        : d_bond_rtag(_d_bond_rtag), compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    __device__ bool operator() (const bdata_tuple_gpu t) const
        {
        unsigned int bond_tag = thrust::get<0>(t);
        return (d_bond_rtag[bond_tag] == compare);
        }
    };



//! Pack bonds on the GPU
/*! \param num_bonds Number of local bonds
    \param d_bond_tag Device array of bond tags
    \param d_bonds Device array of bonds (.x == particle a tag, .y == particle b tag)
    \param d_bond_type Device array of bond types
    \param d_bond_rtag Reverse-lookup table for bond tags
    \param d_out Device array for output (packed data)
 */
void gpu_pack_bonds(unsigned int num_bonds,
                    const unsigned int *d_bond_tag,
                    const uint2 *d_bonds,
                    const unsigned int *d_bond_type,
                    unsigned int *d_bond_rtag,
                    bond_element *d_out,
                    cached_allocator& alloc)
    {
    // wrap device arrays into thrust ptr
    thrust::device_ptr<const unsigned int> bond_tag_ptr(d_bond_tag);
    thrust::device_ptr<const uint2> bonds_ptr(d_bonds);
    thrust::device_ptr<const unsigned int> bond_type_ptr(d_bond_type);

    // wrap output array
    thrust::device_ptr<bond_element> out_ptr(d_out);

    // Construct zip iterator
    bdata_zip_gpu_const bdata_begin(
       thrust::make_tuple(
            bond_tag_ptr,
            bonds_ptr,
            bond_type_ptr
            )
        );

    // set up transform iterator to compact particle data into records
    thrust::transform_iterator<to_bond_element_gpu, bdata_zip_gpu_const> bdata_transform(
        bdata_begin,
        to_bond_element_gpu()
        );

    // compact selected particle elements into output array
    thrust::copy_if(thrust::cuda::par(alloc),
        bdata_transform, bdata_transform+num_bonds, out_ptr, bond_element_select_gpu(d_bond_rtag));

    // wrap bond rtag array
    thrust::device_ptr<unsigned int> bond_rtag_ptr(d_bond_rtag);

    // set up permutation iterator to point into rtags
    thrust::permutation_iterator<
        thrust::device_ptr<unsigned int>, thrust::device_ptr<const unsigned int> >
         bond_rtag_prm(bond_rtag_ptr, bond_tag_ptr);

    // set all BOND_STAGED tags to BOND_NOT_LOCAL
    thrust::replace_if(thrust::cuda::par(alloc),
        bond_rtag_prm, bond_rtag_prm + num_bonds, bond_rtag_compare_gpu(BOND_STAGED), BOND_NOT_LOCAL);
    }

//! A predicate to check if the bond doesn't already exist
struct bond_unique_gpu
    {
    const unsigned int *d_bond_rtag;      //!< Bond reverse-lookup table on device

    //! Constructor
    /*! \param _d_bond_rtag Pointer to reverse-lookup table
     */
    bond_unique_gpu(const unsigned int *_d_bond_rtag)
        : d_bond_rtag(_d_bond_rtag)
        { }

    //! Return true if bond is unique
    __device__ bool operator() (const bdata_tuple_gpu t) const
        {
        unsigned int bond_tag = t.get<0>();

        unsigned int bond_rtag = d_bond_rtag[bond_tag];
        return bond_rtag == BOND_NOT_LOCAL;
        }
    };

/*! \param num_bonds Current number of bonds
    \param num_add_bonds Number of bonds to be added
    \param d_bond_tag Device array of bond tags
    \param d_bonds Device array of bonds
    \param d_bond_type Device array of bond types
    \param d_in Device input array of packed bond data

    \returns new local number of bonds
 */
unsigned int gpu_bdata_add_remove_bonds(const unsigned int num_bonds,
                            const unsigned int num_add_bonds,
                            unsigned int *d_bond_tag,
                            uint2 *d_bonds,
                            unsigned int *d_bond_type,
                            unsigned int *d_bond_rtag,
                            const bond_element *d_in,
                            cached_allocator& alloc)
    {
    // wrap pointers into thrust ptrs
    thrust::device_ptr<unsigned int> bond_tag_ptr(d_bond_tag);
    thrust::device_ptr<uint2> bonds_ptr(d_bonds);
    thrust::device_ptr<unsigned int> bond_type_ptr(d_bond_type);

    // wrap reverse-lookup table
    thrust::device_ptr<unsigned int> bond_rtag_ptr(d_bond_rtag);

    bdata_zip_gpu bdata_begin(thrust::make_tuple(
        bond_tag_ptr,
        bonds_ptr,
        bond_type_ptr
        ));
    bdata_zip_gpu bdata_end = bdata_begin + num_bonds;

    // pointer from tag into rtag
    thrust::permutation_iterator<
        thrust::device_ptr<const unsigned int>, thrust::device_ptr<unsigned int> >
         bond_rtag_prm(bond_rtag_ptr, bond_tag_ptr);

    // erase all elements for which rtag == BOND_NOT_LOCAL
    // maintaing a contiguous array
    bdata_zip_gpu new_bdata_end;
    new_bdata_end = thrust::remove_if(thrust::cuda::par(alloc),
        bdata_begin, bdata_end, bdata_tuple_rtag_compare_gpu(d_bond_rtag, BOND_NOT_LOCAL));

    // wrap packed input data
    thrust::device_ptr<const bond_element> in_ptr(d_in);

    // set up a transform iterator from bond_element to bdata_tuple
    thrust::transform_iterator<to_bdata_tuple_gpu, thrust::device_ptr<const bond_element> > in_transform(
        in_ptr,
        to_bdata_tuple_gpu());

    // add new bonds at the end, omitting duplicates
    new_bdata_end = thrust::copy_if(thrust::cuda::par(alloc),
        in_transform, in_transform + num_add_bonds, new_bdata_end, bond_unique_gpu(d_bond_rtag));

    unsigned int new_n_bonds = new_bdata_end - bdata_begin;

    // recompute bond rtags
    thrust::counting_iterator<unsigned int> idx(0);
    thrust::scatter(thrust::cuda::par(alloc), idx, idx+new_n_bonds, bond_tag_ptr, bond_rtag_ptr);

    return new_n_bonds;
    }
#endif // ENABLE_MPI
