// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "MolecularForceCompute.cuh"

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

/*! \file MolecularForceCompute.cu
    \brief Contains GPU kernel code used by MolecularForceCompute
*/

//! Sort local molecules and assign local molecule indices to particles
cudaError_t gpu_sort_by_molecule(unsigned int nptl,
    const unsigned int *d_tag,
    const unsigned int *d_molecule_tag,
    unsigned int *d_local_molecule_tags,
    unsigned int *d_local_molecules_lowest_idx,
    unsigned int *d_local_unique_molecule_tags,
    unsigned int *d_local_molecule_idx,
    unsigned int *d_sorted_by_tag,
    unsigned int *d_idx_sorted_by_tag,
    unsigned int *d_lowest_idx,
    unsigned int *d_lowest_idx_in_molecules,
    unsigned int *d_lowest_idx_by_molecule_tag,
    unsigned int *d_molecule_length,
    unsigned int &n_local_molecules,
    unsigned int &max_len,
    unsigned int &n_local_ptls_in_molecules,
    const CachedAllocator& alloc)
    {
    thrust::device_ptr<const unsigned int> tag(d_tag);
    thrust::device_ptr<const unsigned int> molecule_tag(d_molecule_tag);
    thrust::device_ptr<unsigned int> local_molecule_tags(d_local_molecule_tags);
    thrust::device_ptr<unsigned int> local_unique_molecule_tags(d_local_unique_molecule_tags);
    thrust::device_ptr<unsigned int> local_molecule_idx(d_local_molecule_idx);
    thrust::device_ptr<unsigned int> sorted_by_tag(d_sorted_by_tag);
    thrust::device_ptr<unsigned int> idx_sorted_by_tag(d_idx_sorted_by_tag);
    thrust::device_ptr<unsigned int> molecule_length(d_molecule_length);

    // sort local particles by tag
    thrust::copy(tag,tag+nptl,sorted_by_tag);

    auto iter = thrust::counting_iterator<unsigned int>(0);
    thrust::copy(iter,
        iter+nptl,
        idx_sorted_by_tag);

    thrust::sort_by_key(thrust::cuda::par(alloc),
        sorted_by_tag,
        sorted_by_tag+nptl,
        idx_sorted_by_tag);

    auto molecule_tag_lookup = thrust::make_permutation_iterator(molecule_tag, tag);
    auto molecule_tag_lookup_sorted_by_tag = thrust::make_permutation_iterator(molecule_tag_lookup, idx_sorted_by_tag);

    thrust::copy(molecule_tag_lookup_sorted_by_tag,
        molecule_tag_lookup_sorted_by_tag+nptl,
        local_molecule_tags);

    // sort local particle indices by global molecule tag, keeping tag order
    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
        local_molecule_tags,
        local_molecule_tags + nptl,
        idx_sorted_by_tag);

    // find the end of the molecule list
    auto end = thrust::lower_bound(thrust::cuda::par,
        local_molecule_tags,
        local_molecule_tags + nptl,
        NO_MOLECULE);

    n_local_ptls_in_molecules = end - local_molecule_tags;

    // gather unique molecule tags, and reduce their lengths by key
    thrust::constant_iterator<unsigned int> one(1);

    #if (CUDART_VERSION < 8000)
    // work around CUDA 7.5 bug
    // https://devtalk.nvidia.com/default/topic/900103/thrust-reduce_by_key-issues-with-maxwell-devices/

    // allocate a temporary vector
    thrust::device_vector<unsigned int> local_molecule_tags_vec(nptl);
    thrust::copy(thrust::cuda::par(alloc),
        local_molecule_tags,
        local_molecule_tags + nptl,
        local_molecule_tags_vec.begin());

    auto new_end = thrust::reduce_by_key(thrust::cuda::par(alloc),
        local_molecule_tags_vec.begin(),
        local_molecule_tags_vec.begin() + n_local_ptls_in_molecules,
        one,
        local_unique_molecule_tags,
        molecule_length
        );
    #else
    auto new_end = thrust::reduce_by_key(thrust::cuda::par(alloc),
        local_molecule_tags,
        end,
        one,
        local_unique_molecule_tags,
        molecule_length
        );
    #endif
    n_local_molecules = new_end.first - local_unique_molecule_tags;

    // find the index of the particle with lowest tag in every molecule
    thrust::device_ptr<unsigned int> lowest_idx_in_molecules(d_lowest_idx_in_molecules);
    thrust::device_ptr<unsigned int> lowest_idx(d_lowest_idx);

    thrust::lower_bound(thrust::cuda::par(alloc),
        local_molecule_tags,
        local_molecule_tags + n_local_ptls_in_molecules,
        local_unique_molecule_tags,
        local_unique_molecule_tags + n_local_molecules,
        lowest_idx_in_molecules);

    thrust::gather(thrust::cuda::par(alloc),
        lowest_idx_in_molecules,
        lowest_idx_in_molecules + n_local_molecules,
        idx_sorted_by_tag,
        lowest_idx);

    // compute maximum molecule length
    thrust::device_ptr<unsigned int> max_ptr = thrust::max_element(molecule_length, molecule_length + n_local_molecules);
    cudaMemcpy(&max_len, max_ptr.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost);

    auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(local_unique_molecule_tags, molecule_length));
    thrust::sort_by_key(thrust::cuda::par(alloc),
        lowest_idx,
        lowest_idx + n_local_molecules,
        zip_it);

    // create a global lookup table for lowest idx by molecule tag
    thrust::device_ptr<unsigned int> lowest_idx_by_molecule_tag(d_lowest_idx_by_molecule_tag);
    thrust::scatter(thrust::cuda::par(alloc),
        lowest_idx,
        lowest_idx + n_local_molecules,
        local_unique_molecule_tags,
        lowest_idx_by_molecule_tag);

    // sort the list of particles in molecules again according to first particle index, keeping order in molecule
    auto lowest_idx_by_ptl_in_molecule = thrust::make_permutation_iterator(
        lowest_idx_by_molecule_tag,
        local_molecule_tags);

    thrust::device_ptr<unsigned int> local_molecules_lowest_idx(d_local_molecules_lowest_idx);
    thrust::copy(thrust::cuda::par(alloc),
        lowest_idx_by_ptl_in_molecule,
        lowest_idx_by_ptl_in_molecule + n_local_ptls_in_molecules,
        local_molecules_lowest_idx);

    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
        local_molecules_lowest_idx,
        local_molecules_lowest_idx + n_local_ptls_in_molecules,
        idx_sorted_by_tag);

    // assign local molecule tags to particles
    thrust::fill(thrust::cuda::par(alloc),
        local_molecule_idx, local_molecule_idx+nptl,NO_MOLECULE);
    auto idx_lookup = thrust::make_permutation_iterator(local_molecule_idx, idx_sorted_by_tag);
    thrust::lower_bound(thrust::cuda::par(alloc),
        lowest_idx,
        lowest_idx + n_local_molecules,
        local_molecules_lowest_idx,
        local_molecules_lowest_idx + n_local_ptls_in_molecules,
        idx_lookup);

    return cudaSuccess;
    }

__global__ void gpu_fill_molecule_table_kernel(
    unsigned int nptl,
    Index2D molecule_idx,
    const unsigned int *d_molecule_idx,
    unsigned int *d_molecule_list,
    unsigned int *d_molecule_order)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx >= nptl) return;

    unsigned int molidx = d_molecule_idx[idx];
    if (molidx != NO_MOLECULE)
        d_molecule_list[molecule_idx(d_molecule_order[idx], molidx)] = idx;
    }

cudaError_t gpu_fill_molecule_table(
    unsigned int nptl,
    unsigned int n_local_ptls_in_molecules,
    Index2D molecule_idx,
    const unsigned int *d_molecule_idx,
    const unsigned int *d_local_molecule_tags,
    const unsigned int *d_idx_sorted_by_tag,
    unsigned int *d_molecule_list,
    unsigned int *d_molecule_order,
    unsigned int block_size,
    const CachedAllocator& alloc
    )
    {
    thrust::device_ptr<unsigned int> molecule_order(d_molecule_order);
    thrust::device_ptr<const unsigned int> local_molecule_tags(d_local_molecule_tags);
    thrust::device_ptr<const unsigned int> idx_sorted_by_tag(d_idx_sorted_by_tag);

    auto idx_lookup = thrust::make_permutation_iterator(molecule_order, idx_sorted_by_tag);

    // generate ascending index for every molecule
    thrust::constant_iterator<unsigned int> one(1);
    thrust::exclusive_scan_by_key(thrust::cuda::par(alloc),
        local_molecule_tags,
        local_molecule_tags+n_local_ptls_in_molecules,
        one,
        idx_lookup);

    // write out the table
    gpu_fill_molecule_table_kernel<<<nptl/block_size+1,block_size>>>(
        nptl,
        molecule_idx,
        d_molecule_idx,
        d_molecule_list,
        d_molecule_order);

    return cudaSuccess;
    }
