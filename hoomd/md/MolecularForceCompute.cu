// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MolecularForceCompute.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#pragma GCC diagnostic pop

#include <hipcub/hipcub.hpp>

#include <exception>
#include <string>
#define CHECK_CUDA()                                                                       \
        {                                                                                  \
        hipError_t err = hipDeviceSynchronize();                                           \
        if (err != hipSuccess)                                                             \
            {                                                                              \
            throw std::runtime_error("CUDA error in MolecularForceCompute "                \
                                     + std::string(hipGetErrorString(err)));               \
            }                                                                              \
        err = hipGetLastError();                                                           \
        if (err != hipSuccess)                                                             \
            {                                                                              \
            throw std::runtime_error("CUDA error " + std::string(hipGetErrorString(err))); \
            }                                                                              \
        }

/*! \file MolecularForceCompute.cu
    \brief Contains GPU kernel code used by MolecularForceCompute
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Sort local molecules and assign local molecule indices to particles
hipError_t gpu_sort_by_molecule(unsigned int nptl,
                                const unsigned int* d_tag,
                                const unsigned int* d_molecule_tag,
                                unsigned int* d_local_molecule_tags,
                                unsigned int* d_local_molecules_lowest_idx,
                                unsigned int* d_local_unique_molecule_tags,
                                unsigned int* d_local_molecule_idx,
                                unsigned int* d_sorted_by_tag,
                                unsigned int* d_idx_sorted_by_tag,
                                unsigned int* d_idx_sorted_by_molecule_and_tag,
                                unsigned int* d_lowest_idx,
                                unsigned int* d_lowest_idx_sort,
                                unsigned int* d_lowest_idx_in_molecules,
                                unsigned int* d_lowest_idx_by_molecule_tag,
                                unsigned int* d_molecule_length,
                                unsigned int& n_local_molecules,
                                unsigned int& max_len,
                                unsigned int& n_local_ptls_in_molecules,
                                CachedAllocator& alloc,
                                bool check_cuda)
    {
    thrust::device_ptr<const unsigned int> tag(d_tag);
    thrust::device_ptr<const unsigned int> molecule_tag(d_molecule_tag);
    thrust::device_ptr<unsigned int> local_molecule_tags(d_local_molecule_tags);
    thrust::device_ptr<unsigned int> local_unique_molecule_tags(d_local_unique_molecule_tags);
    thrust::device_ptr<unsigned int> local_molecule_idx(d_local_molecule_idx);
    thrust::device_ptr<unsigned int> sorted_by_tag(d_sorted_by_tag);
    thrust::device_ptr<unsigned int> idx_sorted_by_tag(d_idx_sorted_by_tag);
    thrust::device_ptr<unsigned int> molecule_length(d_molecule_length);

    // get temp allocations
    unsigned int* d_molecule_length_tmp = alloc.getTemporaryBuffer<unsigned int>(nptl);
    unsigned int* d_local_unique_molecule_tags_tmp = alloc.getTemporaryBuffer<unsigned int>(nptl);

    // sort local particles by tag

    // store ascending index in temp buffer
    unsigned int* d_idx = alloc.getTemporaryBuffer<unsigned int>(nptl);
    thrust::device_ptr<unsigned int> idx(d_idx);
    auto iter = thrust::counting_iterator<unsigned int>(0);
    thrust::copy(iter, iter + nptl, idx);

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_tag,
                                       d_sorted_by_tag,
                                       d_idx,
                                       d_idx_sorted_by_tag,
                                       nptl);
    d_temp_storage = alloc.allocate(temp_storage_bytes);

    // key-value sort
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_tag,
                                       d_sorted_by_tag,
                                       d_idx,
                                       d_idx_sorted_by_tag,
                                       nptl);
    alloc.deallocate((char*)d_temp_storage);

    // release temp buffer
    alloc.deallocate((char*)d_idx);

    unsigned int* d_num_runs_out = (unsigned int*)alloc.allocate(sizeof(unsigned int));
    auto molecule_tag_lookup = thrust::make_permutation_iterator(molecule_tag, tag);
    auto molecule_tag_lookup_sorted_by_tag
        = thrust::make_permutation_iterator(molecule_tag_lookup, idx_sorted_by_tag);

    // get temp buffers
    unsigned int* d_molecule_by_idx = alloc.getTemporaryBuffer<unsigned int>(nptl);
    thrust::device_ptr<unsigned int> molecule_by_idx(d_molecule_by_idx);

#ifdef __HIP_PLATFORM_HCC__
    thrust::copy(thrust::hip::par(alloc),
#else
    thrust::copy(thrust::cuda::par(alloc),
#endif
                 molecule_tag_lookup_sorted_by_tag,
                 molecule_tag_lookup_sorted_by_tag + nptl,
                 molecule_by_idx);
    if (check_cuda)
        CHECK_CUDA();

    // sort local particle indices by global molecule tag, keeping tag order (radix sort is stable)
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_molecule_by_idx,
                                       d_local_molecule_tags,
                                       d_idx_sorted_by_tag,
                                       d_idx_sorted_by_molecule_and_tag,
                                       nptl);
    d_temp_storage = alloc.allocate(temp_storage_bytes);

    // key-value sort
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_molecule_by_idx,
                                       d_local_molecule_tags,
                                       d_idx_sorted_by_tag,
                                       d_idx_sorted_by_molecule_and_tag,
                                       nptl);
    alloc.deallocate((char*)d_temp_storage);

    // release temp buffer
    alloc.deallocate((char*)d_molecule_by_idx);

    // find the end of the molecule list
    auto end = thrust::lower_bound(local_molecule_tags, local_molecule_tags + nptl, NO_MOLECULE);
    if (check_cuda)
        CHECK_CUDA();

    n_local_ptls_in_molecules = (unsigned int)(end - local_molecule_tags);

    // gather unique molecule tags, and reduce their lengths by key
    thrust::constant_iterator<unsigned int> one(1);

    // determine temporary storage
    d_temp_storage = NULL;
    temp_storage_bytes = 0;

    hipcub::DeviceReduce::ReduceByKey(d_temp_storage,
                                      temp_storage_bytes,
                                      d_local_molecule_tags,
                                      d_local_unique_molecule_tags_tmp,
                                      one,
                                      d_molecule_length_tmp,
                                      d_num_runs_out,
                                      thrust::plus<unsigned int>(),
                                      n_local_ptls_in_molecules);

    d_temp_storage = alloc.allocate(temp_storage_bytes);

    hipcub::DeviceReduce::ReduceByKey(d_temp_storage,
                                      temp_storage_bytes,
                                      d_local_molecule_tags,
                                      d_local_unique_molecule_tags_tmp,
                                      one,
                                      d_molecule_length_tmp,
                                      d_num_runs_out,
                                      thrust::plus<unsigned int>(),
                                      n_local_ptls_in_molecules);

    hipMemcpy(&n_local_molecules, d_num_runs_out, sizeof(unsigned int), hipMemcpyDeviceToHost);
    if (check_cuda)
        CHECK_CUDA();

    alloc.deallocate((char*)d_temp_storage);
    alloc.deallocate((char*)d_num_runs_out);

    // find the index of the particle with lowest tag in every molecule
    thrust::device_ptr<unsigned int> lowest_idx_in_molecules(d_lowest_idx_in_molecules);
    thrust::device_ptr<unsigned int> lowest_idx(d_lowest_idx);

    thrust::device_ptr<unsigned int> local_unique_molecule_tags_tmp(
        d_local_unique_molecule_tags_tmp);
    thrust::lower_bound(local_molecule_tags,
                        local_molecule_tags + n_local_ptls_in_molecules,
                        local_unique_molecule_tags_tmp,
                        local_unique_molecule_tags_tmp + n_local_molecules,
                        lowest_idx_in_molecules);
    if (check_cuda)
        CHECK_CUDA();

    thrust::device_ptr<unsigned int> idx_sorted_by_molecule_and_tag(
        d_idx_sorted_by_molecule_and_tag);
    thrust::gather(lowest_idx_in_molecules,
                   lowest_idx_in_molecules + n_local_molecules,
                   idx_sorted_by_molecule_and_tag,
                   lowest_idx);
    if (check_cuda)
        CHECK_CUDA();

    // compute maximum molecule length
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_max = (unsigned int*)alloc.allocate(sizeof(unsigned int));
    hipcub::DeviceReduce::Max(d_temp_storage,
                              temp_storage_bytes,
                              d_molecule_length_tmp,
                              d_max,
                              n_local_molecules);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Max(d_temp_storage,
                              temp_storage_bytes,
                              d_molecule_length_tmp,
                              d_max,
                              n_local_molecules);
    alloc.deallocate((char*)d_temp_storage);
    hipMemcpy(&max_len, d_max, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_max);

    if (check_cuda)
        CHECK_CUDA();

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_lowest_idx,
                                       d_lowest_idx_sort,
                                       d_local_unique_molecule_tags_tmp,
                                       d_local_unique_molecule_tags,
                                       n_local_molecules);
    d_temp_storage = alloc.allocate(temp_storage_bytes);

    // key-value sort
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_lowest_idx,
                                       d_lowest_idx_sort,
                                       d_local_unique_molecule_tags_tmp,
                                       d_local_unique_molecule_tags,
                                       n_local_molecules);
    alloc.deallocate((char*)d_temp_storage);

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_lowest_idx,
                                       d_lowest_idx_sort,
                                       d_molecule_length_tmp,
                                       d_molecule_length,
                                       n_local_molecules);
    d_temp_storage = alloc.allocate(temp_storage_bytes);

    // key-value sort
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_lowest_idx,
                                       d_lowest_idx_sort,
                                       d_molecule_length_tmp,
                                       d_molecule_length,
                                       n_local_molecules);
    alloc.deallocate((char*)d_temp_storage);

    // release temp buffers
    alloc.deallocate((char*)d_molecule_length_tmp);
    alloc.deallocate((char*)d_local_unique_molecule_tags_tmp);

    // create a global lookup table for lowest idx by molecule tag
    thrust::device_ptr<unsigned int> lowest_idx_by_molecule_tag(d_lowest_idx_by_molecule_tag);
    thrust::device_ptr<unsigned int> lowest_idx_sort(d_lowest_idx_sort);
    thrust::scatter(lowest_idx_sort,
                    lowest_idx_sort + n_local_molecules,
                    local_unique_molecule_tags,
                    lowest_idx_by_molecule_tag);
    if (check_cuda)
        CHECK_CUDA();

    // sort the list of particles in molecules again according to first particle index, keeping
    // order in molecule
    auto lowest_idx_by_ptl_in_molecule
        = thrust::make_permutation_iterator(lowest_idx_by_molecule_tag, local_molecule_tags);
    if (check_cuda)
        CHECK_CUDA();

    // get temp buffer
    unsigned int* d_local_molecules_lowest_idx_unsorted
        = alloc.getTemporaryBuffer<unsigned int>(n_local_ptls_in_molecules);

    thrust::device_ptr<unsigned int> local_molecules_lowest_idx_unsorted(
        d_local_molecules_lowest_idx_unsorted);
#ifdef __HIP_PLATFORM_HCC__
    thrust::copy(thrust::hip::par(alloc),
#else
    thrust::copy(thrust::cuda::par(alloc),
#endif
                 lowest_idx_by_ptl_in_molecule,
                 lowest_idx_by_ptl_in_molecule + n_local_ptls_in_molecules,
                 local_molecules_lowest_idx_unsorted);
    if (check_cuda)
        CHECK_CUDA();

    // radix sort is stable
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_local_molecules_lowest_idx_unsorted,
                                       d_local_molecules_lowest_idx,
                                       d_idx_sorted_by_molecule_and_tag,
                                       d_idx_sorted_by_tag,
                                       n_local_ptls_in_molecules);
    d_temp_storage = alloc.allocate(temp_storage_bytes);

    hipcub::DeviceRadixSort::SortPairs(d_temp_storage,
                                       temp_storage_bytes,
                                       d_local_molecules_lowest_idx_unsorted,
                                       d_local_molecules_lowest_idx,
                                       d_idx_sorted_by_molecule_and_tag,
                                       d_idx_sorted_by_tag,
                                       n_local_ptls_in_molecules);
    alloc.deallocate((char*)d_temp_storage);

    // release temp buffer
    alloc.deallocate((char*)d_local_molecules_lowest_idx_unsorted);

    // assign local molecule tags to particles
    thrust::fill(local_molecule_idx, local_molecule_idx + nptl, NO_MOLECULE);

    auto idx_lookup = thrust::make_permutation_iterator(local_molecule_idx, idx_sorted_by_tag);
    thrust::device_ptr<unsigned int> local_molecules_lowest_idx(d_local_molecules_lowest_idx);
    thrust::lower_bound(lowest_idx_sort,
                        lowest_idx_sort + n_local_molecules,
                        local_molecules_lowest_idx,
                        local_molecules_lowest_idx + n_local_ptls_in_molecules,
                        idx_lookup);
    if (check_cuda)
        CHECK_CUDA();

    return hipSuccess;
    }

__global__ void gpu_fill_molecule_table_kernel(unsigned int nptl,
                                               Index2D molecule_idx,
                                               const unsigned int* d_molecule_idx,
                                               unsigned int* d_molecule_list,
                                               unsigned int* d_molecule_order)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nptl)
        return;

    unsigned int molidx = d_molecule_idx[idx];
    if (molidx != NO_MOLECULE)
        d_molecule_list[molecule_idx(d_molecule_order[idx], molidx)] = idx;
    }

hipError_t gpu_fill_molecule_table(unsigned int nptl,
                                   unsigned int n_local_ptls_in_molecules,
                                   Index2D molecule_idx,
                                   const unsigned int* d_molecule_idx,
                                   const unsigned int* d_local_molecule_tags,
                                   const unsigned int* d_idx_sorted_by_tag,
                                   unsigned int* d_molecule_list,
                                   unsigned int* d_molecule_order,
                                   unsigned int block_size,
                                   CachedAllocator& alloc)
    {
    thrust::device_ptr<unsigned int> molecule_order(d_molecule_order);
    thrust::device_ptr<const unsigned int> local_molecule_tags(d_local_molecule_tags);
    thrust::device_ptr<const unsigned int> idx_sorted_by_tag(d_idx_sorted_by_tag);

    auto idx_lookup = thrust::make_permutation_iterator(molecule_order, idx_sorted_by_tag);

    // generate ascending index for every molecule
    thrust::constant_iterator<unsigned int> one(1);

#ifdef __HIP_PLATFORM_HCC__
    thrust::exclusive_scan_by_key(thrust::hip::par(alloc),
#else
    thrust::exclusive_scan_by_key(thrust::cuda::par(alloc),
#endif
                                  local_molecule_tags,
                                  local_molecule_tags + n_local_ptls_in_molecules,
                                  one,
                                  idx_lookup);

    // write out the table
    hipLaunchKernelGGL((gpu_fill_molecule_table_kernel),
                       dim3(nptl / block_size + 1),
                       dim3(block_size),
                       0,
                       0,
                       nptl,
                       molecule_idx,
                       d_molecule_idx,
                       d_molecule_list,
                       d_molecule_order);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
