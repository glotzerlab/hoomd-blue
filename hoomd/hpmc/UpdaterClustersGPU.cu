// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Moves.h"
#include "UpdaterClustersGPU.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#pragma GCC diagnostic pop

#ifdef __HIP_PLATFORM_NVCC__
#include <cusparse.h>
#endif

#include "hoomd/extern/ECL.cuh"

/*! \file UpdaterClustersGPU.cu
    \brief Implements a connected components algorithm on the GPU
*/

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
#ifdef __HIP_PLATFORM_NVCC__
#define check_cusparse(a)                                                                 \
        {                                                                                 \
        cusparseStatus_t status = (a);                                                    \
        if ((int)status != CUSPARSE_STATUS_SUCCESS)                                       \
            {                                                                             \
            printf("cusparse ERROR %d in file %s line %d\n", status, __FILE__, __LINE__); \
            throw std::runtime_error("Error during clusters update");                     \
            }                                                                             \
        }
#endif

struct get_source : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__ unsigned int operator()(const uint2& u) const
        {
        return u.x;
        }
    };

struct get_destination : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__ unsigned int operator()(const uint2& u) const
        {
        return u.y;
        }
    };

struct pair_less : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2& lhs, const uint2& rhs) const
        {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
        }
    };

void __attribute__((visibility("default"))) get_num_neighbors(const unsigned int* d_nneigh,
                                                              unsigned int* d_nneigh_scan,
                                                              unsigned int& nneigh_total,
                                                              const GPUPartition& gpu_partition,
                                                              CachedAllocator& alloc)
    {
    assert(d_nneigh);
    thrust::device_ptr<const unsigned int> nneigh(d_nneigh);
    thrust::device_ptr<unsigned int> nneigh_scan(d_nneigh_scan);

    nneigh_total = 0;
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

#ifdef __HIP_PLATFORM_HCC__
        thrust::exclusive_scan(thrust::hip::par(alloc),
#else
        thrust::exclusive_scan(thrust::cuda::par(alloc),
#endif
                               nneigh + range.first,
                               nneigh + range.second,
                               nneigh_scan + range.first,
                               nneigh_total);

#ifdef __HIP_PLATFORM_HCC__
        nneigh_total += thrust::reduce(thrust::hip::par(alloc),
#else
        nneigh_total += thrust::reduce(thrust::cuda::par(alloc),
#endif
                                       nneigh + range.first,
                                       nneigh + range.second,
                                       0,
                                       thrust::plus<unsigned int>());
        }
    }

namespace kernel
    {
__global__ void concatenate_adjacency_list(const unsigned int* d_adjacency,
                                           const unsigned int* d_nneigh,
                                           const unsigned int* d_nneigh_scan,
                                           const unsigned int maxn,
                                           uint2* d_adjacency_out,
                                           const unsigned int nwork,
                                           const unsigned int work_offset)
    {
    // one group per particle to copy over neighbors
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    unsigned int n_groups = blockDim.y;

    unsigned int i = blockIdx.x * n_groups + group;
    if (i >= nwork)
        return;
    i += work_offset;

    unsigned int nneigh = d_nneigh[i];
    unsigned int start = d_nneigh_scan[i];
    for (unsigned int k = 0; k < nneigh; k += group_size)
        {
        if (k + offset < nneigh)
            {
            unsigned int j = d_adjacency[k + offset + i * maxn];

            // we make the matrix explicitly symmetric, because
            // inserting symmetric pairs in the overlap kernels would violate memory locality
            d_adjacency_out[2 * (start + k + offset)] = make_uint2(i, j);
            d_adjacency_out[2 * (start + k + offset) + 1] = make_uint2(j, i);
            }
        }
    }

__global__ void flip_clusters(Scalar4* d_postype,
                              Scalar4* d_orientation,
                              int3* d_image,
                              const Scalar4* d_postype_backup,
                              const Scalar4* d_orientation_backup,
                              const int3* d_image_backup,
                              const int* d_components,
                              float flip_probability,
                              uint16_t seed,
                              uint64_t timestep,
                              unsigned int nwork,
                              unsigned int work_offset)
    {
    unsigned int work_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (work_idx >= nwork)
        return;

    unsigned int i = work_idx + work_offset;

    // seed by cluster id
    int component = d_components[i];
    hoomd::RandomGenerator rng_i(
        hoomd::Seed(hoomd::RNGIdentifier::UpdaterClusters2, timestep, seed),
        hoomd::Counter(component));

    bool flip = hoomd::detail::generate_canonical<float>(rng_i) <= flip_probability;

    if (!flip)
        {
        d_postype[i] = d_postype_backup[i];
        d_orientation[i] = d_orientation_backup[i];
        d_image[i] = d_image_backup[i];
        }
    }

    } // end namespace kernel

void __attribute__((visibility("default")))
concatenate_adjacency_list(const unsigned int* d_adjacency,
                           const unsigned int* d_nneigh,
                           const unsigned int* d_nneigh_scan,
                           const unsigned int maxn,
                           uint2* d_adjacency_out,
                           const GPUPartition& gpu_partition,
                           const unsigned int block_size,
                           const unsigned int group_size)
    {
    // determine the maximum block size and clamp the input block size down
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::concatenate_adjacency_list));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    // threads per particle
    unsigned int cur_group_size = min(run_block_size, group_size);
    while (run_block_size % cur_group_size != 0)
        cur_group_size--;

    unsigned int n_groups = run_block_size / cur_group_size;
    dim3 threads(cur_group_size, n_groups, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork / n_groups + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::concatenate_adjacency_list,
                           grid,
                           threads,
                           0,
                           0,
                           d_adjacency,
                           d_nneigh,
                           d_nneigh_scan,
                           maxn,
                           d_adjacency_out,
                           nwork,
                           range.first);
        }
    }

void __attribute__((visibility("default"))) flip_clusters(Scalar4* d_postype,
                                                          Scalar4* d_orientation,
                                                          int3* d_image,
                                                          const Scalar4* d_postype_backup,
                                                          const Scalar4* d_orientation_backup,
                                                          const int3* d_image_backup,
                                                          const int* d_components,
                                                          float flip_probability,
                                                          uint16_t seed,
                                                          uint64_t timestep,
                                                          const GPUPartition& gpu_partition,
                                                          const unsigned int block_size)
    {
    // determine the maximum block size and clamp the input block size down
    int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::flip_clusters));
    max_block_size = attr.maxThreadsPerBlock;

    // setup the grid to run the kernel
    unsigned int run_block_size = min(block_size, (unsigned int)max_block_size);

    dim3 threads(run_block_size, 1);

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = nwork / run_block_size + 1;
        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL(kernel::flip_clusters,
                           grid,
                           threads,
                           0,
                           0,
                           d_postype,
                           d_orientation,
                           d_image,
                           d_postype_backup,
                           d_orientation_backup,
                           d_image_backup,
                           d_components,
                           flip_probability,
                           seed,
                           timestep,
                           nwork,
                           range.first);
        }
    }

void connected_components(uint2* d_adj,
                          unsigned int N,
                          const unsigned int n_elements,
                          int* d_components,
                          unsigned int& num_components,
                          const hipDeviceProp_t& dev_prop,
                          CachedAllocator& alloc)
    {
#ifdef __HIP_PLATFORM_NVCC__
    thrust::device_ptr<uint2> adj(d_adj);

    // sort the list of pairs
    thrust::sort(thrust::cuda::par(alloc), adj, adj + n_elements, pair_less());

    // remove duplicates
    auto new_last = thrust::unique(thrust::cuda::par(alloc), adj, adj + n_elements);
    unsigned int nnz = static_cast<unsigned int>(new_last - adj);

    auto source = thrust::make_transform_iterator(adj, get_source());
    auto destination = thrust::make_transform_iterator(adj, get_destination());

    // input matrix in COO format
    unsigned int nverts = N;

    int* d_rowidx = alloc.getTemporaryBuffer<int>(nnz);
    int* d_colidx = alloc.getTemporaryBuffer<int>(nnz);

    thrust::device_ptr<int> rowidx(d_rowidx);
    thrust::device_ptr<int> colidx(d_colidx);

    thrust::copy(source, source + nnz, rowidx);
    thrust::copy(destination, destination + nnz, colidx);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // allocate CSR matrix topology
    int* d_csr_rowptr = alloc.getTemporaryBuffer<int>(nverts + 1);
    check_cusparse(
        cusparseXcoo2csr(handle, d_rowidx, nnz, nverts, d_csr_rowptr, CUSPARSE_INDEX_BASE_ZERO));

    int* d_work = alloc.getTemporaryBuffer<int>(nverts);

    // compute the connected components
    ecl_connected_components(nverts, nnz, d_csr_rowptr, d_colidx, d_components, d_work, dev_prop);

    // reuse work array
    thrust::device_ptr<int> components(d_components);
    thrust::device_ptr<int> work(d_work);

    thrust::copy(thrust::cuda::par(alloc), components, components + nverts, work);
    thrust::sort(thrust::cuda::par(alloc), work, work + nverts);

    int* d_unique = alloc.getTemporaryBuffer<int>(nverts);
    thrust::device_ptr<int> unique(d_unique);

    auto it = thrust::reduce_by_key(thrust::cuda::par(alloc),
                                    work,
                                    work + nverts,
                                    thrust::constant_iterator<int>(1),
                                    unique,
                                    thrust::discard_iterator<int>());

    num_components = static_cast<unsigned int>(it.first - unique);

    // make contiguous
    thrust::lower_bound(thrust::cuda::par(alloc),
                        unique,
                        unique + num_components,
                        components,
                        components + nverts,
                        components);

    // free temporary storage
    alloc.deallocate((char*)d_rowidx);
    alloc.deallocate((char*)d_colidx);
    alloc.deallocate((char*)d_csr_rowptr);
    alloc.deallocate((char*)d_work);
    alloc.deallocate((char*)d_unique);

    // clean cusparse
    cusparseDestroy(handle);
#endif
    }

    } // end namespace gpu
    } // end namespace hpmc
    } // end namespace hoomd
