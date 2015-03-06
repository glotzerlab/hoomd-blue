/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

// Maintainer: mphoward
#include "NeighborListGPUTree.cuh"
#include "TextureTools.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#define MORTON_CODE_N_BINS 1024     // number of bins (2^k) to use to generate 3k bit morton codes
#define PARTICLES_PER_LEAF 4        // max number of particles in a leaf node, must be power of two

/*! \file NeighborListGPUTree.cu
    \brief Defines GPU kernel code for neighbor list tree traversal on the GPU
*/

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;
//! Texture for reading leaf data
scalar4_tex_t leaf_xyzf_tex;
//! Texture for the diameter / body
scalar2_tex_t leaf_db_tex;
//! Texture for reading node upper and lower bounds
scalar4_tex_t aabb_node_bounds_tex;
//! Texture for the head list
texture<unsigned int, 1, cudaReadModeElementType> head_list_tex;
//! Texture for the left children
texture<unsigned int, 1, cudaReadModeElementType> node_left_child_tex;


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
// http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
__device__ inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__global__ void gpu_nlist_morton_codes_kernel(unsigned int *d_morton_codes,
                                              int *d_morton_conditions,
                                              const Scalar4 *d_pos,
                                              const unsigned int *d_map_tree_global,
                                              const unsigned int N,
                                              const BoxDim box,
                                              const Scalar3 ghost_width)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return;
    
    // acquire particle data
    const unsigned int pidx = d_map_tree_global[idx];
    Scalar4 postype = d_pos[ pidx ];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);
        
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos,ghost_width);

    // check if the particle is inside the unit cell + ghost layer in all dimensions
    // this tolerance is small enough that when we multiply by the morton code bin size, we are still in range
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)) )
        {
        *d_morton_conditions = pidx;
        return;
        }

    // find the bin each particle belongs in
    unsigned int ib = f.x * MORTON_CODE_N_BINS;
    unsigned int jb = f.y * MORTON_CODE_N_BINS;
    unsigned int kb = f.z * MORTON_CODE_N_BINS;

    // need to handle the case where the particle is exactly at the box hi
    if (ib == MORTON_CODE_N_BINS && periodic.x)
        ib = 0;
    if (jb == MORTON_CODE_N_BINS && periodic.y)
        jb = 0;
    if (kb == MORTON_CODE_N_BINS && periodic.z)
        kb = 0;
    
    // inline call to some bit swizzling arithmetic
    unsigned int ii = expandBits(ib);
    unsigned int jj = expandBits(jb);
    unsigned int kk = expandBits(kb);
    unsigned int morton_code = ii * 4 + jj * 2 + kk;
    
    // sort morton code by type as we compute it
    d_morton_codes[idx] = morton_code;
    }

cudaError_t gpu_nlist_morton_codes(unsigned int *d_morton_codes,
                                   int *d_morton_conditions,
                                   const Scalar4 *d_pos,
                                   const unsigned int *d_map_tree_global,
                                   const unsigned int N,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_morton_codes_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    gpu_nlist_morton_codes_kernel<<<N/run_block_size + 1, run_block_size>>>(d_morton_codes,
                                                                            d_morton_conditions,
                                                                            d_pos,
                                                                            d_map_tree_global,
                                                                            N,
                                                                            box,
                                                                            ghost_width);
    return cudaSuccess;
    }
    
cudaError_t gpu_nlist_morton_sort(unsigned int *d_morton_codes,
                                  unsigned int *d_map_tree_global,
                                  const unsigned int *h_num_per_type,
                                  const unsigned int ntypes)
    {
    
    // thrust requires to wrap the pod in a thrust object
    thrust::device_ptr<unsigned int> t_morton_codes = thrust::device_pointer_cast(d_morton_codes);
    thrust::device_ptr<unsigned int> t_map_tree_global = thrust::device_pointer_cast(d_map_tree_global);
    
    // loop on types and do a sort by key
    for (unsigned int cur_type=0; cur_type < ntypes; ++cur_type)
        {
        thrust::sort_by_key(t_morton_codes,
                            t_morton_codes + h_num_per_type[cur_type],
                            t_map_tree_global);
                            
        // advance pointers to sort the next type
        t_morton_codes += h_num_per_type[cur_type];
        t_map_tree_global += h_num_per_type[cur_type];
        }
    
    return cudaSuccess;
    }

__global__ void gpu_nlist_merge_particles_kernel(Scalar4 *d_leaf_aabbs,
                                                 unsigned int *d_morton_codes_red,
                                                 uint2 *d_tree_parent_sib,
                                                 const unsigned int *d_morton_codes,
                                                 const Scalar4 *d_pos,
                                                 const unsigned int *d_num_per_type,
                                                 const unsigned int ntypes,
                                                 const unsigned int *d_map_tree_global,
                                                 const unsigned int *d_leaf_offset,
                                                 const unsigned int *d_type_head,
                                                 const unsigned int N,
                                                 const unsigned int nleafs)
    {
    // compute the internal node index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the particle list
    if (idx >= nleafs)
        return;
    
    // get what type of leaf I am
    unsigned int total_bins = 0;
    int leaf_type = -1;
    unsigned int max_idx = N;
    for (unsigned int cur_type=0; leaf_type == -1 && cur_type < ntypes; ++cur_type)
        {
        total_bins += (d_num_per_type[cur_type] + PARTICLES_PER_LEAF - 1)/PARTICLES_PER_LEAF;
        
        if (idx < total_bins)
            {
            leaf_type = cur_type;
            if ( (cur_type + 1) < ntypes)
                {
                max_idx = d_type_head[cur_type + 1];
                }
            }
        }
    
    unsigned int start_idx = idx*PARTICLES_PER_LEAF - d_leaf_offset[leaf_type];
    unsigned int end_idx = (max_idx - start_idx > PARTICLES_PER_LEAF) ? start_idx + PARTICLES_PER_LEAF : max_idx;
    
    
    // upper also holds the skip value, but we have no idea what this is right now
    Scalar4 upper = d_pos[ d_map_tree_global[start_idx] ];
    upper.w = 0.0f;
    
    // lower holds the particle number, we have one already
    Scalar4 lower = upper;
    unsigned int npart = 1;
    
    for (unsigned int cur_p=start_idx+1; cur_p < end_idx; ++cur_p)
        {
        Scalar4 cur_pos = d_pos[ d_map_tree_global[cur_p] ];
        
        // merge the boxes together
        if (cur_pos.x < lower.x) lower.x = cur_pos.x;
        if (cur_pos.x > upper.x) upper.x = cur_pos.x;
            
        if (cur_pos.y < lower.y) lower.y = cur_pos.y;
        if (cur_pos.y > upper.y) upper.y = cur_pos.y;
        
        if (cur_pos.z < lower.z) lower.z = cur_pos.z;
        if (cur_pos.z > upper.z) upper.z = cur_pos.z;

        ++npart;
        }
    
    d_leaf_aabbs[2*idx] = upper;
    d_leaf_aabbs[2*idx + 1] = make_scalar4(lower.x, lower.y, lower.z, __int_as_scalar(npart));
    
    d_morton_codes_red[idx] = d_morton_codes[start_idx];
    
    // fill the parent/sib relationships as if everything is a single leaf at first, to be overridden by hierarchy gen
    // when this is not the case
    d_tree_parent_sib[idx] = make_uint2(idx, idx << 1);
    }
    
cudaError_t gpu_nlist_merge_particles(Scalar4 *d_leaf_aabbs,
                                      unsigned int *d_morton_codes_red,
                                      uint2 *d_tree_parent_sib,
                                      const unsigned int *d_morton_codes,
                                      const Scalar4 *d_pos,
                                      const unsigned int *d_num_per_type,
                                      const unsigned int ntypes,
                                      const unsigned int *d_map_tree_global,
                                      const unsigned int *d_leaf_offset,
                                      const unsigned int *d_type_head,
                                      const unsigned int N,
                                      const unsigned int nleafs,
                                      const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_merge_particles_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    gpu_nlist_merge_particles_kernel<<<nleafs/run_block_size + 1, block_size>>>(d_leaf_aabbs,
                                                                                d_morton_codes_red,
                                                                                d_tree_parent_sib,
                                                                                d_morton_codes,
                                                                                d_pos,
                                                                                d_num_per_type,
                                                                                ntypes,
                                                                                d_map_tree_global,
                                                                                d_leaf_offset,
                                                                                d_type_head,
                                                                                N,
                                                                                nleafs);
    return cudaSuccess;
    }

//! slow implementation of delta from karras paper
/*!
 * computes the longest common prefix
 */
__device__ inline int delta(const unsigned int *d_morton_codes, unsigned int i, unsigned int j, int min_idx, int max_idx)
    {
    if (j > max_idx || j < min_idx)
        {
        return -1;
        }
    
    unsigned int first_code = d_morton_codes[i];
    unsigned int last_code = d_morton_codes[j];
    
    // if codes match, then use index as tie breaker
    if (first_code == last_code)
        {
        return __clz(i ^ j);
        }
    else
        {
        return __clz(first_code ^ last_code);
        }
    }

//! slow implementation of determineRange, needs refining
/*!
 * This is a literal implementation of the Karras pseudocode, with no optimizations or refinement
 */
__device__ inline uint2 determineRange(const unsigned int *d_morton_codes,
                                       const int min_idx,
                                       const int max_idx,
                                       const int idx)
    {
    int forward_prefix = delta(d_morton_codes, idx, idx+1, min_idx, max_idx);
    int backward_prefix = delta(d_morton_codes, idx, idx-1, min_idx, max_idx);
        
    // depends on sign
    int d = ((forward_prefix - backward_prefix) > 0) ? 1 : -1;
    int min_prefix = delta(d_morton_codes, idx, idx-d, min_idx, max_idx);
    
    int lmax = 2;
    while( delta(d_morton_codes, idx, idx + d*lmax, min_idx, max_idx) > min_prefix)
        {
        lmax = lmax << 1;
        }
    
    unsigned int len = 0;
    unsigned int step = lmax;
    do 
        {
        step = step >> 1;
        unsigned int new_len = len + step;
        if (delta(d_morton_codes, idx, idx + d*new_len, min_idx, max_idx) > min_prefix)
            len = new_len;
        }
    while (step > 1);
   
    uint2 range;
    if (d > 0)
        {
        range.x = idx;
        range.y = idx + len;
        }
    else
        {
        range.x = idx - len;
        range.y = idx;
        }
    return range;
    }

__device__ inline unsigned int findSplit(const unsigned int *d_morton_codes,
                                         const unsigned int first,
                                         const unsigned int last)
    {
    unsigned int first_code = d_morton_codes[first];
    unsigned int last_code = d_morton_codes[last];
    
    // if codes match, then just split evenly
    if (first_code == last_code)
        return (first + last) >> 1;
    
    // get the length of the common prefix
    int common_prefix = __clz(first_code ^ last_code);
    
    // assume split starts at first, and begin binary search
    unsigned int split = first;
    unsigned int step = last - first;
    do
        {
        // exponential decrease (is factor of 2 best?)
        step = (step + 1) >> 1;
        unsigned int new_split = split + step;
        
        // if proposed split lies within range
        if (new_split < last)
            {
            unsigned int split_code = d_morton_codes[new_split];
            int split_prefix = __clz(first_code ^ split_code);
            
            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                split = new_split;
                }
            }
        }
    while (step > 1);
    
    return split;
    }
    
__global__ void gpu_nlist_gen_hierarchy_kernel(unsigned int *d_node_left_child,
                                               uint2 *d_tree_parent_sib,
                                               const unsigned int *d_morton_codes,
                                               const unsigned int *d_num_per_type,
                                               const unsigned int ntypes,
                                               const unsigned int nleafs)
    {
    // compute the internal node index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // at first, we work with internal nodes
    if (idx >= (nleafs - ntypes))
        return;
    
    // get what type of leaf I am
    unsigned int min_idx = 0; // the "0" of the leaf node array
    unsigned int max_idx = 0; // the "N-1" of the leaf node array
    
    unsigned int node_idx = idx;
    unsigned int origin = 0;
    unsigned int end = 0;
    
    unsigned int cur_type=0;
    for (cur_type=0; cur_type < ntypes; ++cur_type)
        {
        // current min index is the previous max index
        min_idx = max_idx;
        // max index adds the number of internal nodes in this type (nleaf - 1)
        max_idx += (d_num_per_type[cur_type] + PARTICLES_PER_LEAF - 1)/PARTICLES_PER_LEAF-1;
        
        // we break the loop if we are in range
        if (idx < max_idx)
            {
            // now, we repurpose the min and max index to now correspond to the *leaf* index.
            // the min index is the minimum *leaf* index
            origin = min_idx + cur_type;
            end = max_idx + cur_type;
            node_idx += cur_type;
            break;
            }
        }
        
    // this is a one leaf tree, so process it in a special way
    // it has no children, since it is a leaf
    // it is its own parent and sibling, like a root node
    // no other nodes will process it as a child, so we have to set this directly
//     if (origin == end)
//         {
//         uint2 parent_sib;
//         parent_sib.x = idx;
//         parent_sib.y = idx << 1;
//         d_tree_parent_sib[idx] = parent_sib;
//         return;
//         }
    
    
    // internal nodes have all of the fun
        
    // enact the magical split determining
    uint2 range = determineRange(d_morton_codes, origin, end, node_idx);
    unsigned int first = range.x;
    unsigned int last = range.y;
    unsigned int split = findSplit(d_morton_codes, first, last);
    
    uint2 children;
    // set the children, shifting ahead by nleafs - cur_type to account for leaf shifting
    // this factor comes out from resetting 0 = N_leaf,i each time, and then remapping this to
    // an internal node
    children.x = (split == first) ? split : (nleafs - cur_type + split);
    children.y = ((split + 1) == last) ? (split + 1) : nleafs + split - cur_type + 1;
    
    uint2 parent_sib;
    parent_sib.x = nleafs + idx;
    
    // encode the sibling as the right child
    parent_sib.y = children.y << 1;
    parent_sib.y |= 1;
    
    d_tree_parent_sib[children.x] = parent_sib;
    
    // encode the sibling as the left child
    parent_sib.y = children.x << 1;
    d_tree_parent_sib[children.y] = parent_sib;
    
    // root is always number "zero", but only it can set its parent / sibling
    // we mark both of these as the root for traversing, since only the root node
    // will be its own sibling
    if (node_idx == origin)
        {
        parent_sib.x = nleafs + idx;
        parent_sib.y = (nleafs + idx) << 1;
        
        d_tree_parent_sib[nleafs + idx] = parent_sib;
        }

    // set the left child of this node
    d_node_left_child[idx] = children.x;
    }
    
cudaError_t gpu_nlist_gen_hierarchy(unsigned int *d_node_left_child,
                                    uint2 *d_tree_parent_sib,
                                    const unsigned int *d_morton_codes,
                                    const unsigned int *d_num_per_type,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_gen_hierarchy_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    // one thread per internal node
    gpu_nlist_gen_hierarchy_kernel<<<(nleafs-ntypes)/run_block_size + 1, run_block_size>>>(d_node_left_child,
                                                                                           d_tree_parent_sib,
                                                                                           d_morton_codes,
                                                                                           d_num_per_type,
                                                                                           ntypes,
                                                                                           nleafs);
    return cudaSuccess;
    }


__global__ void gpu_nlist_bubble_aabbs_kernel(unsigned int *d_node_locks,
                                              Scalar4 *d_tree_aabbs,
                                              const uint2 *d_tree_parent_sib,
                                              const unsigned int ntypes,
                                              const unsigned int nleafs,
                                              const unsigned int block_size)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= nleafs)
        return;
            
    // okay, first we start from the leaf and set my bounding box
    Scalar4 cur_upper = d_tree_aabbs[2*idx];
    Scalar4 cur_lower = d_tree_aabbs[2*idx+1];
    
    // zero the counters for internal nodes
    cur_upper.w = 0.0f;
    cur_lower.w = 0.0f;
    
    unsigned int cur_node = idx;
    unsigned int lock_key = 0;
    do
        {
        uint2 cur_parent_sib = d_tree_parent_sib[cur_node];
        unsigned int cur_parent = cur_parent_sib.x;
        unsigned int cur_sibling = cur_parent_sib.y >> 1;
        
        // first we compute the skip for this node always
        // back track up the tree until you find a left child
        // we have a check in place so that we don't stall on the root node
        uint2 backtrack = cur_parent_sib;
        while (!(backtrack.y & 1) && backtrack.x != (backtrack.y >> 1))
            {
            backtrack = d_tree_parent_sib[backtrack.x];
            }
        // then, the skip is to the sibling of that node, or else to quit
        if (backtrack.y & 1)
            {
            d_tree_aabbs[2*cur_node].w = __int_as_scalar(backtrack.y >> 1);
            }
        else
            {
            d_tree_aabbs[2*cur_node].w = __int_as_scalar(-1.0);
            }
                
        // then, we do an atomicAdd on the lock to see if we need to process the parent AABBs
        // check to make sure the parent is bigger than nleafs, or else the node lock always fails
        // so that we terminate the thread
        lock_key = (cur_parent >= nleafs) ? atomicAdd(d_node_locks + cur_parent - nleafs, 1) : 0;
    
        // process the node
        if (lock_key == 1)
            {
            // compute the max upper bound
            Scalar4 sib_upper = d_tree_aabbs[2*cur_sibling];
            if (sib_upper.x > cur_upper.x) cur_upper.x = sib_upper.x;
            if (sib_upper.y > cur_upper.y) cur_upper.y = sib_upper.y;
            if (sib_upper.z > cur_upper.z) cur_upper.z = sib_upper.z;
            d_tree_aabbs[2*cur_parent] = cur_upper;
            
            // compute the min lower bound
            Scalar4 sib_lower = d_tree_aabbs[2*cur_sibling+1];
            if (sib_lower.x < cur_lower.x) cur_lower.x = sib_lower.x;
            if (sib_lower.y < cur_lower.y) cur_lower.y = sib_lower.y;
            if (sib_lower.z < cur_lower.z) cur_lower.z = sib_lower.z;
            d_tree_aabbs[2*cur_parent+1] = cur_lower;
            
            // bump the current node one level
            cur_node = cur_parent;
            }
        }
    while (lock_key == 1);
    
    }
    
cudaError_t gpu_nlist_bubble_aabbs(unsigned int *d_node_locks,
                                   Scalar4 *d_tree_aabbs,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int block_size)
    {
    cudaMemset(d_node_locks, 0, sizeof(unsigned int)*(nleafs - ntypes));
    
    // later we can use block size to perform atomic operations in shared memory rather than
    // global memory for nodes in the same block. but for now, just use slow global memory for everything.
    gpu_nlist_bubble_aabbs_kernel<<<nleafs/block_size + 1, block_size>>>(d_node_locks,
                                                                         d_tree_aabbs,
                                                                         d_tree_parent_sib,
                                                                         ntypes,
                                                                         nleafs,
                                                                         block_size);
    
    return cudaSuccess;
    }
    
__global__ void gpu_nlist_move_particles_kernel(Scalar4 *d_leaf_xyzf,
                                                Scalar2 *d_leaf_db,
                                                const Scalar4 *d_pos,
                                                const Scalar *d_diameter,
                                                const unsigned int *d_body,
                                                const unsigned int *d_map_tree_global,
                                                const unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= N)
        return;
        
    unsigned int p_idx = d_map_tree_global[idx];
    Scalar4 pos_i = d_pos[p_idx];
    d_leaf_xyzf[idx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(p_idx));
    
    Scalar2 db = make_scalar2(d_diameter[p_idx], __int_as_scalar(d_body[p_idx]));
    d_leaf_db[idx] = db;
    }
                                         
cudaError_t gpu_nlist_move_particles(Scalar4 *d_leaf_xyzf,
                                     Scalar2 *d_leaf_db,
                                     const Scalar4 *d_pos,
                                     const Scalar *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int *d_map_tree_global,
                                     const unsigned int N,
                                     const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_move_particles_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    gpu_nlist_move_particles_kernel<<<N/run_block_size + 1, run_block_size>>>(d_leaf_xyzf,
                                                                              d_leaf_db,
                                                                              d_pos,
                                                                              d_diameter,
                                                                              d_body,
                                                                              d_map_tree_global,
                                                                              N);
    return cudaSuccess;
    }


template<unsigned char flags>
__global__ void gpu_nlist_traverse_tree_kernel(unsigned int *d_nlist,
                                               unsigned int *d_n_neigh,
                                               Scalar4 *d_last_updated_pos,
                                               unsigned int *d_conditions,
                                               const unsigned int *d_Nmax,
                                               const unsigned int *d_head_list,
                                               const unsigned int N,
                                               const unsigned int nghosts,
                                               // tree data
                                               const unsigned int *d_map_tree_global,
                                               const unsigned int *d_leaf_offset,
                                               const unsigned int *d_tree_roots,
                                               const unsigned int *d_node_left_child,
                                               const Scalar4 *d_tree_aabbs,
                                               const unsigned int nleafs,
                                               const Scalar4 *d_leaf_xyzf,
                                               const Scalar2 *d_leaf_db,
                                               // particle data
                                               const Scalar4 *d_pos,
                                               // images
                                               const Scalar3 *d_image_list,
                                               const unsigned int nimages,
                                               // neighbor list cutoffs
                                               const Scalar *d_r_cut,
                                               const Scalar r_buff,
                                               const Scalar max_diam,
                                               const unsigned int ntypes)
    {
    bool filter_body = flags & 1;
    bool diameter_shift = flags & 2;
    bool has_internal = !(flags & 4);

    // cache the r_listsq parameters into shared memory
    const Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for per type pair parameters
    extern __shared__ unsigned char s_data[];
    
    // pointer for the r_listsq data
    Scalar *s_r_list = (Scalar *)(&s_data[0]);
    unsigned int *s_Nmax = (unsigned int *)(&s_data[sizeof(Scalar)*num_typ_parameters]);
    unsigned int *s_leaf_offset = (unsigned int *)(&s_data[sizeof(Scalar)*num_typ_parameters + sizeof(unsigned int)*ntypes]);

    // load in the per type pair r_list
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_r_list[cur_offset + threadIdx.x] = d_r_cut[cur_offset + threadIdx.x]+r_buff;
            }
            
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = d_Nmax[cur_offset + threadIdx.x];
            s_leaf_offset[cur_offset + threadIdx.x] = d_leaf_offset[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();
    
    
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the leaf list
    if (idx >= (N+nghosts))
        return;  
    
    // read in the current position and orientation
    unsigned int my_pidx = d_map_tree_global[idx];
    // we only process particles owned by this processor for neighbors
    if (my_pidx >= N)
        return;
        
    const Scalar4 postype_i = texFetchScalar4(d_pos, pdata_pos_tex, my_pidx);
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const unsigned int type_i = __scalar_as_int(postype_i.w);
    
    // fetch the diameter and body out of the leaf texture since it's bound anyway
    const Scalar2 db_i = texFetchScalar2(d_leaf_db, leaf_db_tex, idx);
    const Scalar diam_i = db_i.x;
    const unsigned int body_i = __scalar_as_int(db_i.y);
    
    const unsigned int nlist_head_i = texFetchUint(d_head_list, head_list_tex, my_pidx);
    
    unsigned int n_neigh_i = 0;
    for (unsigned int cur_pair_type=0; cur_pair_type < ntypes; ++cur_pair_type)
        {
        // Check primary box
        const Scalar r_cut_i = s_r_list[typpair_idx(type_i,cur_pair_type)];
        
        // stash the r_cutsq before any diameter shifting
        const Scalar r_cutsq_i = r_cut_i*r_cut_i;
        
        // the rlist to use for the AABB search has to be at least as big as the biggest diameter
        Scalar r_list_i = r_cut_i;
        if (diameter_shift)
            r_list_i += max_diam - Scalar(1.0);
        
        const unsigned int cur_tree_root = d_tree_roots[cur_pair_type];
        
        for (unsigned int cur_image = 0; cur_image < nimages; ++cur_image)
            {
            const Scalar3 pos_i_image = pos_i + d_image_list[cur_image];
            const Scalar3 aabb_upper = make_scalar3(pos_i_image.x + r_list_i,
                                                    pos_i_image.y + r_list_i,
                                                    pos_i_image.z + r_list_i);
            const Scalar3 aabb_lower = make_scalar3(pos_i_image.x - r_list_i,
                                                    pos_i_image.y - r_list_i,
                                                    pos_i_image.z - r_list_i);
            
            // stackless search
            int cur_node_idx = cur_tree_root;
            while (cur_node_idx > -1)
                {
                const Scalar4 upper_rope = texFetchScalar4(d_tree_aabbs, aabb_node_bounds_tex, 2*cur_node_idx);
                const Scalar4 lower_np = texFetchScalar4(d_tree_aabbs, aabb_node_bounds_tex, 2*cur_node_idx+1);
                
                if (!(aabb_upper.x < lower_np.x
                      || aabb_lower.x > upper_rope.x
                      || aabb_upper.y < lower_np.y
                      || aabb_lower.y > upper_rope.y
                      || aabb_upper.z < lower_np.z
                      || aabb_lower.z > upper_rope.z))                
                    {
                    const unsigned int n_part = __scalar_as_int(lower_np.w);
                    if(n_part > 0)
                        {
                        // leaf node
                        // all leaves must have at least 1 particle, so we can use this to decide
                        const unsigned int node_head = PARTICLES_PER_LEAF*cur_node_idx - s_leaf_offset[cur_pair_type];
                        for (unsigned int cur_p = node_head; cur_p < node_head + n_part; ++cur_p)
                            { 
                            // neighbor j
                            const Scalar4 cur_xyzf = texFetchScalar4(d_leaf_xyzf, leaf_xyzf_tex, cur_p);
                            const Scalar3 pos_j = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                            const unsigned int j = __scalar_as_int(cur_xyzf.w);
                        
                            const Scalar2 cur_db = texFetchScalar2(d_leaf_db, leaf_db_tex, cur_p);
                            const Scalar diam_j = cur_db.x;
                            const unsigned int body_j = __scalar_as_int(cur_db.y);

                            bool excluded = (my_pidx == j);

                            if (filter_body && body_i != 0xffffffff)
                                excluded = excluded | (body_i == body_j);
                            
                            if (!excluded)
                                {
                                // now we can trim down the actual particles based on diameter
                                // compute the shift for the cutoff if not excluded
                                Scalar sqshift = Scalar(0.0);
                                if (diameter_shift)
                                    {
                                    const Scalar delta = (diam_i + diam_j) * Scalar(0.5) - Scalar(1.0);
                                    // r^2 < (r_list + delta)^2
                                    // r^2 < r_listsq + delta^2 + 2*r_list*delta
                                    sqshift = (delta + Scalar(2.0) * r_cut_i) * delta;
                                    }
                                    
                                // compute distance and wrap back into box
                                Scalar3 drij = pos_j - pos_i_image;
                                Scalar dr2 = dot(drij,drij);

                                if (dr2 <= (r_cutsq_i + sqshift))
                                    {
                                    if (n_neigh_i < s_Nmax[type_i])
                                        {
                                        d_nlist[nlist_head_i + n_neigh_i] = j;
                                        }
                                    ++n_neigh_i;
                                    }
                                }
                            }
                        
                        // leaf nodes always move to their rope
                        cur_node_idx = __scalar_as_int(upper_rope.w);
                        }
                    else if (has_internal)
                        {
                        // internal node, take left child
                        cur_node_idx = texFetchUint(d_node_left_child, node_left_child_tex, cur_node_idx - nleafs);
                        }
                    }
                else
                    {
                    cur_node_idx = __scalar_as_int(upper_rope.w); // no overlap, rope ahead
                    }
                } // end stackless search  
            } // end loop over images
        } // end loop over pair types
        
    // could try reordering by idx instead of pidx, but that seems to not make much difference in microbenchmarking.
    d_n_neigh[my_pidx] = n_neigh_i;
    d_last_updated_pos[my_pidx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __scalar_as_int(type_i));
    
    // update the number of neighbors for this type if allocated memory is exceeded
    if (n_neigh_i >= s_Nmax[type_i])
        atomicMax(&d_conditions[type_i], n_neigh_i);
    }
    
                                         
cudaError_t gpu_nlist_traverse_tree(unsigned int *d_nlist,
                                    unsigned int *d_n_neigh,
                                    Scalar4 *d_last_updated_pos,
                                    unsigned int *d_conditions,
                                    const unsigned int *d_Nmax,
                                    const unsigned int *d_head_list,
                                    const unsigned int N,
                                    const unsigned int nghosts,
                                    // tree data
                                    const unsigned int *d_map_tree_global,
                                    const unsigned int *d_leaf_offset,
                                    const unsigned int *d_tree_roots,
                                    const unsigned int *d_node_left_child,
                                    const Scalar4 *d_tree_aabbs,
                                    const unsigned int nleafs,
                                    const unsigned int ninternal,
                                    const unsigned int nnodes,
                                    const Scalar4 *d_leaf_xyzf,
                                    const Scalar2 *d_leaf_db,
                                    // particle data
                                    const Scalar4 *d_pos,
                                    // images
                                    const Scalar3 *d_image_list,
                                    const unsigned int nimages,
                                    // neighbor list cutoffs
                                    const Scalar *d_r_cut,
                                    const Scalar r_buff,
                                    const Scalar max_diam,
                                    const unsigned int ntypes,
                                    bool filter_body,
                                    bool diameter_shift,
                                    const unsigned int compute_capability,
                                    const unsigned int block_size)
    {
    // shared memory = r_list + Nmax
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = sizeof(Scalar)*typpair_idx.getNumElements() + 2*sizeof(unsigned int)*ntypes;
    
    // bind the neighborlist texture
    if (compute_capability < 35)
        {
        pdata_pos_tex.normalized = false;
        pdata_pos_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*(N+nghosts));
        
        leaf_xyzf_tex.normalized = false;
        leaf_xyzf_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, leaf_xyzf_tex, d_leaf_xyzf, sizeof(Scalar4)*(N+nghosts));
        
        leaf_db_tex.normalized = false;
        leaf_db_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, leaf_db_tex, d_leaf_db, sizeof(Scalar2)*(N+nghosts));
        
        aabb_node_bounds_tex.normalized = false;
        aabb_node_bounds_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, aabb_node_bounds_tex, d_tree_aabbs, sizeof(Scalar4)*2*nnodes);
        
        head_list_tex.normalized = false;
        head_list_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, head_list_tex, d_head_list, sizeof(unsigned int)*(N+nghosts));
        
        if (ninternal > 0)
            {
            node_left_child_tex.normalized = false;
            node_left_child_tex.filterMode = cudaFilterModePoint;
            cudaBindTexture(0, node_left_child_tex, d_node_left_child, sizeof(unsigned int)*ninternal);
            }
        }
    
    // here we template on a few simple bools, which seems to make the traversal a little faster in microbenchmarks
    // the bools are encoded as bitwise flags in a char
    // filter body = (flags & 1)
    // diameter shift = (flags & 2)
    // has_internal_nodes = !(flags & 4)
    //
    // We enumerate all possibilities below
    
    if (!filter_body && !diameter_shift && ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<0>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<0><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (filter_body && !diameter_shift && ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<1>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<1><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (!filter_body && diameter_shift && ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<2>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<2><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (filter_body && diameter_shift && ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<3>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<3><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (!filter_body && !diameter_shift && !ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<4>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<4><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (filter_body && !diameter_shift && !ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<5>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<5><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (!filter_body && diameter_shift && !ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<6>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<6><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    else if (filter_body && diameter_shift && !ninternal)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<7>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        int nblocks = (N+nghosts)/run_block_size + 1;
        gpu_nlist_traverse_tree_kernel<7><<<nblocks, run_block_size, shared_size>>>(d_nlist,
                                                                                    d_n_neigh,
                                                                                    d_last_updated_pos,
                                                                                    d_conditions,
                                                                                    d_Nmax,
                                                                                    d_head_list,
                                                                                    N,
                                                                                    nghosts,
                                                                                    d_map_tree_global,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
                                                                                    d_node_left_child,
                                                                                    d_tree_aabbs,
                                                                                    nleafs,
                                                                                    d_leaf_xyzf,
                                                                                    d_leaf_db,
                                                                                    d_pos,
                                                                                    d_image_list,
                                                                                    nimages,
                                                                                    d_r_cut,
                                                                                    r_buff,
                                                                                    max_diam,
                                                                                    ntypes);
        }
    
    return cudaSuccess;
    }
    

__global__ void gpu_nlist_map_particles_gen_mask_kernel(unsigned int *d_type_mask,
                                                        const Scalar4 *d_pos,
                                                        const unsigned int N,
                                                        const unsigned int type)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return; 

    d_type_mask[idx] = (__scalar_as_int(d_pos[idx].w) == type);
    }

cudaError_t gpu_nlist_map_particles_gen_mask(unsigned int *d_type_mask,
                                             const Scalar4 *d_pos,
                                             const unsigned int N,
                                             const unsigned int type,
                                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_map_particles_gen_mask_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    gpu_nlist_map_particles_gen_mask_kernel<<<N/run_block_size + 1, run_block_size>>>(d_type_mask,
                                                                                      d_pos,
                                                                                      N,
                                                                                      type);
    return cudaSuccess;
    }
    
__global__ void gpu_nlist_map_particles_kernel(unsigned int *d_map_tree_global,
                                               unsigned int *d_num_per_type,
                                               unsigned int *d_type_head,
                                               unsigned int *d_leaf_offset,
                                               unsigned int *d_tree_roots,
                                               unsigned int *d_cumulative_pids,
                                               const unsigned int *d_type_mask,
                                               const unsigned int N,
                                               const unsigned int type,
                                               const unsigned int ntypes)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return; 
    
    // all threads of the current type 
    bool is_type_i = (d_type_mask[idx] == 1);
    if (is_type_i)
        {
        unsigned int new_idx = d_cumulative_pids[idx] + d_type_head[type];
        d_map_tree_global[new_idx] = idx;
        }
        
    // last thread processes the offset
    if (idx == (N-1))
        {
        // add one if this thread is last to make up for exclusive count style
        const unsigned int num_type_i = d_cumulative_pids[N-1] + is_type_i;
        d_num_per_type[type] = num_type_i;
        
        // number of leafs comes from number of particles
        const unsigned int num_leaf_i = (num_type_i + PARTICLES_PER_LEAF - 1) / PARTICLES_PER_LEAF;
        
        // increment all subsequent type heads by the current number of particles
        for (unsigned int type_it = 0; type_it < ntypes; ++type_it)
            {
            d_tree_roots[type_it] += num_leaf_i;
            
            // forward incrementing
            if (type_it >= (type + 1))
                {
                // forward add the number of particles for the type head
                d_type_head[type_it] += num_type_i;
            
                // forward add the number of internal nodes
                d_tree_roots[type_it] += num_leaf_i - 1;
            
                unsigned int remainder = num_type_i % PARTICLES_PER_LEAF;
                if (remainder > 0)
                    {
                    d_leaf_offset[type_it] += (PARTICLES_PER_LEAF - remainder);
                    }
                }
            }
        
        // the last time through, clean up the tree roots
        // this might seem slow to repeat this work, but it really isn't that expensive to loop through the types
        //
        // any type that has only one leaf needs to have its root overridden to just be the leaf
//         if (type == (ntypes-1))
//             {
//             unsigned int leaf_head = 0;
//             for (unsigned int type_it = 0; type_it < ntypes; ++type_it)
//                 {
//                 unsigned int num_type_i = d_num_per_type[type_it];
//                 unsigned int num_leaf_i = (num_type_i + PARTICLES_PER_LEAF - 1) / PARTICLES_PER_LEAF;
//                 if (num_leaf_i == 1)
//                     {
//                     d_tree_roots[type_it] = leaf_head;
//                     }
//                 leaf_head += num_leaf_i;
//                 }
//             }
        }
    }
    
cudaError_t gpu_nlist_map_particles(unsigned int *d_map_tree_global,
                                    unsigned int *d_num_per_type,
                                    unsigned int *d_type_head,
                                    unsigned int *d_leaf_offset,
                                    unsigned int *d_tree_roots,
                                    unsigned int *d_cumulative_pids,
                                    const unsigned int *d_type_mask,
                                    const unsigned int N,
                                    const unsigned int type,
                                    const unsigned int ntypes,
                                    const unsigned int block_size)
    {
    thrust::device_ptr<const unsigned int> t_type_mask = thrust::device_pointer_cast(d_type_mask);
    thrust::device_ptr<unsigned int> t_cumulative_pids = thrust::device_pointer_cast(d_cumulative_pids);
    
    // count up all particles masked for this type with partial sum scan
    thrust::exclusive_scan(t_type_mask, t_type_mask + N, t_cumulative_pids);
    
    // for the first type, you need to clear out the partial sums
    if (type == 0)
        {
        cudaMemset(d_num_per_type, 0, sizeof(unsigned int)*ntypes);
        cudaMemset(d_type_head, 0, sizeof(unsigned int)*ntypes);
        cudaMemset(d_leaf_offset, 0, sizeof(unsigned int)*ntypes);
        cudaMemset(d_tree_roots, 0, sizeof(unsigned int)*ntypes);
        }
    
    // apply the scan
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_map_particles_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    
    gpu_nlist_map_particles_kernel<<<N/run_block_size + 1, run_block_size>>>(d_map_tree_global,
                                                                             d_num_per_type,
                                                                             d_type_head,
                                                                             d_leaf_offset,
                                                                             d_tree_roots,
                                                                             d_cumulative_pids,
                                                                             d_type_mask,
                                                                             N,
                                                                             type,
                                                                             ntypes);
    return cudaSuccess;
    }

#undef MORTON_CODE_N_BINS
#undef PARTICLES_PER_LEAF
