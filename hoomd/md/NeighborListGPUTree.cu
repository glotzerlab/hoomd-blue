// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward
#include "NeighborListGPUTree.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/extern/cub/cub/cub.cuh"

#define MORTON_CODE_BITS   30       //!< Length of the Morton code in bits (k = 10 bits per direction)
#define MORTON_CODE_N_BINS 1024     //!< Number of bins (2^10) per direction to generate 30 bit Morton codes
#define MORTON_TYPE_MASK_64 0x000000003fffffffu //!< 64 bit mask to turn morton code-type back to morton code

/*! \file NeighborListGPUTree.cu
    \brief Defines GPU kernel code for neighbor list tree traversal on the GPU
*/

//!< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
/*!
 * \param v unsigned integer with 10 bits set
 * \returns The integer expanded with two zeros interleaved between bits
 * http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
__device__ inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

//! Assigns the Morton code-type key for each particle on this processor
/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_map_tree_pid List to be overwritten with particle ids in ascending order
 * \param d_morton_conditions Flag if a local particle (not a ghost) is detected out of bounds
 * \param d_pos Particle positions
 * \param N Number of local particles
 * \param nghosts Number of ghost particles
 * \param box Local simulation box
 * \param ghost_width Anticipated size of the ghost layer for nonbonded interactions
 *
 * \b Implementation
 * A sorting key is generated for each particle by determining the 30 bit Morton code for each particle, and then
 * concatenating onto the type. Both the Morton code and the type are 32 bit integers, so the concatenation is stored
 * compactly in a 64 bit integer morton_type = (type << 30) + morton code. In this way, a lexicographic sort will
 * sort first by type, then by morton code. The corresponding particle id (thread index) is stashed into d_map_tree_pid
 * to track particles after sorting.
 */
__global__ void gpu_nlist_morton_types_kernel(uint64_t *d_morton_types,
                                              unsigned int *d_map_tree_pid,
                                              int *d_morton_conditions,
                                              const Scalar4 *d_pos,
                                              const unsigned int N,
                                              const unsigned int nghosts,
                                              const BoxDim box,
                                              const Scalar3 ghost_width)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N+nghosts)
        return;

    // acquire particle data
    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos,ghost_width);

    /* check if the particle is inside the unit cell + ghost layer in all dimensions
     * this tolerance is small enough that when we multiply by the morton code bin size, we are still in range
     * we silently ignore ghosts outside of this width, and instead deal with that special case below
     * where extra ghosts are communicated (e.g. for bonded interactions)
     */
    if (((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001))) && idx < N)
        {
        atomicMax(d_morton_conditions,idx+1);
        return;
        }

    // find the bin each particle belongs in
    int ib = (int)(f.x * MORTON_CODE_N_BINS);
    int jb = (int)(f.y * MORTON_CODE_N_BINS);
    int kb = (int)(f.z * MORTON_CODE_N_BINS);

    if (!periodic.x) // ghosts exist and may be past layer width
        {
        // handle special cases where random ghosts are beyond the expected layer
        // by just rounding to the nearest edge
        if (ib < 0)
            {
            ib = 0;
            }
        else if (ib >= MORTON_CODE_N_BINS)
            {
            ib = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (ib == MORTON_CODE_N_BINS) // some particles lie exactly on the edge, floor them to zero
        {
        ib = 0;
        }

    // do as for x in y
    if (!periodic.y)
        {
        if (jb < 0)
            {
            jb = 0;
            }
        else if (jb >= MORTON_CODE_N_BINS)
            {
            jb = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (jb == MORTON_CODE_N_BINS)
        {
        jb = 0;
        }

    // do as for y in z
    if (!periodic.z)
        {
        if (kb < 0)
            {
            kb = 0;
            }
        else if (kb >= MORTON_CODE_N_BINS)
            {
            kb = MORTON_CODE_N_BINS - 1;
            }
        }
    else if (kb == MORTON_CODE_N_BINS)
        {
        kb = 0;
        }

    // inline call to some bit swizzling arithmetic
    unsigned int ii = expandBits((unsigned int)ib);
    unsigned int jj = expandBits((unsigned int)jb);
    unsigned int kk = expandBits((unsigned int)kb);
    unsigned int morton_code = ii * 4 + jj * 2 + kk;

    // save the morton code and corresponding particle index for sorting
    // the morton codes hold both the type and the code to sort by both type and position simultaneously
    d_morton_types[idx] = (((uint64_t)type) << MORTON_CODE_BITS) + (uint64_t)morton_code;
    d_map_tree_pid[idx] = idx;
    }

/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_map_tree_pid List to be overwritten with particle ids in ascending order
 * \param d_morton_conditions Flag if a local particle (not a ghost) is detected out of bounds
 * \param d_pos Particle positions
 * \param N Number of local particles
 * \param nghosts Number of ghost particles
 * \param box Local simulation box
 * \param ghost_width Anticipated size of the ghost layer for nonbonded interactions
 * \param block_size Requested thread block size of kernel launch
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_morton_types(uint64_t *d_morton_types,
                                   unsigned int *d_map_tree_pid,
                                   int *d_morton_conditions,
                                   const Scalar4 *d_pos,
                                   const unsigned int N,
                                   const unsigned int nghosts,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_morton_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    gpu_nlist_morton_types_kernel<<<(N+nghosts)/run_block_size + 1, run_block_size>>>(d_morton_types,
                                                                                      d_map_tree_pid,
                                                                                      d_morton_conditions,
                                                                                      d_pos,
                                                                                      N,
                                                                                      nghosts,
                                                                                      box,
                                                                                      ghost_width);
    return cudaSuccess;
    }

/*!
 * \param d_morton_types Morton code-type keys per particle
 * \param d_morton_types_alt Auxiliary array of equal size to d_morton_types for double buffered sorting
 * \param d_map_tree_pid List of particle ids
 * \param d_map_tree_pid_alt Auxiliary array of equal size to d_map_tree_pid for double buffered sorting
 * \param d_tmp_storage Temporary storage in device memory
 * \param tmp_storage_bytes Number of bytes allocated for temporary storage
 * \param swap_morton Flag to switch real data from auxiliary array to primary array after sorting
 * \param swap_map Flag to switch real data from auxiliary array to primary array after sorting
 * \param Ntot Total number of keys to sort
 * \param n_type_bits Number of bits to check for particle types
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation
 * The CUB library is used for device-wide radix sorting. Radix sorting is O(kN) where k is the number of bits to check
 * in an unsigned integer key, and N is the number of keys. We restrict the number of bits checked in the max 64 bit
 * keys by only checking up to the MORTON_CODE_BITS + n_type_bits most significant bit. CUB DeviceRadixSort performs
 * its own tuning at run time.
 *
 * Because CUB requires temporary storage, this function must be called twice. First, when \a d_tmp_storage is NULL,
 * the number of bytes required for temporary storage is saved in \a tmp_storage_bytes. This memory must then be
 * allocated in \a d_tmp_storage. On the second call, the radix sort is performed. Because the radix sort may put the
 * active (sorted) buffer in either slot of the DoubleBuffer, a boolean flag is set in \a swap_morton and \a swap_map
 * for whether these data arrays should be swapped.
 */
cudaError_t gpu_nlist_morton_sort(uint64_t *d_morton_types,
                                  uint64_t *d_morton_types_alt,
                                  unsigned int *d_map_tree_pid,
                                  unsigned int *d_map_tree_pid_alt,
                                  void *d_tmp_storage,
                                  size_t &tmp_storage_bytes,
                                  bool &swap_morton,
                                  bool &swap_map,
                                  const unsigned int Ntot,
                                  const unsigned int n_type_bits)
    {
    // initialize memory as "double buffered"
    cub::DoubleBuffer<uint64_t> d_keys(d_morton_types, d_morton_types_alt);
    cub::DoubleBuffer<unsigned int> d_vals(d_map_tree_pid, d_map_tree_pid_alt);

    // on the first pass, this just sizes the temporary storage
    // on the second pass, it actually does the radix sort
    cub::DeviceRadixSort::SortPairs(d_tmp_storage,
                                    tmp_storage_bytes,
                                    d_keys,
                                    d_vals,
                                    Ntot,
                                    0,
                                    MORTON_CODE_BITS+n_type_bits);

    // we've only done something to the buffers on the second time when temporary storage is allocated
    if (d_tmp_storage != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the right array
        swap_morton = (d_keys.selector == 1);
        swap_map = (d_vals.selector == 1);
        }

    return cudaSuccess;
    }

//! Kernel to merge adjacent codes into leaf nodes
/*!
 * \param d_tree_aabbs Flat array holding all AABBs for the tree
 * \param d_morton_codes_red The Morton codes corresponding to the merged leafs
 * \param d_tree_parent_sib Parent and sibling indexes for all nodes
 * \param d_morton_types Morton-code type keys for all particles
 * \param d_pos Particle positions
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of particle types
 * \param d_map_tree_pid Sorted particle order (maps local index to ParticleData index)
 * \param d_leaf_offset Amount to subtract from the expected leaf starting index to make an array with no holes by type
 * \param d_type_head Index to first type and leaf ordered particles by type
 * \param Ntot Total number of keys to sort
 * \param nleafs Number of leaf nodes
 *
 * \b Implementation
 * One thread per leaf is called, and is responsible for merging NLIST_GPU_PARTICLES_PER_LEAF into an AABB. Each thread
 * first determines what type of leaf particle it is operating on by calculating and iterating on the number of leafs
 * of each type. Then, the starting index is determined by subtracting d_leaf_offset[type] from the starting index that
 * would be set in a nleaf x NLIST_GPU_PARTICLES_PER_LEAF array. The reason for this complexity is that the leaf particle
 * array is not permitted to have any "holes" in it for faster traversal. The AABB is merged from the particle
 * positions, and a Morton code is assigned to this AABB for determining tree hierarchy based on the Morton code of
 * the first particle in the leaf. Although this does not necessarily generate the best ordering along the Z order curve
 * for the newly merged leafs, it does guarantee that the leaf Morton codes are still in lexicographic ordering.
 *
 * AABBs are stored as two Scalar4s in a flat array. The first three coordinates of each Scalar4 correspond to the upper
 * and lower bounds of the AABB. The last value of the upper AABB will hold a "rope" for traversing the tree (see
 * gpu_nlist_bubble_aabbs_kernel), while the last value of the lower AABB holds the number of particles for a leaf node,
 * or the left child for an internal node. This is determined by setting a bit to mark this value as a rope or as child.
 */
__global__ void gpu_nlist_merge_particles_kernel(Scalar4 *d_tree_aabbs,
                                                 uint32_t *d_morton_codes_red,
                                                 uint2 *d_tree_parent_sib,
                                                 const uint64_t *d_morton_types,
                                                 const Scalar4 *d_pos,
                                                 const unsigned int *d_num_per_type,
                                                 const unsigned int ntypes,
                                                 const unsigned int *d_map_tree_pid,
                                                 const unsigned int *d_leaf_offset,
                                                 const unsigned int *d_type_head,
                                                 const unsigned int Ntot,
                                                 const unsigned int nleafs)
    {
    // leaf index
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per leaf
    if (idx >= nleafs)
        return;

    // get what type of leaf I am
    unsigned int total_bins = 0;
    int leaf_type = -1;
    unsigned int max_idx = Ntot;
    for (unsigned int cur_type=0; leaf_type == -1 && cur_type < ntypes; ++cur_type)
        {
        total_bins += (d_num_per_type[cur_type] + NLIST_GPU_PARTICLES_PER_LEAF - 1)/NLIST_GPU_PARTICLES_PER_LEAF;

        if (idx < total_bins)
            {
            leaf_type = cur_type;
            for (unsigned int next_type=cur_type+1; next_type < ntypes; ++next_type)
                {
                if (d_type_head[next_type])
                    {
                    max_idx = d_type_head[next_type] - 1;
                    break; // quit out of this inner loop once a match is found
                    }
                }
            break; // quit the outer loop
            }
        }

    // get the starting particle index assuming naive leaf structure, and then subtract offset to eliminate "holes"
    unsigned int start_idx = idx*NLIST_GPU_PARTICLES_PER_LEAF - d_leaf_offset[leaf_type];
    unsigned int end_idx = (max_idx - start_idx > NLIST_GPU_PARTICLES_PER_LEAF) ? start_idx + NLIST_GPU_PARTICLES_PER_LEAF : max_idx;


    // upper also holds the skip value, but we have no idea what this is right now
    Scalar4 upper = d_pos[ d_map_tree_pid[start_idx] ];
    upper.w = 0.0f;

    // lower holds the particle number, we have one already
    Scalar4 lower = upper;
    unsigned int npart = 1;

    for (unsigned int cur_p=start_idx+1; cur_p < end_idx; ++cur_p)
        {
        Scalar4 cur_pos = d_pos[ d_map_tree_pid[cur_p] ];

        // merge the boxes together
        if (cur_pos.x < lower.x) lower.x = cur_pos.x;
        if (cur_pos.x > upper.x) upper.x = cur_pos.x;

        if (cur_pos.y < lower.y) lower.y = cur_pos.y;
        if (cur_pos.y > upper.y) upper.y = cur_pos.y;

        if (cur_pos.z < lower.z) lower.z = cur_pos.z;
        if (cur_pos.z > upper.z) upper.z = cur_pos.z;

        ++npart;
        }

    d_tree_aabbs[2*idx] = upper;
    d_tree_aabbs[2*idx + 1] = make_scalar4(lower.x, lower.y, lower.z, __int_as_scalar(npart << 1));

    // take logical AND with the 30 bit mask for the morton codes to extract just the morton code
    // no sense swinging around 64 bit integers anymore
    d_morton_codes_red[idx] = (unsigned int)(d_morton_types[start_idx] & MORTON_TYPE_MASK_64);

    // fill the parent/sib relationships as if everything is a single leaf at first, to be overridden by hierarchy gen
    // when this is not the case
    d_tree_parent_sib[idx] = make_uint2(idx, idx << 1);
    }

/*!
 * \param d_tree_aabbs Flat array holding all AABBs for the tree
 * \param d_morton_codes_red The Morton codes corresponding to the merged leafs
 * \param d_tree_parent_sib Parent and sibling indexes for all nodes
 * \param d_morton_types Morton-code type keys for all particles
 * \param d_pos Particle positions
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of particle types
 * \param d_map_tree_pid Sorted particle order (maps local index to ParticleData index)
 * \param d_leaf_offset Amount to subtract from the expected leaf starting index to make an array with no holes by type
 * \param d_type_head Index to first type and leaf ordered particles by type
 * \param Ntot Total number of keys to sort
 * \param nleafs Number of leaf nodes
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_merge_particles(Scalar4 *d_tree_aabbs,
                                      uint32_t *d_morton_codes_red,
                                      uint2 *d_tree_parent_sib,
                                      const uint64_t *d_morton_types,
                                      const Scalar4 *d_pos,
                                      const unsigned int *d_num_per_type,
                                      const unsigned int ntypes,
                                      const unsigned int *d_map_tree_pid,
                                      const unsigned int *d_leaf_offset,
                                      const unsigned int *d_type_head,
                                      const unsigned int Ntot,
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

    gpu_nlist_merge_particles_kernel<<<nleafs/run_block_size + 1, block_size>>>(d_tree_aabbs,
                                                                                d_morton_codes_red,
                                                                                d_tree_parent_sib,
                                                                                d_morton_types,
                                                                                d_pos,
                                                                                d_num_per_type,
                                                                                ntypes,
                                                                                d_map_tree_pid,
                                                                                d_leaf_offset,
                                                                                d_type_head,
                                                                                Ntot,
                                                                                nleafs);
    return cudaSuccess;
    }

//! Computes the longest common prefix between Morton codes
/*!
 * \param d_morton_codes Array of Morton codes
 * \param i First Morton code index
 * \param j Second Morton code index
 * \param min_idx The smallest index considered "in range" (inclusive)
 * \param max_idx The last index considered "in range" (inclusive)
 *
 * \returns number of bits shared between the Morton codes of i and j
 *
 * delta(i,j) is defined as the largest number of bits shared between Morton codes i and j. When the Morton codes are
 * sorted, this implies delta(i',j') >= delta(i,j) for any i',j' in [i,j]. If i and j lie outside
 * of the range of Morton codes corresponding to this tree, then it always returns -1. If the Morton codes for i and j
 * are identical, then the longest prefix of i and j is used as a tie breaker.
 */
__device__ inline int delta(const uint32_t *d_morton_codes,
                            const int i,
                            const int j,
                            const int min_idx,
                            const int max_idx)
    {
    assert(i >= min_idx && i <= max_idx);

    if (j > max_idx || j < min_idx)
        {
        return -1;
        }

    uint32_t first_code = d_morton_codes[i];
    uint32_t last_code = d_morton_codes[j];

    // if codes match, then use index as tie breaker
    // the number of shared bits is equal to the 32 bits in the integer, plus the number of bits shared between the
    // indexes (offset from the start of the node range to make things simpler)
    if (first_code == last_code)
        {
        return (32 + __clz((i-min_idx) ^ (j-min_idx)));
        }
    else
        {
        return __clz(first_code ^ last_code);
        }
    }

//! Determines the range of Morton codes that a node covers
/*!
 * \param d_morton_codes Array of Morton codes
 * \param min_idx The smallest Morton code index considered "in range" (inclusive)
 * \param max_idx The last Morton code index considered "in range" (inclusive)
 * \param i Current node (Morton code) index
 *
 * \returns the minimum and maximum leafs covered by this node
 * \note This is a literal implementation of the Karras pseudocode, with no optimizations or refinement.
 *       Tero Karras, "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees",
 *       High Performance Graphics (2012).
 */
__device__ inline uint3 determineRangeSplit(const uint32_t *d_morton_codes,
                                            const int min_idx,
                                            const int max_idx,
                                            const int i)
    {
    const int forward_prefix = delta(d_morton_codes, i, i+1, min_idx, max_idx);
    const int backward_prefix = delta(d_morton_codes, i, i-1, min_idx, max_idx);

    // get direction of the range based on sign
    const int d = (forward_prefix >= backward_prefix) ? 1 : -1;

    // get minimum prefix
    const int min_prefix = delta(d_morton_codes, i, i-d, min_idx, max_idx);

    // get maximum prefix by binary search
    int lmax = 2;
    while( delta(d_morton_codes, i, i + d*lmax, min_idx, max_idx) > min_prefix)
        {
        lmax = lmax << 1;
        }
    int l = 0; int t = lmax;
    do
        {
        t = t >> 1;
        if (delta(d_morton_codes, i, i + (l+t)*d, min_idx, max_idx) > min_prefix)
            l = l + t;
        }
    while (t > 1);
    const int j = i + l*d;

    // get the length of the common prefix
    const int common_prefix = delta(d_morton_codes, i, j, min_idx, max_idx);

    // binary search to find split position
    int s = 0; t = l;
    do
        {
        t = (t + 1) >> 1;
        // if proposed split lies within range
        if (s+t < l)
            {
            const int split_prefix = delta(d_morton_codes, i, i+(s+t)*d, min_idx, max_idx);

            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                s = s + t;
                }
            }
        }
    while (t > 1);
    const int split = i + s*d + min(d,0);

    return make_uint3(min(i,j), max(i,j), split);
    }

//! Kernel to generate the parent-child-sibling relationships between nodes
/*!
 * \param d_tree_parent_sib Parent and sibling for each node in the tree
 * \param d_morton_codes Morton codes for each leaf node
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of types
 * \param nleafs Number of leafs
 *
 * \b Implementation
 * One thread is called per internal node in a single kernel launch. Each thread first determines its "local" index
 * as an internal node within a tree based on the number of leafs per tree. The range of leafs covered by the internal
 * node is determined, and then its split position is identified. The split identifies the children of the node as
 * another internal node or as a leaf node.
 *
 * The parent and sibling of each child node is saved. The sibling id is bit shifted so as to use a single bit to encode
 * the sibling as a right child or left child (after shifting, we set the bit to 1 if the sibling is a right child).
 * If the child is a root node, it also saves information for itself (since no other node ever identifies a root as a
 * child node).
 */
__global__ void gpu_nlist_gen_hierarchy_kernel(uint2 *d_tree_parent_sib,
                                               const uint32_t *d_morton_codes,
                                               const unsigned int *d_num_per_type,
                                               const unsigned int ntypes,
                                               const unsigned int nleafs,
                                               const unsigned int ninternal)
    {
    // compute the internal node index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per internal node
    if (idx >= ninternal)
        return;

    // get what type of leaf I am
    unsigned int min_idx = 0; // the "0" of the leaf node array
    unsigned int max_idx = 0; // the "N-1" of the leaf node array

    unsigned int node_idx = idx;
    unsigned int origin = 0;
    unsigned int end = 0;

    unsigned int cur_type=0;
    unsigned int active_types=0;
    for (cur_type=0; cur_type < ntypes; ++cur_type)
        {
        // current min index is the previous max index
        min_idx = max_idx;
        // max index adds the number of internal nodes in this type (nleaf - 1)
        const unsigned int cur_nleaf = (d_num_per_type[cur_type] + NLIST_GPU_PARTICLES_PER_LEAF - 1)/NLIST_GPU_PARTICLES_PER_LEAF;
        if (cur_nleaf > 0)
            {
            max_idx += cur_nleaf-1;
            ++active_types;
            }

        // we break the loop if we are in range
        if (idx < max_idx)
            {
            // decrement by 1 to get this back into the number we really need
            --active_types;

            // now, we repurpose the min and max index to now correspond to the *leaf* index.
            // the min index is the minimum *leaf* index
            origin = min_idx + active_types;
            end = max_idx + active_types;
            node_idx += active_types;
            break;
            }
        }

    // enact the magical split determining
    uint3 range_split = determineRangeSplit(d_morton_codes, origin, end, node_idx);
    unsigned int first = range_split.x;
    unsigned int last = range_split.y;
    unsigned int split = range_split.z;

    uint2 children;
    // set the children, shifting ahead by nleafs - cur_type to account for leaf shifting
    // this factor comes out from resetting 0 = N_leaf,i each time, and then remapping this to
    // an internal node
    children.x = (split == first) ? split : (nleafs - active_types + split);
    children.y = ((split + 1) == last) ? (split + 1) : nleafs - active_types + split + 1;

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
    }

/*!
 * \param d_tree_parent_sib Parent and sibling for each node in the tree
 * \param d_morton_codes Morton codes for each leaf node
 * \param d_num_per_type Number of particles per type
 * \param ntypes Number of types
 * \param nleafs Number of leafs
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_gen_hierarchy(uint2 *d_tree_parent_sib,
                                    const uint32_t *d_morton_codes,
                                    const unsigned int *d_num_per_type,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int ninternal,
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
    gpu_nlist_gen_hierarchy_kernel<<<ninternal/run_block_size + 1, run_block_size>>>(d_tree_parent_sib,
                                                                                     d_morton_codes,
                                                                                     d_num_per_type,
                                                                                     ntypes,
                                                                                     nleafs,
                                                                                     ninternal);
    return cudaSuccess;
    }

//! Kernel to bubble up enclosing AABBs to internal nodes from leaf nodes
/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_tree_aabbs AABB array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 *
 * \b Implementation
 * One thread is called per leaf node. The second thread to reach an internal node processes its two children,
 * which guarantees that no node AABB is prematurely processed. The arrival order at a node is controlled by an atomic
 * thread lock in global memory. This locking could be accelerated by using shared memory whenever a node is being
 * processed by threads in the same block.
 *
 * When processing the node, the thread also walks up the tree to find the "rope" that tells a traverser
 * how to navigate the tree. If a query AABB intersects the current node, then the traverser always moves the the left
 * child of the current node. If the AABB does not intersect, it moves along the "rope" to the next portion of the tree.
 * The "rope" is calculated by walking back up the tree to find the earliest ancestor that is a left child of its
 * parent. The rope then goes to that ancestor's sibling. If the root node is reached, then the rope is set to -1 to
 * indicate traversal should be aborted.
 *
 * This kernel also encodes the left child of a node into the AABB for internal nodes. The thread processing the node
 * checks if it arrived from a left child or right child of the node it is processing, and sets the left child of that
 * parent accordingly. A child is indicated by bit shifting, and setting the first bit to 1.
 */
__global__ void gpu_nlist_bubble_aabbs_kernel(unsigned int *d_node_locks,
                                              Scalar4 *d_tree_aabbs,
                                              const uint2 *d_tree_parent_sib,
                                              const unsigned int ntypes,
                                              const unsigned int nleafs)
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

        // if the current sibling is a right child, then the current node is a left child
        bool cur_is_left = (cur_parent_sib.y & 1);

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
            d_tree_aabbs[2*cur_node].w = __int_as_scalar(-1);
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

            // this must always be some internal node, so stash the left child of this node here
            unsigned int left_child_masked = ((cur_is_left ? cur_node : cur_sibling) << 1) | 1;
            cur_lower.w = __int_as_scalar( left_child_masked );

            d_tree_aabbs[2*cur_parent+1] = cur_lower;

            // bump the current node one level
            cur_node = cur_parent;
            }
        }
    while (lock_key == 1);

    }

/*!
 * \param d_node_locks Atomic flags identifying when node has been visited
 * \param d_tree_aabbs AABB array for all tree nodes
 * \param d_tree_parent_sib Parent and sibling indexes of each node
 * \param ntypes Number of particle types
 * \param nleafs Number of leaf nodes
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_bubble_aabbs(unsigned int *d_node_locks,
                                   Scalar4 *d_tree_aabbs,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int ninternal,
                                   const unsigned int block_size)
    {
    cudaMemset(d_node_locks, 0, sizeof(unsigned int)*ninternal);

    gpu_nlist_bubble_aabbs_kernel<<<nleafs/block_size + 1, block_size>>>(d_node_locks,
                                                                         d_tree_aabbs,
                                                                         d_tree_parent_sib,
                                                                         ntypes,
                                                                         nleafs);

    return cudaSuccess;
    }

//! Kernel to rearrange particle data into leaf order for faster traversal
/*!
 * \param d_leaf_xyzf Particle xyz coordinates + particle id in leaf order
 * \param d_leaf_db Particle diameter and body id in leaf order
 * \param d_pos Particle positions
 * \param d_diameter Particle diameters
 * \param d_body Particle body ids
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param Ntot Number of particles owned by this rank
 *
 * \b Implementation
 * One thread per particle is called. Writes are coalesced by writing in leaf order, and reading in a scattered way.
 */
__global__ void gpu_nlist_move_particles_kernel(Scalar4 *d_leaf_xyzf,
                                                Scalar2 *d_leaf_db,
                                                const Scalar4 *d_pos,
                                                const Scalar *d_diameter,
                                                const unsigned int *d_body,
                                                const unsigned int *d_map_tree_pid,
                                                const unsigned int Ntot)
    {
    // get thread index
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= Ntot)
        return;

    // read and write particle data
    unsigned int p_idx = d_map_tree_pid[idx];
    Scalar4 pos_i = d_pos[p_idx];
    d_leaf_xyzf[idx] = make_scalar4(pos_i.x, pos_i.y, pos_i.z, __int_as_scalar(p_idx));

    Scalar2 db = make_scalar2(d_diameter[p_idx], __int_as_scalar(d_body[p_idx]));
    d_leaf_db[idx] = db;
    }

/*!
 * \param d_leaf_xyzf Particle xyz coordinates + particle id in leaf order
 * \param d_leaf_db Particle diameter and body id in leaf order
 * \param d_pos Particle positions
 * \param d_diameter Particle diameters
 * \param d_body Particle body ids
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param Ntot Number of particles owned by this rank
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_move_particles(Scalar4 *d_leaf_xyzf,
                                     Scalar2 *d_leaf_db,
                                     const Scalar4 *d_pos,
                                     const Scalar *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int *d_map_tree_pid,
                                     const unsigned int Ntot,
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

    gpu_nlist_move_particles_kernel<<<Ntot/run_block_size + 1, run_block_size>>>(d_leaf_xyzf,
                                                                                 d_leaf_db,
                                                                                 d_pos,
                                                                                 d_diameter,
                                                                                 d_body,
                                                                                 d_map_tree_pid,
                                                                                 Ntot);
    return cudaSuccess;
    }


//! Kernel for traversing tree to generate neighbor list
/*!
 * \param d_nlist Neighbor list for writing
 * \param d_n_neigh Number of neighbors per particle
 * \param d_last_updated_pos Records current particle positions
 * \param d_conditions Store overflow condition by type
 * \param d_Nmax Maximum number of neighbors allocated by type
 * \param d_head_list Indexes for writing into neighbor list
 * \param N Number of particles
 * \param nghosts Number of ghost particles
 * \param d_map_tree_pid Map leaf index to local particle index
 * \param d_leaf_offset Offset for reading leaf particles by type
 * \param d_tree_roots Index for tree root by type
 * \param d_tree_aabbs Tree AABBs
 * \param nleafs Total number of leafs
 * \param d_leaf_xyzf Leaf position-id array
 * \param d_leaf_db Leaf diameter-body array
 * \param d_pos Particle positions
 * \param d_image_list Translation vectors to check for traversal
 * \param nimages Number of translation vectors to check
 * \param d_r_cut Cutoff radius by type r_cut(i,j)
 * \param r_buff Buffer around cutoff radius
 * \param max_diam Maximum diameter attained by a particle for diameter shifting
 * \param ntypes Number of particle types
 *
 * \b Implementation
 * One thread is launched per particle, but the threads operate on particles in leaf order rather than ParticleData
 * order in order to minimize divergence within a warp (particles in the same leaf should intersect similar parts of the
 * tree). Each thread iterates on the particle types (trees) and queries on all translation vectors using a stackless
 * search. When the query AABB intersects a node AABB, the node AABB is checked to be an internal node or a leaf node.
 * If an internal node, then the traversal advances to that node's left child. If a leaf node, the leaf particles are
 * tested directly to be included in the neighbor list. The node then advances along that leaf node's rope. If the AABB
 * is not intersected, the traversal advances along the rope. This process proceeds until a rope signals that the
 * traversal is complete.
 */
template<unsigned char flags>
__global__ void gpu_nlist_traverse_tree_kernel(unsigned int *d_nlist,
                                               unsigned int *d_n_neigh,
                                               Scalar4 *d_last_updated_pos,
                                               unsigned int *d_conditions,
                                               const unsigned int *d_Nmax,
                                               const unsigned int *d_head_list,
                                               const unsigned int N,
                                               const unsigned int nghosts,
                                               const unsigned int *d_map_tree_pid,
                                               const unsigned int *d_leaf_offset,
                                               const unsigned int *d_tree_roots,
                                               const Scalar4 *d_tree_aabbs,
                                               const unsigned int nleafs,
                                               const Scalar4 *d_leaf_xyzf,
                                               const Scalar2 *d_leaf_db,
                                               const Scalar4 *d_pos,
                                               const Scalar3 *d_image_list,
                                               const unsigned int nimages,
                                               const Scalar *d_r_cut,
                                               const Scalar r_buff,
                                               const Scalar max_diam,
                                               const unsigned int ntypes)
    {
    bool filter_body = flags & 1;
    bool diameter_shift = flags & 2;

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
            Scalar r_cut = d_r_cut[cur_offset + threadIdx.x];
            // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
            s_r_list[cur_offset + threadIdx.x] = (r_cut > Scalar(0.0)) ? r_cut+r_buff : Scalar(-1.0);
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

    // read in the current position
    unsigned int my_pidx = d_map_tree_pid[idx];
    // we only process particles owned by this processor for neighbors
    if (my_pidx >= N)
        return;

    const Scalar4 postype_i = __ldg(d_pos + my_pidx);
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const unsigned int type_i = __scalar_as_int(postype_i.w);

    // fetch the diameter and body out of the leaf texture since it's bound anyway
    const Scalar2 db_i = __ldg(d_leaf_db + idx);
    const Scalar diam_i = db_i.x;
    const unsigned int body_i = __scalar_as_int(db_i.y);

    const unsigned int nlist_head_i = __ldg(d_head_list + my_pidx);

    unsigned int n_neigh_i = 0;
    for (unsigned int cur_pair_type=0; cur_pair_type < ntypes; ++cur_pair_type)
        {
        // Check primary box
        const Scalar r_cut_i = s_r_list[typpair_idx(type_i,cur_pair_type)];

        // Skip this tree type if it is not needed
        if (r_cut_i <= Scalar(0.0))
            continue;

        // stash the r_cutsq before any diameter shifting
        const Scalar r_cutsq_i = r_cut_i*r_cut_i;

        // the rlist to use for the AABB search has to be at least as big as the biggest diameter
        Scalar r_list_i = r_cut_i;
        if (diameter_shift)
            r_list_i += max_diam - Scalar(1.0);

        const unsigned int cur_tree_root = d_tree_roots[cur_pair_type];
        // skip this type if we don't have it
        if (cur_tree_root == NLIST_GPU_INVALID_NODE)
            continue;

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
                const Scalar4 upper_rope = __ldg(d_tree_aabbs + 2*cur_node_idx);
                const Scalar4 lower_np = __ldg(d_tree_aabbs + 2*cur_node_idx+1);

                if (!(aabb_upper.x < lower_np.x
                      || aabb_lower.x > upper_rope.x
                      || aabb_upper.y < lower_np.y
                      || aabb_lower.y > upper_rope.y
                      || aabb_upper.z < lower_np.z
                      || aabb_lower.z > upper_rope.z))
                    {
                    const unsigned int np_child_masked = __scalar_as_int(lower_np.w);

                    if(!(np_child_masked & 1))
                        {
                        // leaf node
                        // all leaves must have at least 1 particle, so we can use this to decide
                        const unsigned int node_head = NLIST_GPU_PARTICLES_PER_LEAF*cur_node_idx - s_leaf_offset[cur_pair_type];
                        const unsigned int n_part = np_child_masked >> 1;
                        for (unsigned int cur_p = node_head; cur_p < node_head + n_part; ++cur_p)
                            {
                            // neighbor j
                            const Scalar4 cur_xyzf = __ldg(d_leaf_xyzf + cur_p);
                            const Scalar3 pos_j = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                            const unsigned int j = __scalar_as_int(cur_xyzf.w);

                            const Scalar2 cur_db = __ldg(d_leaf_db + cur_p);
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
                    else
                        {
                        // internal node, take left child
                        cur_node_idx = (np_child_masked >> 1);
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

/*!
 * \param d_nlist Neighbor list for writing
 * \param d_n_neigh Number of neighbors per particle
 * \param d_last_updated_pos Records current particle positions
 * \param d_conditions Store overflow condition by type
 * \param d_Nmax Maximum number of neighbors allocated by type
 * \param d_head_list Indexes for writing into neighbor list
 * \param N Number of particles
 * \param nghosts Number of ghost particles
 * \param d_map_tree_pid Map leaf index to local particle index
 * \param d_leaf_offset Offset for reading leaf particles by type
 * \param d_tree_roots Index for tree root by type
 * \param d_tree_aabbs Tree AABBs
 * \param nleafs Total number of leafs
 * \param d_leaf_xyzf Leaf position-id array
 * \param d_leaf_db Leaf diameter-body array
 * \param d_pos Particle positions
 * \param d_image_list Translation vectors to check for traversal
 * \param nimages Number of translation vectors to check
 * \param d_r_cut Cutoff radius by type r_cut(i,j)
 * \param r_buff Buffer around cutoff radius
 * \param max_diam Maximum diameter attained by a particle for diameter shifting
 * \param ntypes Number of particle types
 * \param filter_body True if body filtering is enabled
 * \param diameter_shift True if rcut(i,j) should be shifted by the particle diameters
 * \param compute_capability Compute capability of the GPU (in 20, 30, 35 format)
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 * \returns cudaError on failure to texture bind
 *
 * \note Kernel calls are templated on body filtering and diameter shifting for optimization.
 * \note One thread is called for all leaf particles. Some of these threads will die because they correspond to ghost
 *       particles not owned by the rank. Because the leaf particles are sorted, there is no easy way to skip these
 *       particles, and this inefficiency is assumed to be relatively small.
 */
cudaError_t gpu_nlist_traverse_tree(unsigned int *d_nlist,
                                    unsigned int *d_n_neigh,
                                    Scalar4 *d_last_updated_pos,
                                    unsigned int *d_conditions,
                                    const unsigned int *d_Nmax,
                                    const unsigned int *d_head_list,
                                    const unsigned int N,
                                    const unsigned int nghosts,
                                    const unsigned int *d_map_tree_pid,
                                    const unsigned int *d_leaf_offset,
                                    const unsigned int *d_tree_roots,
                                    const Scalar4 *d_tree_aabbs,
                                    const unsigned int nleafs,
                                    const unsigned int ninternal,
                                    const unsigned int nnodes,
                                    const Scalar4 *d_leaf_xyzf,
                                    const Scalar2 *d_leaf_db,
                                    const Scalar4 *d_pos,
                                    const Scalar3 *d_image_list,
                                    const unsigned int nimages,
                                    const Scalar *d_r_cut,
                                    const Scalar r_buff,
                                    const Scalar max_diam,
                                    const unsigned int ntypes,
                                    bool filter_body,
                                    bool diameter_shift,
                                    const unsigned int block_size)
    {
    // shared memory = r_list + Nmax
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = sizeof(Scalar)*typpair_idx.getNumElements() + 2*sizeof(unsigned int)*ntypes;

    if (!filter_body && !diameter_shift)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_nlist_traverse_tree_kernel<0>);
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
                                                                                    d_map_tree_pid,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
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
    else if (filter_body && !diameter_shift)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_nlist_traverse_tree_kernel<1>);
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
                                                                                    d_map_tree_pid,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
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
    else if (!filter_body && diameter_shift)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_nlist_traverse_tree_kernel<2>);
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
                                                                                    d_map_tree_pid,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
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
    else if (filter_body && diameter_shift)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_nlist_traverse_tree_kernel<3>);
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
                                                                                    d_map_tree_pid,
                                                                                    d_leaf_offset,
                                                                                    d_tree_roots,
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

//! Kernel to find divisions between particle types in sorted order
/*!
 * \param d_type_head Index to first type in leaf ordered particles by type
 * \param d_pos Particle positions
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param N Total number of particles on rank (including ghosts)
 *
 * The starting index for each type of particles is the first particle where the left neighbor is not of the same type.
 */
__global__ void gpu_nlist_get_divisions_kernel(unsigned int *d_type_head,
                                               const Scalar4 *d_pos,
                                               const unsigned int *d_map_tree_pid,
                                               const unsigned int N)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    const unsigned int cur_pidx = d_map_tree_pid[idx];
    // get type of the current particle
    const Scalar4 cur_postype = d_pos[cur_pidx];
    const unsigned int cur_type = __scalar_as_int(cur_postype.w);

    // all particles except for the first one should look left
    if (idx > 0)
        {
        const unsigned int left_pidx = d_map_tree_pid[idx - 1];

        // get type of the particle to my left
        const Scalar4 left_postype = d_pos[left_pidx];
        const unsigned int left_type = __scalar_as_int(left_postype.w);

        // if the left has a different type, then this is a type boundary, and the type starts at the current thread index
        if (left_type != cur_type)
            {
            d_type_head[cur_type] = idx + 1; // offset the index +1 so that we can use 0 to mean "none of this found"
            }
        }
    else // the first particle just sets its type to be 1
        {
        d_type_head[cur_type] = 1;
        }
    }

/*!
 * \param d_type_head Index to first type in leaf ordered particles by type
 * \param d_num_per_type Number of particles per type
 * \param d_leaf_offset Offset for reading particles out of leaf order
 * \param d_tree_roots Root node of each tree
 * \param d_pos Particles positions
 * \param d_map_tree_pid ParticleData indexes corresponding to a leaf particle id
 * \param N Total number of particles on rank (including ghosts)
 * \param ntypes Number of types
 * \param block_size Requested thread block size
 *
 * \returns cudaSuccess on completion
 */
cudaError_t gpu_nlist_init_count(unsigned int *d_type_head,
                                 const Scalar4 *d_pos,
                                 const unsigned int *d_map_tree_pid,
                                 const unsigned int N,
                                 const unsigned int ntypes,
                                 const unsigned int block_size)
    {
    // apply the scan
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_get_divisions_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);

    // zero out the head list
    cudaMemset(d_type_head, 0, sizeof(unsigned int)*ntypes);

    // get the head list divisions
    gpu_nlist_get_divisions_kernel<<<N/run_block_size + 1, run_block_size>>>(d_type_head, d_pos, d_map_tree_pid, N);

    return cudaSuccess;
    }

#undef MORTON_CODE_BITS
#undef MORTON_TYPE_MASK_64
#undef MORTON_CODE_N_BINS
