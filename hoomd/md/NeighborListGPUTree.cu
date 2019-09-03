// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "NeighborListGPUTree.cuh"

#include "hoomd/neighbor/LBVH.cuh"
#include "hoomd/neighbor/LBVHTraverser.cuh"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include "hoomd/extern/cub/cub/cub.cuh"

__global__ void gpu_nlist_mark_types_kernel(unsigned int *d_types,
                                            unsigned int *d_indexes,
                                            unsigned int *d_lbvh_errors,
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
    unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos, ghost_width);

    /*
     * check if the particle is inside the unit cell + ghost layer.
     * we silently ignore ghosts outside of this width, since they could be in bonds, etc.
     */
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
        {
        // error for owned particle
        if (idx < N)
            {
            atomicMax(d_lbvh_errors,idx+1);
            return;
            }
        else // silent for ghosts
            {
            type = NeigborListTypeSentinel;
            }
        }

    d_types[idx] = type;
    d_indexes[idx] = idx;
    }

cudaError_t gpu_nlist_mark_types(unsigned int *d_types,
                                 unsigned int *d_indexes,
                                 unsigned int *d_lbvh_errors,
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
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_mark_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size,max_block_size);
    const unsigned int num_blocks = ((N+nghosts) + run_block_size - 1)/run_block_size;
    gpu_nlist_mark_types_kernel<<<num_blocks, run_block_size>>>(d_types,
                                                                d_indexes,
                                                                d_lbvh_errors,
                                                                d_pos,
                                                                N,
                                                                nghosts,
                                                                box,
                                                                ghost_width);
    return cudaSuccess;
    }

uchar2 gpu_nlist_sort_types(void *d_tmp,
                            size_t &tmp_bytes,
                            unsigned int *d_types,
                            unsigned int *d_sorted_types,
                            unsigned int *d_indexes,
                            unsigned int *d_sorted_indexes,
                            const unsigned int N)
    {
    cub::DoubleBuffer<unsigned int> d_keys(d_types, d_sorted_types);
    cub::DoubleBuffer<unsigned int> d_vals(d_indexes, d_sorted_indexes);

    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N);

    uchar2 swap = make_uchar2(0,0);
    if (d_tmp != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the sorted array (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

__global__ void gpu_nlist_count_types_kernel(unsigned int *d_first,
                                             unsigned int *d_last,
                                             const unsigned int *d_types,
                                             const unsigned int ntypes,
                                             const unsigned int N)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    // my type
    const unsigned int type = d_types[idx];
    // look to left if not first
    const unsigned int left = (idx > 0) ? d_types[idx-1] : NeigborListTypeSentinel;
    // look to right if not last
    const unsigned int right = (idx < N-1) ? d_types[idx+1] : NeigborListTypeSentinel;

    // if left is not same as self (or idx == 0 by use of sentinel), this is the first index in the type
    if (left != type && type < ntypes)
        {
        d_first[type] = idx;
        }
    // if right is not the same as self (or idx == N-1 by use of sentinel), this is the last index in the type
    if (right != type && type < ntypes)
        {
        d_last[type] = idx + 1;
        }
    }

cudaError_t gpu_nlist_count_types(unsigned int *d_first,
                                  unsigned int *d_last,
                                  const unsigned int *d_types,
                                  const unsigned int ntypes,
                                  const unsigned int N,
                                  const unsigned int block_size)

    {
    // initially, fill all cells as empty
    thrust::fill(thrust::device, d_first, d_first+ntypes, NeigborListTypeSentinel);
    cudaMemset(d_last, 0, sizeof(unsigned int)*ntypes);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_count_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    gpu_nlist_count_types_kernel<<<N/run_block_size + 1, run_block_size>>>(d_first,
                                                                           d_last,
                                                                           d_types,
                                                                           ntypes,
                                                                           N);
    return cudaSuccess;
    }

__global__ void gpu_nlist_copy_primitives_kernel(unsigned int *d_traverse_order,
                                                 const unsigned int *d_indexes,
                                                 const unsigned int *d_primitives,
                                                 const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    const unsigned int primitive = d_primitives[idx];
    d_traverse_order[idx] = __ldg(d_indexes + primitive);
    }

cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_copy_primitives_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    gpu_nlist_copy_primitives_kernel<<<N/run_block_size + 1, run_block_size>>>(d_traverse_order,
                                                                               d_indexes,
                                                                               d_primitives,
                                                                               N);
    return cudaSuccess;
    }

struct IsGhost
    {
    IsGhost(unsigned int N_) : N(N_) {}

    __host__ __device__
    bool operator()(const unsigned int idx)
        {
        return (idx >= N);
        }

    unsigned int N;
    };

unsigned int gpu_nlist_remove_ghosts(unsigned int *d_traverse_order,
                                     const unsigned int N,
                                     const unsigned int N_own)
    {
    unsigned int *end = thrust::remove_if(thrust::device, d_traverse_order, d_traverse_order+N, IsGhost(N_own));
    return (end - d_traverse_order);
    }

// explicit templates for neighbor::LBVH with PointMapInsertOp and NullOp
template void neighbor::gpu::lbvh_gen_codes(unsigned int *, unsigned int *, const PointMapInsertOp&,
    const Scalar3, const Scalar3, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_bubble_aabbs(const neighbor::gpu::LBVHData, const PointMapInsertOp&,
    unsigned int *, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_one_primitive(const neighbor::gpu::LBVHData, const PointMapInsertOp&, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
    const ParticleQueryOp&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);

#if 0
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
#endif
