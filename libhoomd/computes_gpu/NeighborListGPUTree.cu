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
#include "NeighborListGPU.cuh"
#include "TextureTools.h"

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;
scalar4_tex_t leaf_xyzf_tex;

/*! \file NeighborListGPUTree.cu
    \brief Defines GPU kernel code for neighbor list tree traversal on the GPU
*/
template<unsigned char flags>
__global__ void gpu_nlist_traverse_tree_kernel(unsigned int *d_nlist,
                                     unsigned int *d_n_neigh,
                                     Scalar4 *d_last_updated_pos,
                                     unsigned int *d_conditions,
                                     const unsigned int *d_Nmax,
                                     const unsigned int *d_head_list,
                                     const Scalar4 *d_pos,
                                     const unsigned int *d_body,
                                     const Scalar *d_diameter,
                                     const unsigned int N,
                                     const AABBTreeGPU *d_aabb_trees,
                                     const AABBNodeGPU *d_aabb_nodes,
                                     const unsigned int *d_aabb_leaf_particles,
                                     const Scalar4 *d_leaf_xyzf,
                                     const Scalar2 *d_leaf_db,
                                     const Scalar3 *d_image_list,
                                     const unsigned int nimages,
                                     const Scalar *d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     bool filter_body)
    {
    filter_body = flags & 1;

    // cache the r_listsq parameters into shared memory
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for per type pair parameters
    extern __shared__ unsigned char s_data[];
    
    // pointer for the r_listsq data
    Scalar *s_r_list = (Scalar *)(&s_data[0]);
    unsigned int *s_Nmax = (unsigned int *)(&s_data[sizeof(Scalar)*num_typ_parameters]);

    // load in the per type pair r_list
    unsigned int max_offset = max(num_typ_parameters, nimages);
    for (unsigned int cur_offset = 0; cur_offset < max_offset; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_r_list[cur_offset + threadIdx.x] = d_r_cut[cur_offset + threadIdx.x]+r_buff;
            }
            
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = d_Nmax[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();
    
    
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return;  
    
    // read in the current position and orientation
    const unsigned int my_pidx = d_aabb_leaf_particles[idx];
    const Scalar4 postype_i = texFetchScalar4(d_pos, pdata_pos_tex, my_pidx);
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const unsigned int type_i = __scalar_as_int(postype_i.w);
    const unsigned int body_i = d_body[my_pidx];
    
//     const unsigned int Nmax_i = s_Nmax[type_i];
    const unsigned int nlist_head_i = d_head_list[my_pidx];
    
    unsigned int n_neigh_i = 0;
    
    for (unsigned int cur_pair_type=0; cur_pair_type < ntypes; ++cur_pair_type)
        {
        // Check primary box
        Scalar r_cut_i = s_r_list[typpair_idx(type_i,cur_pair_type)];
        Scalar r_cutsq_i = r_cut_i*r_cut_i;
        AABBTreeGPU cur_aabb_tree = d_aabb_trees[cur_pair_type];
        
        for (unsigned int cur_image = 0; cur_image < nimages; ++cur_image)
            {
            // can we put the image list in shared memory? 27 x 3 x 4 = 648 B (yes! optimize later)
            Scalar3 pos_i_image = pos_i + d_image_list[cur_image];
            AABBGPU aabb(pos_i_image, r_cut_i);
            
            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree.num_nodes; ++cur_node_idx)
                {
                const AABBNodeGPU *cur_node = &d_aabb_nodes[cur_aabb_tree.node_head + cur_node_idx];
                Scalar4 lower_np = cur_node->lower_np;
                Scalar4 upper_skip = cur_node->upper_skip;
                
                bool overlap = !(   aabb.upper.x < lower_np.x
                                 || aabb.lower.x > upper_skip.x
                                 || aabb.upper.y < lower_np.y
                                 || aabb.lower.y > upper_skip.y
                                 || aabb.upper.z < lower_np.z
                                 || aabb.lower.z > upper_skip.z
                                );
                
                if (overlap)
                    {      
                    unsigned int leaf_num_particles = __scalar_as_int(lower_np.w);                  
                    for (unsigned int cur_p = 0; cur_p < leaf_num_particles; ++cur_p)
                        {
                        // neighbor j
                        Scalar4 cur_xyzf = texFetchScalar4(d_leaf_xyzf, leaf_xyzf_tex, cur_node->leaf_head_idx + cur_p);
                        Scalar3 pos_j = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                        unsigned int j = __scalar_as_int(cur_xyzf.w);
                        
                        Scalar2 cur_db = d_leaf_db[cur_node->leaf_head_idx + cur_p];
                        unsigned int body_j = __scalar_as_int(cur_db.y);

                        bool excluded = (my_pidx == j);

                        if (filter_body && body_i != 0xffffffff)
                            excluded = excluded | (body_i == body_j);
                            
                        if (!excluded)
                            {
                            // compute distance and wrap back into box
                            Scalar3 drij = pos_j - pos_i_image;
                            Scalar dr2 = dot(drij,drij);

                            if (dr2 <= r_cutsq_i)
                                {
                                if (n_neigh_i < s_Nmax[type_i])
                                    {
                                    d_nlist[nlist_head_i + n_neigh_i] = j;
                                    }
                                ++n_neigh_i;
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += __scalar_as_int(upper_skip.w);
                    }
                } // end stackless search  
            } // end loop over images
            
        } // end loop over pair types
        
    d_n_neigh[my_pidx] = n_neigh_i;
    d_last_updated_pos[my_pidx] = d_pos[my_pidx];
    
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
                                     const Scalar4 *d_pos,
                                     const unsigned int *d_body,
                                     const Scalar *d_diameter,
                                     const unsigned int N,
                                     const AABBTreeGPU *d_aabb_trees,
                                     const AABBNodeGPU *d_aabb_nodes,
                                     const unsigned int *d_aabb_leaf_particles,
                                     const Scalar4 *d_leaf_xyzf,
                                     const Scalar2 *d_leaf_db,
                                     const Scalar3 *d_image_list,
                                     const unsigned int nimages,
                                     const Scalar *d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     bool filter_body,
                                     const unsigned int compute_capability,
                                     const unsigned int block_size)
    {
    // shared memory = r_list + Nmax
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = sizeof(Scalar)*typpair_idx.getNumElements() + sizeof(unsigned int)*ntypes;
    
    // bind the neighborlist texture
    if (compute_capability < 35)
        {
        // bind the position texture
        pdata_pos_tex.normalized = false;
        pdata_pos_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
        
        leaf_xyzf_tex.normalized = false;
        leaf_xyzf_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, leaf_xyzf_tex, d_leaf_xyzf, sizeof(Scalar4)*N);
        }
    
    if (!filter_body)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<0>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        gpu_nlist_traverse_tree_kernel<0><<<N/run_block_size + 1, block_size, shared_size>>>(d_nlist,
                                                                                 d_n_neigh,
                                                                                 d_last_updated_pos,
                                                                                 d_conditions,
                                                                                 d_Nmax,
                                                                                 d_head_list,
                                                                                 d_pos,
                                                                                 d_body,
                                                                                 d_diameter,
                                                                                 N,
                                                                                 d_aabb_trees,
                                                                                 d_aabb_nodes,
                                                                                 d_aabb_leaf_particles,
                                                                                 d_leaf_xyzf,
                                                                                 d_leaf_db,
                                                                                 d_image_list,
                                                                                 nimages,
                                                                                 d_r_cut,
                                                                                 r_buff,
                                                                                 ntypes,
                                                                                 filter_body);
        }
    else
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_traverse_tree_kernel<1>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        int run_block_size = min(block_size,max_block_size);
        gpu_nlist_traverse_tree_kernel<1><<<N/run_block_size + 1, block_size, shared_size>>>(d_nlist,
                                                                                 d_n_neigh,
                                                                                 d_last_updated_pos,
                                                                                 d_conditions,
                                                                                 d_Nmax,
                                                                                 d_head_list,
                                                                                 d_pos,
                                                                                 d_body,
                                                                                 d_diameter,
                                                                                 N,
                                                                                 d_aabb_trees,
                                                                                 d_aabb_nodes,
                                                                                 d_aabb_leaf_particles,
                                                                                 d_leaf_xyzf,
                                                                                 d_leaf_db,
                                                                                 d_image_list,
                                                                                 nimages,
                                                                                 d_r_cut,
                                                                                 r_buff,
                                                                                 ntypes,
                                                                                 filter_body);
        }
    return cudaSuccess;
    }
