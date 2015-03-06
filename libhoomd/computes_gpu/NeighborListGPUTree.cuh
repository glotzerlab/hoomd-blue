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

#ifndef __NEIGHBORLISTGPUTREE_CUH__
#define __NEIGHBORLISTGPUTREE_CUH__

/*! \file NeighborListGPUTree.cuh
    \brief Declares GPU kernel code for neighbor list tree traversal on the GPU
*/

#include <cuda_runtime.h>

#include "HOOMDMath.h"
#include "ParticleData.cuh"
#include "Index1D.h"
                                     
//! Kernel driver to generate morton codes for particles and reorder by type
cudaError_t gpu_nlist_morton_codes(unsigned int *d_morton_codes,
                                   int *d_morton_conditions,
                                   const Scalar4 *d_pos,
                                   const unsigned int *d_map_tree_global,
                                   const unsigned int N,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size);

//! Wrapper to Thrust sort for morton codes
cudaError_t gpu_nlist_morton_sort(unsigned int *d_morton_codes,
                                  unsigned int *d_map_tree_global,
                                  const unsigned int *h_num_per_type,
                                  const unsigned int ntypes);
                                  
//! Kernel driver to merge the bottom layers of particles into leaf nodes
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
                                      const unsigned int block_size);

//! Kernel driver to generate the AABB tree hierarchy from morton codes
cudaError_t gpu_nlist_gen_hierarchy(unsigned int *d_node_left_child,
                                    uint2 *d_tree_parent_sib,
                                    const unsigned int *d_morton_codes,
                                    const unsigned int *d_type_head,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int block_size);
                                    
cudaError_t gpu_nlist_bubble_aabbs(unsigned int *d_node_locks,
                                   Scalar4 *d_tree_aabbs,
                                   const uint2 *d_tree_parent_sib,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int block_size);

cudaError_t gpu_nlist_move_particles(Scalar4 *d_leaf_xyzf,
                                     Scalar2 *d_leaf_db,
                                     const Scalar4 *d_pos,
                                     const Scalar *d_diameter,
                                     const unsigned int *d_body,
                                     const unsigned int *d_map_tree_global,
                                     const unsigned int N,
                                     const unsigned int block_size);

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
                                    const unsigned int block_size);
                                     
cudaError_t gpu_nlist_map_particles_gen_mask(unsigned int *d_type_mask,
                                             const Scalar4 *d_pos,
                                             const unsigned int N,
                                             const unsigned int type,
                                             const unsigned int block_size);
                                                                                  
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
                                    const unsigned int block_size);

#endif //__NEIGHBORLISTGPUTREE_CUH__
