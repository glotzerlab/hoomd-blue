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
#include "AABBTreeGPU.h"

#define PARTICLES_PER_LEAF 1        // max number of particles in a leaf node, must be power of two

using namespace hpmc::detail;

//! Kernel driver to traverse tree and build neighbor list on the device
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
                                     const Scalar4 *d_aabb_node_bounds,
                                     const unsigned int *d_aabb_node_head_idx,
                                     const unsigned int n_nodes,
                                     const unsigned int *d_aabb_leaf_particles,
                                     const Scalar4 *d_leaf_xyzf,
                                     const Scalar2 *d_leaf_tdb,
                                     const Scalar3 *d_image_list,
                                     const unsigned int nimages,
                                     const Scalar *d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     bool filter_body,
                                     const unsigned int compute_capability,
                                     const unsigned int block_size);
                                     
//! Kernel driver to generate morton codes for particles and reorder by type
cudaError_t gpu_nlist_morton_codes(unsigned int *d_morton_codes,
                                   unsigned int *d_particle_ids,
                                   const Scalar4 *d_pos,
                                   const unsigned int *d_map_p_global_tree,
                                   const unsigned int N,
                                   const unsigned int *d_type_head,
                                   const unsigned int ntypes,
                                   const BoxDim& box,
                                   const Scalar3 ghost_width,
                                   const unsigned int block_size);

//! Wrapper to Thrust sort for morton codes
cudaError_t gpu_nlist_morton_sort(unsigned int *d_morton_codes,
                                  unsigned int *d_leaf_particles,
                                  const unsigned int *h_num_per_type,
                                  const unsigned int ntypes);
                                  
//! Kernel driver to merge the bottom layers of particles into leaf nodes
cudaError_t gpu_nlist_merge_particles(Scalar4 *d_leaf_aabbs,
                                      unsigned int *d_morton_codes_red,
                                      const unsigned int *d_morton_codes,
                                      const Scalar4 *d_pos,
                                      const unsigned int *d_num_per_type,
                                      const unsigned int ntypes,
                                      const unsigned int *d_leaf_particles,
                                      const unsigned int *d_leaf_offset,
                                      const unsigned int *d_type_head,
                                      const unsigned int N,
                                      const unsigned int nleafs,
                                      const unsigned int block_size);

//! Kernel driver to generate the AABB tree hierarchy from morton codes
cudaError_t gpu_nlist_gen_hierarchy(unsigned int *d_leaf_parents,
                                    unsigned int *d_node_parents,
                                    uint2 *d_node_children,
                                    const unsigned int *d_morton_codes,
                                    const unsigned int *d_type_head,
                                    const unsigned int N,
                                    const unsigned int ntypes,
                                    const unsigned int nleafs,
                                    const unsigned int block_size);
                                    
cudaError_t gpu_nlist_bubble_aabbs(unsigned int *d_node_locks,
                                   Scalar4 *d_tree_aabbs,
                                   const unsigned int *d_leaf_parents,
                                   const unsigned int *d_node_parents,
                                   const uint2 *d_node_children,
                                   const unsigned int ntypes,
                                   const unsigned int nleafs,
                                   const unsigned int block_size);

#endif //__NEIGHBORLISTGPUTREE_CUH__
