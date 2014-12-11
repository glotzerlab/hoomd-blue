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

#include "AABB.h"
#include "AABBTree.h"


#ifndef __AABB_TREE_GPU_H__
#define __AABB_TREE_GPU_H__

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

using namespace hpmc::detail;

struct AABBGPU
    {
    DEVICE AABBGPU(const Scalar3& pos, Scalar radius)
        {
        upper.x = pos.x + radius;
        upper.y = pos.y + radius;
        upper.z = pos.z + radius;
        lower.x = pos.x - radius;
        lower.y = pos.y - radius;
        lower.z = pos.z - radius;
        }
    Scalar3 upper;
    Scalar3 lower;
    };

struct AABBNodeGPU
    {
    AABBNodeGPU()
        {
        upper_skip = make_scalar4(0.,0.,0.,0.);
        lower_np = make_scalar4(0.,0.,0.,0.);
        leaf_head_idx = 0;
        }        
        
    #ifndef NVCC      
    AABBNodeGPU(const AABB& aabb, unsigned int skip, unsigned int num_particles, int head)
        {
        upper_skip = vec_to_scalar4(aabb.getUpper(), __int_as_scalar(skip));
        lower_np = vec_to_scalar4(aabb.getLower(), __int_as_scalar(num_particles));
        leaf_head_idx = head;
        }
    #endif
    
    Scalar4 upper_skip;
    Scalar4 lower_np;
    unsigned int leaf_head_idx;
    };
    
    
//! Check if two AABBs overlap
/*! \param a First AABB
    \param b Second AABB
    \returns true when the two AABBs overlap, false otherwise
*/
// DEVICE inline bool overlap(const AABBNodeGPU& a, const AABBGPU& b)
//     {
//     return !(   b.upper.x < a.lower_leaf.x
//              || b.lower.x > a.upper_skip.x
//              || b.upper.y < a.lower_leaf.y
//              || b.lower.y > a.upper_skip.y
//              || b.upper.z < a.lower_leaf.z
//              || b.lower.z > a.upper_skip.z
//             );
//     }

struct AABBTreeGPU
    {
    AABBTreeGPU() : num_nodes(0), node_head(0) {}
    
    #ifndef NVCC
    AABBTreeGPU(const AABBTree& tree)
        {
        num_nodes = tree.getNumNodes();
        node_head = 0;
        }
    inline AABBTreeGPU& operator= (const AABBTree& tree)
        {
        num_nodes = tree.getNumNodes();
        node_head = 0;
        return *this;
        }
    #endif
    
    unsigned int num_nodes;
    unsigned int node_head;
    };

#undef DEVICE

#endif //__AABB_TREE_GPU_H__
