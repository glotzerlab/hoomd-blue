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

/*! \file NeighborListGPUTree.cc
    \brief Defines NeighborListGPUTree
*/

#include "NeighborListGPUTree.h"
#include "NeighborListGPUTree.cuh"

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

NeighborListGPUTree::NeighborListGPUTree(boost::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff)
    : NeighborListGPU(sysdef, r_cut, r_buff), m_n_images(0), m_type_changed(false), m_box_changed(true),
      m_max_num_changed(false), m_n_leaf(0), m_n_internal(0), m_n_node(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUTree" << endl;
    
    m_num_type_change_conn = m_pdata->connectNumTypesChange(bind(&NeighborListGPUTree::slotNumTypesChanged, this));
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&NeighborListGPUTree::slotBoxChanged, this));
    m_max_numchange_conn = m_pdata->connectMaxParticleNumberChange(bind(&NeighborListGPUTree::slotMaxNumChanged, this));
    
    m_tuner_morton.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_morton_codes", this->m_exec_conf));
    m_tuner_merge.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_merge_particles", this->m_exec_conf));
    m_tuner_hierarchy.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_gen_hierarchy", this->m_exec_conf));
    m_tuner_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_bubble_aabbs", this->m_exec_conf));
    m_tuner_move.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_move_particles", this->m_exec_conf));
    m_tuner_map.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_map_particles", this->m_exec_conf));
    m_tuner_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_traverse_tree", this->m_exec_conf));
    
    allocateTree();
    
    calcTypeBits();
    }

NeighborListGPUTree::~NeighborListGPUTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUTree" << endl;
    m_num_type_change_conn.disconnect();
    m_boxchange_connection.disconnect();
    m_max_numchange_conn.disconnect();
    }

void NeighborListGPUTree::buildNlist(unsigned int timestep)
    {
    // allocate the tree memory as needed based on the mapping
    setupTree();

    // build the tree
    buildTree();
    
    // walk with the tree
    traverseTree();
    }

void NeighborListGPUTree::allocateTree()
    {
    // allocate per particle memory
    GPUArray<uint64_t> morton_codes(m_pdata->getMaxN(), m_exec_conf);
    m_morton_codes.swap(morton_codes);
    GPUArray<uint64_t> morton_codes_alt(m_pdata->getMaxN(), m_exec_conf);
    m_morton_codes_alt.swap(morton_codes_alt);
            
    GPUArray<unsigned int> map_tree_global(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_global.swap(map_tree_global);
    GPUArray<unsigned int> map_tree_global_alt(m_pdata->getMaxN(), m_exec_conf);
    m_map_tree_global_alt.swap(map_tree_global_alt);
    
    GPUArray<Scalar4> leaf_xyzf(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_xyzf.swap(leaf_xyzf);
    
    GPUArray<Scalar2> leaf_db(m_pdata->getMaxN(), m_exec_conf);
    m_leaf_db.swap(leaf_db);
    
    // allocate per type memory
    GPUArray<unsigned int> leaf_offset(m_pdata->getNTypes(), m_exec_conf);
    m_leaf_offset.swap(leaf_offset);

    GPUArray<unsigned int> tree_roots(m_pdata->getNTypes(), m_exec_conf);
    m_tree_roots.swap(tree_roots);

    GPUArray<unsigned int> num_per_type(m_pdata->getNTypes(), m_exec_conf);
    m_num_per_type.swap(num_per_type);

    GPUArray<unsigned int> type_head(m_pdata->getNTypes(), m_exec_conf);
    m_type_head.swap(type_head);
    
    // allocate the tree memory to default lengths of 0 (will be resized later)
    // we use a GPUVector instead of GPUArray for the amortized resizing
    GPUVector<uint2> tree_parent_sib(m_exec_conf);
    m_tree_parent_sib.swap(tree_parent_sib);
    
    // holds two Scalar4s per node in tree
    GPUVector<Scalar4> tree_aabbs(m_exec_conf);
    m_tree_aabbs.swap(tree_aabbs);
    
    // we really only need as many morton codes as we have leafs
    GPUVector<uint32_t> morton_codes_red(m_exec_conf);
    m_morton_codes_red.swap(morton_codes_red);
    
    // left children of all internal nodes
    GPUVector<unsigned int> node_left_child(m_exec_conf);
    m_node_left_child.swap(node_left_child);

    // 1 / 0 locks for traversing up the tree
    GPUVector<unsigned int> node_locks(m_exec_conf);
    m_node_locks.swap(node_locks);
    
    // conditions
    GPUFlags<int> morton_conditions(m_exec_conf);
    m_morton_conditions.swap(morton_conditions);
    }

//! memory management for tree and particle mapping
void NeighborListGPUTree::setupTree()
    {    
    // increase arrays that depend on the local number of particles
    if (m_max_num_changed)
        {
        m_morton_codes.resize(m_pdata->getMaxN());
        m_morton_codes_alt.resize(m_pdata->getMaxN());
        m_map_tree_global.resize(m_pdata->getMaxN());
        m_map_tree_global_alt.resize(m_pdata->getMaxN());
        m_leaf_xyzf.resize(m_pdata->getMaxN());
        m_leaf_db.resize(m_pdata->getMaxN());
        
        // all done with the particle data reallocation
        m_max_num_changed = false;
        }
     
    // allocate memory that depends on type    
    if (m_type_changed)
        {
        m_leaf_offset.resize(m_pdata->getNTypes());
        m_tree_roots.resize(m_pdata->getNTypes());
        m_num_per_type.resize(m_pdata->getNTypes());
        m_type_head.resize(m_pdata->getNTypes());
        
        // get the number of bits needed to represent all the types
        calcTypeBits();
        
        // all done with the type reallocation
        m_type_changed = false;
        }
        
    if (m_box_changed)
        {
        updateImageVectors();
        m_box_changed = false;
        }
    }

//! get the number of bits needed to represent all the types
inline void NeighborListGPUTree::calcTypeBits()
    {
    if (m_pdata->getNTypes() > 1)
        {
        unsigned int n_type_bits = 0;
        
        // start with the maximum type id that there can be
        unsigned int tmp = m_pdata->getNTypes() - 1;
    
        // see how many times you can bit shift
        while (tmp >>= 1)
            {
            ++n_type_bits;
            }
    
        // add one to get the number of bits needed (rounding up int logarithm)
        m_n_type_bits = n_type_bits + 1;
        }
    else
        {
        // if there is only one type, you don't need to do any sorting
        m_n_type_bits = 0;
        }
    }

//! map particle ids by type
void NeighborListGPUTree::mapParticlesByType()
    {
    if (m_prof) m_prof->push(m_exec_conf,"map");
    
    
    if (m_pdata->getNTypes() > 1)
        {
        // first do the easily parallelized stuff on the gpu
            {
            ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tree_roots(m_tree_roots, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::read);
            m_tuner_map->begin();
            gpu_nlist_map_particles(d_type_head.data,
                                    d_num_per_type.data,
                                    d_leaf_offset.data,
                                    d_tree_roots.data,
                                    d_pos.data,
                                    d_map_tree_global.data,
                                    m_pdata->getN() + m_pdata->getNGhosts(),
                                    m_pdata->getNTypes(),
                                    m_tuner_map->getParam());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_map->end();
            }


        // then do the hard to parallelize stuff on the cpu because the number of types is small
        // and you don't get much out of doing scans on the data
            {
            // the number of leafs is the first tree root
            ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::readwrite);
    
            // loop through types and get the total number of leafs
            m_n_leaf = 0;
            unsigned int total_offset = 0;
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                m_n_leaf += h_tree_roots.data[cur_type];
        
                // accumulate the offset of all types
                const unsigned int cur_offset = h_leaf_offset.data[cur_type];
                h_leaf_offset.data[cur_type] = total_offset;
                total_offset += cur_offset;
                }
    
            // each tree has Nleaf,i - 1 internal nodes
            // a binary radix tree has N_leaf - N_types internal nodes
            m_n_internal = m_n_leaf - m_pdata->getNTypes();
            m_n_node = m_n_leaf + m_n_internal;
    
            // now loop over the roots one more time, and set each of them
            unsigned int leaf_head = 0;
            for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
                {
                const unsigned int n_leaf_i = h_tree_roots.data[cur_type];
                if (n_leaf_i == 1)
                    {
                    h_tree_roots.data[cur_type] = leaf_head;
                    }
                else
                    {
                    h_tree_roots.data[cur_type] = m_n_leaf + (leaf_head - cur_type);
                    }
                leaf_head += n_leaf_i;
                }
            }
        }
    else
        {
        ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_leaf_offset(m_leaf_offset, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tree_roots(m_tree_roots, access_location::host, access_mode::overwrite);
        
        // with one type, we don't need to do anything fancy
        // type head is the first particle
        h_type_head.data[0] = 0;
        
        // num per type is all the particles in the rank
        h_num_per_type.data[0] = m_pdata->getN() + m_pdata->getNGhosts();
        
        // there is no leaf offset
        h_leaf_offset.data[0] = 0;
        
        // number of leafs is for all particles
        m_n_leaf = (m_pdata->getN() + m_pdata->getNGhosts() + NLIST_PARTICLES_PER_LEAF - 1)/NLIST_PARTICLES_PER_LEAF;
        
        // number of internal nodes is one less than number of leafs
        m_n_internal = m_n_leaf - 1;
        m_n_node = m_n_leaf + m_n_internal;
        
        // the root is the end of the leaf list if multiple leafs, otherwise the only leaf
        h_tree_roots.data[0] = (m_n_leaf > 1) ? m_n_leaf : 0;
        }
        
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! build tree on gpu    
void NeighborListGPUTree::buildTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Build tree");
    
    // step one: morton code calculation
    calcMortonCodes();
    
    // step two: particle sorting
    sortMortonCodes();
    
    // step three: map the particles by type
    mapParticlesByType();
    
    // (re-) allocate memory that depends on tree size
    // GPUVector should only do this as needed
    m_tree_parent_sib.resize(m_n_node);
    m_tree_aabbs.resize(2*m_n_node);
    m_morton_codes_red.resize(m_n_leaf);
    m_node_left_child.resize(m_n_internal);
    m_node_locks.resize(m_n_internal);
    
    // step four: merge leaf particles into aabbs by morton code
    mergeLeafParticles();   
    
    // step five: hierarchy generation from morton codes
    genTreeHierarchy();
    
    // step six: bubble up the aabbs
    bubbleAABBs();
            
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! calculate the morton codes for each particle
void NeighborListGPUTree::calcMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Morton codes");
    // particle data and where to write it
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::read);

    ArrayHandle<uint64_t> d_morton_codes(m_morton_codes, access_location::device, access_mode::overwrite);
    
    const BoxDim& box = m_pdata->getBox();

    Scalar ghost_layer_width = m_r_cut_max + m_r_buff;
    if (m_diameter_shift)
        ghost_layer_width += m_d_max - Scalar(1.0);
        
    Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
    if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
    if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
    if (this->m_sysdef->getNDimensions() == 3 && !box.getPeriodic().z)
        {
        ghost_width.z = ghost_layer_width;
        }


    m_morton_conditions.resetFlags(-1);

    m_tuner_morton->begin();  
    gpu_nlist_morton_codes(d_morton_codes.data,
                           d_map_tree_global.data,
                           m_morton_conditions.getDeviceFlags(),
                           d_pos.data,
                           m_pdata->getN(),
                           m_pdata->getNGhosts(),
                           box,
                           ghost_width,
                           m_tuner_morton->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_morton->end();
    
    const int morton_conditions = m_morton_conditions.readFlags();
    if (morton_conditions >= 0)
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        Scalar4 post_i = h_pos.data[morton_conditions];
        Scalar3 pos_i = make_scalar3(post_i.x, post_i.y, post_i.z);
        Scalar3 f = box.makeFraction(pos_i, ghost_width);
        m_exec_conf->msg->error() << "nlist: Particle " << morton_conditions << " is out of bounds "
                                  << "(x: " << post_i.x << ", y: " << post_i.y << ", z: " << post_i.z
                                  << ", fx: "<< f.x <<", fy: "<<f.y<<", fz:"<<f.z<<")"<<endl;
        throw runtime_error("Error updating neighborlist");
        }
                           
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! sort the morton codes within a type using thrust libraries    
void NeighborListGPUTree::sortMortonCodes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Sort");

    bool swap_morton = false;
    bool swap_map = false;    
        {
        ArrayHandle<uint64_t> d_morton_codes(m_morton_codes, access_location::device, access_mode::readwrite);
        ArrayHandle<uint64_t> d_morton_codes_alt(m_morton_codes_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_map_tree_global_alt(m_map_tree_global_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::read);
        
        gpu_nlist_morton_sort(d_morton_codes.data,
                              d_morton_codes_alt.data,
                              d_map_tree_global.data,
                              d_map_tree_global_alt.data,
                              m_tmp_allocator,
                              swap_morton,
                              swap_map,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_type_bits);    
        }
        
    // we want the sorted data in the real data because the alt is just a tmp holder
    if (swap_morton)
        {
        m_morton_codes.swap(m_morton_codes_alt);
        }
        
    if (swap_map)
        {
        m_map_tree_global.swap(m_map_tree_global_alt);
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }
    
//! merge leaf particles by ID into AABBs    
void NeighborListGPUTree::mergeLeafParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Leaf merge");
    
    // particle position data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_type_head(m_type_head, access_location::device, access_mode::read);
    
    // leaf particle data
    ArrayHandle<uint64_t> d_morton_codes(m_morton_codes, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::read);
    
    // tree aabbs and reduced morton codes to overwrite
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::overwrite);
    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::overwrite);
    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);
    
    m_tuner_merge->begin();  
    gpu_nlist_merge_particles(d_tree_aabbs.data,
                              d_morton_codes_red.data,
                              d_tree_parent_sib.data,
                              d_morton_codes.data,
                              d_pos.data,
                              d_num_per_type.data,
                              m_pdata->getNTypes(),
                              d_map_tree_global.data,
                              d_leaf_offset.data,
                              d_type_head.data,
                              m_pdata->getN() + m_pdata->getNGhosts(),
                              m_n_leaf,
                              m_tuner_merge->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_merge->end();
    
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! generate the parent/child/sibling relationships using the morton codes    
/*!
 * This function should always be called alongside bubbleAABBs to generate a complete hierarchy.
 * genTreeHierarchy saves only the left children of the nodes for downward traversal because bubbleAABBs
 * saves the right child as a rope.
 */
void NeighborListGPUTree::genTreeHierarchy()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Hierarchy");
    
    // don't bother to process if there are no internal nodes
    if (!m_n_internal)
        return;    
        
    ArrayHandle<unsigned int> d_node_left_child(m_node_left_child, access_location::device, access_mode::overwrite);
    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::overwrite);
    
    ArrayHandle<uint32_t> d_morton_codes_red(m_morton_codes_red, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_num_per_type(m_num_per_type, access_location::device, access_mode::read);
    
    m_tuner_hierarchy->begin();  
    gpu_nlist_gen_hierarchy(d_node_left_child.data,
                            d_tree_parent_sib.data,
                            d_morton_codes_red.data,
                            d_num_per_type.data,
                            m_pdata->getNTypes(),
                            m_n_leaf,
                            m_tuner_hierarchy->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_hierarchy->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! walk up the tree from the leaves, and assign stackless ropes for traversal, and conservative AABBs    
void NeighborListGPUTree::bubbleAABBs()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Bubble");
    ArrayHandle<unsigned int> d_node_locks(m_node_locks, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::readwrite);
    
    ArrayHandle<uint2> d_tree_parent_sib(m_tree_parent_sib, access_location::device, access_mode::read);
    
    m_tuner_bubble->begin();  
    gpu_nlist_bubble_aabbs(d_node_locks.data,
                           d_tree_aabbs.data,
                           d_tree_parent_sib.data,
                           m_pdata->getNTypes(),
                           m_n_leaf,
                           m_tuner_bubble->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_bubble->end();
    
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! rearrange the leaf positions in memory to make xyzf and tdb for faster traversal
void NeighborListGPUTree::moveLeafParticles()
    {    
    if (m_prof) m_prof->push(m_exec_conf,"move");
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::overwrite); 
    
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::read);
    
    m_tuner_move->begin();  
    gpu_nlist_move_particles(d_leaf_xyzf.data,
                             d_leaf_db.data,
                             d_pos.data,
                             d_diameter.data,
                             d_body.data,
                             d_map_tree_global.data,
                             m_pdata->getN() + m_pdata->getNGhosts(),
                             m_tuner_move->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_move->end();
    
    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! compute the image vectors for translating periodically around tree
void NeighborListGPUTree::updateImageVectors()
    {
    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    // check that rcut fits in the box
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
    Scalar rmax = m_r_cut_max + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);
        
    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
        (this->m_sysdef->getNDimensions() == 3 && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << endl;
        throw runtime_error("Error updating neighborlist bins");
        }

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (periodic.x + periodic.y + periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }
    
    // reallocate memory if necessary
    if (m_n_images > m_image_list.getPitch())
        {
        GPUArray<Scalar3> image_list(m_n_images, m_exec_conf);
        m_image_list.swap(image_list);
        }
    
    ArrayHandle<Scalar3> h_image_list(m_image_list, access_location::host, access_mode::overwrite);
    Scalar3 latt_a = box.getLatticeVector(0);
    Scalar3 latt_b = box.getLatticeVector(1);
    Scalar3 latt_c = box.getLatticeVector(2);
    
    // there is always at least 1 image, which we put as our first thing to look at
    h_image_list.data[0] = make_scalar3(0.0, 0.0, 0.0);
    
    // iterate over all other combinations of images, skipping those that are 
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x) continue;
                    if (j != 0 && !periodic.y) continue;
                    if (k != 0 && !periodic.z) continue;
                    
                    h_image_list.data[n_images] = Scalar(i) * latt_a + Scalar(j) * latt_b + Scalar(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }
    }
    
//! traverse the tree built on the GPU using two kernels
void NeighborListGPUTree::traverseTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Traverse");

    // move the leaf particles into leaf_xyzf and leaf_tdb for fast traversal
    moveLeafParticles();

    // neighborlist data
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_updated_pos(m_last_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);
    
    // tree data    
    ArrayHandle<unsigned int> d_map_tree_global(m_map_tree_global, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_leaf_offset(m_leaf_offset, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tree_roots(m_tree_roots, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_node_left_child(m_node_left_child, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_tree_aabbs(m_tree_aabbs, access_location::device, access_mode::read);
    
    // tree particle data
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::read);
    
    // particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    
    // images
    ArrayHandle<Scalar3> d_image_list(m_image_list, access_location::device, access_mode::read);
    
    // pairwise cutoffs
    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    
    m_tuner_traverse->begin();
    gpu_nlist_traverse_tree(d_nlist.data,
                            d_n_neigh.data,
                            d_last_updated_pos.data,
                            d_conditions.data,
                            d_Nmax.data,
                            d_head_list.data,
                            m_pdata->getN(),
                            m_pdata->getNGhosts(),
                            d_map_tree_global.data,
                            d_leaf_offset.data,
                            d_tree_roots.data,
                            d_node_left_child.data,
                            d_tree_aabbs.data,
                            m_n_leaf,
                            m_n_internal,
                            m_n_node,
                            d_leaf_xyzf.data,
                            d_leaf_db.data,
                            d_pos.data,
                            d_image_list.data,
                            m_image_list.getPitch(),
                            d_r_cut.data,
                            m_r_buff,
                            m_d_max,
                            m_pdata->getNTypes(),
                            m_filter_body,
                            m_diameter_shift,
                            m_exec_conf->getComputeCapability()/10,
                            m_tuner_traverse->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_traverse->end();
            
    if (m_prof) m_prof->pop(m_exec_conf);
    }
        
void export_NeighborListGPUTree()
    {
    class_<NeighborListGPUTree, boost::shared_ptr<NeighborListGPUTree>, bases<NeighborListGPU>, boost::noncopyable >
                     ("NeighborListGPUTree", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >());
    }
    
