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

// needed for temporary scheduleDistanceCheck kernel
#include "NeighborListGPU.cuh"

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
    : NeighborListGPU(sysdef, r_cut, r_buff), m_max_n_local(0)
    {
    // allocate AABB trees to match the number of types, and initialize
    GPUArray<AABBTree> aabb_trees(m_pdata->getNTypes(), m_exec_conf);
    m_aabb_trees.swap(aabb_trees);
    {
        ArrayHandle<AABBTree> h_aabb_trees(m_aabb_trees, access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < m_aabb_trees.getPitch(); ++i)
            h_aabb_trees.data[i] = AABBTree();
    }
    GPUArray<AABBTreeGPU> aabb_trees_gpu(m_pdata->getNTypes(), m_exec_conf); // copied data structure for the GPU
    m_aabb_trees_gpu.swap(aabb_trees_gpu);
    
    // leaf particles is of size N, since all particles are in a leaf    
    GPUArray<unsigned int> aabb_leaf_particles(m_pdata->getN(), m_exec_conf);
    m_aabb_leaf_particles.swap(aabb_leaf_particles);
    GPUArray<Scalar4> leaf_xyzf(m_pdata->getN(), m_exec_conf);
    m_leaf_xyzf.swap(leaf_xyzf);
    GPUArray<Scalar2> leaf_db(m_pdata->getN(), m_exec_conf);
    m_leaf_db.swap(leaf_db);
    
    
    // allocate storage for number of particles per type (including ghosts)
    GPUArray<unsigned int> num_per_type(m_pdata->getNTypes(), m_exec_conf);
    m_num_per_type.swap(num_per_type);
    GPUArray<unsigned int> type_head(m_pdata->getNTypes(), m_exec_conf);
    m_type_head.swap(type_head);
    
    if (this->m_sysdef->getNDimensions() == 3) // 3D periodic = 27 vectors
        {
        GPUArray<Scalar3> image_list(27, m_exec_conf);
        m_image_list.swap(image_list);
        }
    else // 2D periodic = 9 translation vectors
        {
        GPUArray<Scalar3> image_list(9, m_exec_conf);
        m_image_list.swap(image_list);
        }
    
    // allocate AABB Tree memory
    allocateTree(m_max_n_local);

    m_box_changed = true; // by default, assume the box has "changed" at first, so that we always do this action once
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&NeighborListGPUTree::slotBoxChanged, this));
    
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_tree", this->m_exec_conf));
    }

NeighborListGPUTree::~NeighborListGPUTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUTree" << endl;
    m_boxchange_connection.disconnect();
    }

void NeighborListGPUTree::buildNlist(unsigned int timestep)
    {
    if (this->m_prof) this->m_prof->push("AABB Tree");
    // cout<<timestep<<endl;
    // build the trees 
    buildTree();
    
    // now walk the trees
    traverseTree();
    
    if (this->m_prof) this->m_prof->pop();
    }

//! manage the malloc of the AABB list
void NeighborListGPUTree::allocateTree(unsigned int n_local)
    {
    if (n_local > m_max_n_local)
        {
        m_max_n_local = n_local;
        GPUArray<AABB> aabbs(m_max_n_local, m_exec_conf);
        m_aabbs.swap(aabbs);
        
        GPUArray<unsigned int> map_p_global_tree(m_max_n_local, m_exec_conf);
        m_map_p_global_tree.swap(map_p_global_tree);
        }
    }

//! get the number of particles by type (including ghost particles)
void NeighborListGPUTree::getNumPerType()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Histogram");
    
    // zero the arrays
    ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_map_p_global_tree(m_map_p_global_tree, access_location::host, access_mode::overwrite);

    // clear out counters
    unsigned int n_types = m_pdata->getNTypes();
    for (unsigned int i=0; i < n_types; ++i)
        {
        h_num_per_type.data[i] = 0;
        }
    
    // histogram the particles
    unsigned int n_local = m_pdata->getN() + m_pdata->getNGhosts();
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int i=0; i < n_local; ++i)
        {
        unsigned int my_type = __scalar_as_int(h_postype.data[i].w);
        h_map_p_global_tree.data[i] = h_num_per_type.data[my_type]; // global id i is particle num_per_type after head of my_type
        ++h_num_per_type.data[my_type];
        }
    
    // set the head for each type in m_aabbs
    unsigned int local_head = 0;
    for (unsigned int i=0; i < n_types; ++i)
        {
        h_type_head.data[i] = local_head;
        local_head += h_num_per_type.data[i];
        }
        
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void NeighborListGPUTree::buildTree()
    {
    m_exec_conf->msg->notice(4) << "Building AABB tree: " << m_pdata->getN() << " ptls "
                                << m_pdata->getNGhosts() << " ghosts" << endl;
    
    // reallocate the data structures if needed
    unsigned int n_local = m_pdata->getN()+m_pdata->getNGhosts();
    allocateTree(n_local);
    
    // histogram the particle types
    getNumPerType();
    
    // do the build
    if (m_prof) m_prof->push(m_exec_conf,"Build");
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_num_per_type(m_num_per_type, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_map_p_global_tree(m_map_p_global_tree, access_location::host, access_mode::read);
    
    ArrayHandle<AABB> h_aabbs(m_aabbs, access_location::host, access_mode::overwrite);
    ArrayHandle<AABBTree> h_aabb_trees(m_aabb_trees, access_location::host, access_mode::readwrite);
    
    for (unsigned int i=0; i < n_local; ++i)
        {
        // make a point particle AABB
        vec3<Scalar> my_pos(h_postype.data[i]);
        unsigned int my_type = __scalar_as_int(h_postype.data[i].w);
        unsigned int my_aabb_idx = h_type_head.data[my_type] + h_map_p_global_tree.data[i];
        h_aabbs.data[my_aabb_idx] = AABB(my_pos,i);
        }
    
    unsigned int n_types = m_pdata->getNTypes();
    for (unsigned int i=0; i < n_types; ++i) 
        {
        h_aabb_trees.data[i].buildTree(h_aabbs.data + h_type_head.data[i], h_num_per_type.data[i]);
        }
    if (m_prof) m_prof->pop(m_exec_conf);
    
    }

void NeighborListGPUTree::updateImageVectors()
    {
    if (m_box_changed)
        {
        // validate simulation box
        const BoxDim& box = m_pdata->getBox();
        Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
    
        // start by creating a temporary copy of r_cut squared
        Scalar rmax = m_r_cut_max + m_r_buff;
        if ((box.getPeriodic().x && nearest_plane_distance.x <= rmax * 2.0) ||
            (box.getPeriodic().y && nearest_plane_distance.y <= rmax * 2.0) ||
            (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z <= rmax * 2.0))
            {
            m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << endl;
            throw runtime_error("Error updating neighborlist bins");
            }
        
        // boxes must be periodic
        if(!box.getPeriodic().x || !box.getPeriodic().y || (this->m_sysdef->getNDimensions() == 3 && !box.getPeriodic().z))
            {
            m_exec_conf->msg->error() << "nlist: Tree builds are currently only supported in fully periodic geometries"<<endl;
            throw runtime_error("Error traversing neighborlist AABB tree");
            }
        
        ArrayHandle<Scalar3> h_image_list(m_image_list, access_location::host, access_mode::readwrite);
        Scalar3 latt_a = box.getLatticeVector(0);
        Scalar3 latt_b = box.getLatticeVector(1);
        Scalar3 latt_c = box.getLatticeVector(2);

        // iterate on all possible combinations of lattice vectors
        unsigned int latt_idx = 1;
        h_image_list.data[0] = make_scalar3(0.0, 0.0, 0.0);
        for (int i=-1; i <= 1; ++i)
            {
            for (int j=-1; j <=1 ; ++j)
                {
                if(this->m_sysdef->getNDimensions() == 3) // 3D periodic needs another loop
                    {
                    for (int k=-1; k <= 1; ++k)
                        {
                        if (!(i == 0 && j == 0 && k == 0))
                            {
                            h_image_list.data[latt_idx] = Scalar(i)*latt_a + Scalar(j)*latt_b + Scalar(k)*latt_c;
                            ++latt_idx;
                            }
                        }
                    }
                else // 2D periodic
                    {
                    if (!(i == 0 && j == 0))
                        {
                        h_image_list.data[latt_idx] = Scalar(i)*latt_a + Scalar(j)*latt_b;
                        ++latt_idx;
                        }
                    }
                }
            }
        m_box_changed = false;
        }
    }
    
void NeighborListGPUTree::copyCPUtoGPU()
    {
    ArrayHandle<AABBTree> h_aabb_trees_cpu(m_aabb_trees, access_location::host, access_mode::read);
    ArrayHandle<AABBTreeGPU> h_aabb_trees(m_aabb_trees_gpu, access_location::host, access_mode::overwrite);
    unsigned int n_tree_nodes = 0;
    for (unsigned int i=0; i < m_pdata->getNTypes(); ++i)
        {
        h_aabb_trees.data[i].num_nodes = h_aabb_trees_cpu.data[i].getNumNodes();
        h_aabb_trees.data[i].node_head = n_tree_nodes;
        n_tree_nodes += h_aabb_trees.data[i].num_nodes;
        }

    // reallocate if necessary
    if (n_tree_nodes > m_aabb_node_head_idx.getPitch())
        {
        // alternative data struct to the AABBNodeGPU as flat array that can be texture cached
        GPUArray<Scalar4> aabb_node_bounds(2*n_tree_nodes, m_exec_conf);
        m_aabb_node_bounds.swap(aabb_node_bounds);
        
        // stick this index in separate array since it's rarely accessed and improves mem alignment
        GPUArray<unsigned int> aabb_node_head_idx(n_tree_nodes, m_exec_conf);
        m_aabb_node_head_idx.swap(aabb_node_head_idx);
        }            
    
    // copy the nodes into a separate gpu array
    ArrayHandle<unsigned int> h_aabb_leaf_particles(m_aabb_leaf_particles,
                                                    access_location::host,
                                                    access_mode::overwrite);
    
    // trying out this data struct instead
    ArrayHandle<Scalar4> h_aabb_node_bounds(m_aabb_node_bounds, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_aabb_node_head_idx(m_aabb_node_head_idx, access_location::host, access_mode::overwrite);
    
    
    ArrayHandle<Scalar4> h_leaf_xyzf(m_leaf_xyzf, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar2> h_leaf_db(m_leaf_db, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    int leaf_head_idx = 0;
    for (unsigned int i=0; i < m_pdata->getNTypes(); ++i)
        {
        const AABBTree *tree = &h_aabb_trees_cpu.data[i];
        unsigned int head = h_aabb_trees.data[i].node_head;
        for (unsigned int j=0; j < tree->getNumNodes(); ++j)
            {
            const unsigned int leaf_idx = (tree->isNodeLeaf(j)) ? leaf_head_idx : 0;
            AABBNodeGPU node(tree->getNodeAABB(j), tree->getNodeSkip(j), tree->getNodeNumParticles(j), leaf_idx);
            
            // temporary copy into alt struct
            h_aabb_node_bounds.data[2*(head + j)] = node.upper_skip;
            h_aabb_node_bounds.data[2*(head + j) + 1] = node.lower_np;
            h_aabb_node_head_idx.data[head + j] = leaf_idx;
            
            if (tree->isNodeLeaf(j))
                {
                for (unsigned int cur_particle=0; cur_particle < tree->getNodeNumParticles(j); ++cur_particle)
                    {
                    unsigned int my_pidx = tree->getNodeParticleTag(j, cur_particle);
                    Scalar4 my_postype = h_pos.data[my_pidx];
                    
                    h_aabb_leaf_particles.data[leaf_head_idx + cur_particle] = my_pidx;
                    h_leaf_xyzf.data[leaf_head_idx + cur_particle] = make_scalar4(my_postype.x, my_postype.y, my_postype.z, __int_as_scalar(my_pidx));
                    h_leaf_db.data[leaf_head_idx + cur_particle] = make_scalar2(h_diameter.data[my_pidx], __int_as_scalar(h_body.data[my_pidx]));
                    }
                leaf_head_idx += tree->getNodeNumParticles(j);
                }
            }
        }
    }

void NeighborListGPUTree::traverseTree()
    {
    if (m_prof) m_prof->push(m_exec_conf,"Traverse");

    // need to construct a list of box vectors to translate, which is triggered to update on a box resize
    if (m_box_changed)
        {
        updateImageVectors();
        }
    ArrayHandle<Scalar3> d_image_list(m_image_list, access_location::device, access_mode::read);
    
    // perform the copy from the cpu to the gpu format
    // time the data transfer to the GPU as well
    if (m_prof) m_prof->push(m_exec_conf,"copy");
    copyCPUtoGPU();
        
    ArrayHandle<AABBTreeGPU> d_aabb_trees(m_aabb_trees_gpu, access_location::device, access_mode::read);
    
    // acquire handle to alt struct
    ArrayHandle<Scalar4> d_aabb_node_bounds(m_aabb_node_bounds, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_aabb_node_head_idx(m_aabb_node_head_idx, access_location::device, access_mode::read);
    
    ArrayHandle<unsigned int> d_aabb_leaf_particles(m_aabb_leaf_particles, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_leaf_xyzf(m_leaf_xyzf, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_leaf_db(m_leaf_db, access_location::device, access_mode::read);
    
    // acquire particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_last_updated_pos(m_last_pos, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    
    
    // neighborlist data
    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_map_p_global_tree(m_map_p_global_tree, access_location::device, access_mode::read);
    if (m_prof) m_prof->pop(m_exec_conf);
    
    m_tuner->begin();
    gpu_nlist_traverse_tree(d_nlist.data,
                                     d_n_neigh.data,
                                     d_last_updated_pos.data,
                                     d_conditions.data,
                                     d_Nmax.data,
                                     d_head_list.data,
                                     d_pos.data,
                                     d_body.data,
                                     d_diameter.data,
                                     m_pdata->getN(),
                                     d_aabb_trees.data,
                                     d_aabb_node_bounds.data,
                                     d_aabb_node_head_idx.data,
                                     m_aabb_node_head_idx.getPitch(),
                                     d_aabb_leaf_particles.data,
                                     d_leaf_xyzf.data,
                                     d_leaf_db.data,
                                     d_image_list.data,
                                     m_image_list.getPitch(),
                                     d_r_cut.data,
                                     m_r_buff,
                                     m_pdata->getNTypes(),
                                     m_filter_body,
                                     m_exec_conf->getComputeCapability()/10,
                                     m_tuner->getParam());
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
        
    if (m_prof) m_prof->pop(m_exec_conf);
    }
    
void NeighborListGPUTree::scheduleDistanceCheck(unsigned int timestep)
    {
    // prevent against unnecessary calls
    if (! shouldCheckDistance(timestep))
        {
        m_distcheck_scheduled = false;
        return;
        }
    // scan through the particle data arrays and calculate distances
    if (m_prof) m_prof->push(exec_conf, "dist-check");

    // access data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::read);

    // get current global nearest plane distance
    Scalar3 L_g = m_pdata->getGlobalBox().getNearestPlaneDistance();

    // Cutoff distance for inclusion in neighbor list
    Scalar rmax = m_r_cut_max + m_r_buff;
    // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
    Scalar3 lambda = L_g / m_last_L;
    Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
    lambda_min = (lambda_min < lambda.z) ? lambda_min : lambda.z;

    // maximum displacement for each particle (after subtraction of homogeneous dilations)
    Scalar delta_max = (rmax*lambda_min - m_r_cut_max)/Scalar(2.0);
    Scalar maxshiftsq = delta_max > 0  ? delta_max*delta_max : 0;

    ArrayHandle<unsigned int> h_flags(m_flags, access_location::device, access_mode::readwrite);
//     gpu_nlist_needs_update_check_new(h_flags.data,
//                                      d_last_pos.data,
//                                      d_pos.data,
//                                      m_pdata->getN(),
//                                      box,
//                                      maxshiftsq,
//                                      lambda,
//                                      ++m_checkn);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_distcheck_scheduled = true;
    m_last_schedule_tstep = timestep;

    // record synchronization point
    cudaEventRecord(m_event);

    if (m_prof) m_prof->pop(exec_conf);
    }
        
void export_NeighborListGPUTree()
    {
    class_<NeighborListGPUTree, boost::shared_ptr<NeighborListGPUTree>, bases<NeighborListGPU>, boost::noncopyable >
                     ("NeighborListGPUTree", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                    .def("setTuningParam", &NeighborListGPUTree::setTuningParam)
                     ;
    }
    
