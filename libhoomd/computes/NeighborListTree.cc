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

/*! \file NeighborListTree.cc
    \brief Defines NeighborListTree
*/

#include "NeighborListTree.h"
#include "SystemDefinition.h"

#include <boost/bind.hpp>
#include <boost/python.hpp>
using namespace boost;
using namespace boost::python;

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

NeighborListTree::NeighborListTree(boost::shared_ptr<SystemDefinition> sysdef,
                                       Scalar r_cut,
                                       Scalar r_buff)
    : NeighborList(sysdef, r_cut, r_buff), m_max_n_local(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListTree" << endl;

    // allocate AABB trees to match the number of types, and initialize
    GPUArray<AABBTree> aabb_trees(m_pdata->getNTypes(), exec_conf);
    m_aabb_trees.swap(aabb_trees);
    {
        ArrayHandle<AABBTree> h_aabb_trees(m_aabb_trees, access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < m_aabb_trees.getPitch(); ++i)
            h_aabb_trees.data[i] = AABBTree();
    }
    
    // allocate storage for number of particles per type (including ghosts)
    GPUArray<unsigned int> num_per_type(m_pdata->getNTypes(), exec_conf);
    m_num_per_type.swap(num_per_type);
    GPUArray<unsigned int> type_head(m_pdata->getNTypes(), exec_conf);
    m_type_head.swap(type_head);
    
    if (this->m_sysdef->getNDimensions() == 3) // 3D periodic = 27 vectors
        {
        GPUArray< vec3<Scalar> > image_list(27, exec_conf);
        m_image_list.swap(image_list);
        }
    else // 2D periodic = 9 translation vectors
        {
        GPUArray< vec3<Scalar> > image_list(9, exec_conf);
        m_image_list.swap(image_list);
        }
    
    // allocate AABB Tree memory
    allocateTree(m_max_n_local);

    m_box_changed = true; // by default, assume the box has "changed" at first, so that we always do this action once
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&NeighborListTree::slotBoxChanged, this));
    }

NeighborListTree::~NeighborListTree()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListTree" << endl;
    m_boxchange_connection.disconnect();
    }

void NeighborListTree::buildNlist(unsigned int timestep)
    {
    if (this->m_prof) this->m_prof->push("AABB Tree");
    
    // build the trees 
    buildTree();
    
    // now walk the trees
    traverseTree();
    
    if (this->m_prof) this->m_prof->pop();
    }

//! manage the malloc of the AABB list
void NeighborListTree::allocateTree(unsigned int n_local)
    {
    if (n_local > m_max_n_local)
        {
        m_max_n_local = n_local;
        
        GPUArray<AABB> aabbs(m_max_n_local, exec_conf);
        m_aabbs.swap(aabbs);
        
        GPUArray<unsigned int> map_p_global_tree(m_max_n_local, exec_conf);
        m_map_p_global_tree.swap(map_p_global_tree);
        }
    }

//! get the number of particles by type (including ghost particles)
void NeighborListTree::getNumPerType()
    {
    if (this->m_prof) this->m_prof->push("Histogram");
    
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
        
    if (this->m_prof) this->m_prof->pop();
    }

void NeighborListTree::buildTree()
    {
    m_exec_conf->msg->notice(4) << "Building AABB tree: " << m_pdata->getN() << " ptls "
                                << m_pdata->getNGhosts() << " ghosts" << endl;
    
    // reallocate the data structures if needed
    unsigned int n_local = m_pdata->getN()+m_pdata->getNGhosts();
    allocateTree(n_local);
    
    // histogram the particle types
    getNumPerType();
    
    // do the build
    if (this->m_prof) this->m_prof->push("Build");
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
    if (this->m_prof) this->m_prof->pop();
    
    }

void NeighborListTree::traverseTree()
    {
    if (this->m_prof) this->m_prof->push("Traverse");
    
    // acquire particle data
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    
    ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);

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

    // need to construct a list of box vectors to translate, which is triggered to update on a box resize
    const unsigned int n_images = m_image_list.getPitch(); // 27 (3D) or 9 (2D)
    ArrayHandle< vec3<Scalar> > h_image_list(m_image_list, access_location::host, access_mode::readwrite);
    if (m_box_changed)
        {
            vec3<Scalar> latt_a = vec3<Scalar>(box.getLatticeVector(0));
            vec3<Scalar> latt_b = vec3<Scalar>(box.getLatticeVector(1));
            vec3<Scalar> latt_c = vec3<Scalar>(box.getLatticeVector(2));
        
            // iterate on all possible combinations of lattice vectors
            unsigned int latt_idx = 1;
            h_image_list.data[0] = vec3<Scalar>(0.0, 0.0, 0.0);
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
    
    // tree data
    ArrayHandle<AABBTree> h_aabb_trees(m_aabb_trees, access_location::host, access_mode::read);
    ArrayHandle<AABB> h_aabbs(m_aabbs, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_type_head(m_type_head, access_location::host, access_mode::read);
    
    // neighborlist data
    ArrayHandle<unsigned int> h_head_list(m_head_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    
    // Loop over all particles
    for (unsigned int i=0; i < m_pdata->getN(); ++i)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
        unsigned int type_i = __scalar_as_int(postype_i.w);
        unsigned int body_i = h_body.data[i];
        
        unsigned int Nmax_i = h_Nmax.data[type_i];
        unsigned int nlist_head_i = h_head_list.data[i];
        
        unsigned int n_neigh_i = 0;
        for (unsigned int cur_pair_type=0; cur_pair_type < m_pdata->getNTypes(); ++cur_pair_type)
            {
            // Check primary box
            Scalar r_cut_i = h_r_cut.data[m_typpair_idx(type_i,cur_pair_type)]+m_r_buff;
            Scalar r_cutsq_i = r_cut_i*r_cut_i;
            AABBTree *cur_aabb_tree = &h_aabb_trees.data[cur_pair_type];

            for (unsigned int cur_image = 0; cur_image < n_images; ++cur_image)
                {
                vec3<Scalar> pos_i_image = pos_i + h_image_list.data[cur_image];
                AABB aabb = AABB(pos_i_image, r_cut_i);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
                    {
                    if (overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                        {                        
                        if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                                {
                                // neighbor j
                                unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);
                                
                                // skip self-interaction always
                                if (i == j)
                                    continue;
                                bool excluded = (i == j);

                                if (m_filter_body && body_i != NO_BODY)
                                    excluded = excluded | (body_i == h_body.data[j]);
                                    
                                if (!excluded)
                                    {
                                    // compute distance and wrap back into box
                                    Scalar4 postype_j = h_postype.data[j];
                                    Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                                                   - vec_to_scalar3(pos_i_image);
                                    Scalar dr_sq = dot(drij,drij);

                                    if (dr_sq <= r_cutsq_i)
                                        {
                                        if (m_storage_mode == full || i < j)
                                            {
                                            if (n_neigh_i < Nmax_i)
                                                h_nlist.data[nlist_head_i + n_neigh_i] = j;
                                            else
                                                h_conditions.data[type_i] = max(h_conditions.data[type_i], n_neigh_i+1);

                                            ++n_neigh_i;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                        }
                    } // end stackless search
                } // end loop over images
            } // end loop over pair types
            h_n_neigh.data[i] = n_neigh_i;
        } // end loop over particles
    
    if (this->m_prof) this->m_prof->pop();
    }

void export_NeighborListTree()
    {
    class_<NeighborListTree, boost::shared_ptr<NeighborListTree>, bases<NeighborList>, boost::noncopyable >
                     ("NeighborListTree", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     ;
    }
