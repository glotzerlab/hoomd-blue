// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborList.h"
#include "hoomd/AABBTree.h"
#include <vector>

/*! \file NeighborListTree.h
    \brief Declares the NeighborListTree class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __NEIGHBORLISTTREE_H__
#define __NEIGHBORLISTTREE_H__

//! Efficient neighbor list build on the CPU using BVH trees
/*!
 * A bounding volume hierarchy (BVH) tree is a binary search tree. It is constructed from axis-aligned bounding boxes
 * (AABBs). The AABB for a node in the tree encloses all child AABBs. A leaf AABB holds multiple particles. The tree
 * is constructed in a balanced way using a heuristic to minimize AABB volume. We build one tree per particle type,
 * and use point AABBs for the particles. The neighbor list is built by traversing down the tree with an AABB
 * that encloses the pairwise cutoff for the particle. Periodic boundaries are treated by translating the query AABB
 * by all possible image vectors, many of which are trivially rejected for not intersecting the root node.
 *
 * Because one tree is built per type, complications can arise if particles change type "on the fly" during a
 * a simulation. At present, there is no signal for the types of particles changing (only the total number of types).
 * Any class directly modifying the types of particles \b must signal this change to NeighborListTree using
 * notifyParticleSort().
 *
 * \ingroup computes
 */
class PYBIND11_EXPORT NeighborListTree : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListTree(std::shared_ptr<SystemDefinition> sysdef,
                           Scalar r_cut,
                           Scalar r_buff);

        //! Destructor
        virtual ~NeighborListTree();

    protected:
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

    private:
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        //! Notification of a max number of particle change
        void slotMaxNumChanged()
            {
            m_max_num_changed = true;
            }

        //! Notification of a particle sort
        void slotRemapParticles()
            {
            m_remap_particles = true;
            }

        //! Notification of a number of types change
        void slotNumTypesChanged()
            {
            m_type_changed = true;
            }

        bool m_box_changed;                                 //!< Flag if box size has changed
        bool m_max_num_changed;                             //!< Flag if the particle arrays need to be resized
        bool m_remap_particles;                             //!< Flag if the particles need to remapped (triggered by sort)
        bool m_type_changed;                                //!< Flag if the number of types has changed

        // we use stl vectors here because these tree data structures should *never* be
        // accessed on the GPU, they were optimized for the CPU with SIMD support
        std::vector<hpmc::detail::AABBTree>      m_aabb_trees;     //!< Flat array of AABB trees of all types
        GPUVector<hpmc::detail::AABB>            m_aabbs;          //!< Flat array of AABBs of all types
        std::vector<unsigned int>  m_num_per_type;   //!< Total number of particles per type
        std::vector<unsigned int>  m_type_head;      //!< Index of first particle of each type, after sorting
        std::vector<unsigned int>  m_map_pid_tree;   //!< Maps the particle id to its tag in tree for sorting

        std::vector< vec3<Scalar> > m_image_list;    //!< List of translation vectors
        unsigned int m_n_images;                //!< The number of image vectors to check

        //! Driver for tree configuration
        void setupTree();

        //! Maps particles by local id to their id within their type trees
        void mapParticlesByType();

        //! Computes the image vectors to query for
        void updateImageVectors();

        //! Driver to build AABB trees
        void buildTree();

        //! Traverses AABB trees to compute neighbors
        void traverseTree();
    };

//! Exports NeighborListTree to python
void export_NeighborListTree(pybind11::module& m);

#endif // __NEIGHBORLISTTREE_H__
