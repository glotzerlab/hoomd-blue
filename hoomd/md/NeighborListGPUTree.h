// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborListGPU.h"
#include "NeighborListGPUTree.cuh"
#include "hoomd/Autotuner.h"

/*! \file NeighborListGPUTree.h
    \brief Declares the NeighborListGPUTree class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUTREE_H__
#define __NEIGHBORLISTGPUTREE_H__

//! Efficient neighbor list build on the GPU using BVH trees
/*!
 * GPU kernel methods are defined in NeighborListGPUTree.cuh and implemented in NeighborListGPUTree.cu.
 *
 * \ingroup computes
 */
class PYBIND11_EXPORT NeighborListGPUTree : public NeighborListGPU
    {
    public:
        //! Constructs the compute
        NeighborListGPUTree(std::shared_ptr<SystemDefinition> sysdef,
                            Scalar r_cut,
                            Scalar r_buff);

        //! Destructor
        virtual ~NeighborListGPUTree();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            NeighborListGPU::setAutotunerParams(enable, period);

            m_tuner_morton->setPeriod(period/10);
            m_tuner_morton->setEnabled(enable);

            m_tuner_merge->setPeriod(period/10);
            m_tuner_merge->setEnabled(enable);

            m_tuner_hierarchy->setPeriod(period/10);
            m_tuner_hierarchy->setEnabled(enable);

            m_tuner_bubble->setPeriod(period/10);
            m_tuner_bubble->setEnabled(enable);

            m_tuner_move->setPeriod(period/10);
            m_tuner_move->setEnabled(enable);

            m_tuner_map->setPeriod(period/10);
            m_tuner_map->setEnabled(enable);

            m_tuner_traverse->setPeriod(period/10);
            m_tuner_traverse->setEnabled(enable);
            }

    protected:
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

    private:
        //! \name Autotuners
        // @{
        std::unique_ptr<Autotuner> m_tuner_morton;    //!< Tuner for kernel to calculate morton codes
        std::unique_ptr<Autotuner> m_tuner_merge;     //!< Tuner for kernel to merge particles into leafs
        std::unique_ptr<Autotuner> m_tuner_hierarchy; //!< Tuner for kernel to generate tree hierarchy
        std::unique_ptr<Autotuner> m_tuner_bubble;    //!< Tuner for kernel to bubble aabbs up hierarchy
        std::unique_ptr<Autotuner> m_tuner_move;      //!< Tuner for kernel to move particles to leaf order
        std::unique_ptr<Autotuner> m_tuner_map;       //!< Tuner for kernel to help map particles by type
        std::unique_ptr<Autotuner> m_tuner_traverse;  //!< Tuner for kernel to traverse generated tree
        // @}

        //! \name Signal updates
        // @{

        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        //! Notification of a change in the maximum number of particles on any rank
        void slotMaxNumChanged()
            {
            m_max_num_changed = true;
            }

        //! Notification of a change in the number of types
        void slotNumTypesChanged()
            {
            // skip the reallocation if the number of types does not change
            // this keeps old parameters when restoring a snapshot
            // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
            if (m_pdata->getNTypes() == m_prev_ntypes)
                return;

            m_type_changed = true;
            }

        unsigned int m_prev_ntypes;                         //!< Previous number of types
        bool m_type_changed;                                //!< Flag if types changed
        bool m_box_changed;                                 //!< Flag if box changed
        bool m_max_num_changed;                             //!< Flag if max number of particles changed
        // @}

        //! \name Tree building
        // @{
        // mapping and sorting
        GPUArray<unsigned int> m_map_tree_pid;      //!< Map a leaf order id to a particle id
        GPUArray<unsigned int> m_map_tree_pid_alt;  //!< Double buffer for map needed for sorting

        GPUArray<uint64_t> m_morton_types;      //!< 30 bit morton codes + type for particles to sort on z-order curve
        GPUArray<uint64_t> m_morton_types_alt;  //!< Double buffer for morton codes needed for sorting
        GPUFlags<int> m_morton_conditions;      //!< Condition flag to catch out of bounds particles

        GPUArray<unsigned int> m_leaf_offset;   //!< Total offset in particle index for leaf nodes by type
        GPUArray<unsigned int> m_num_per_type;  //!< Number of particles per type
        GPUArray<unsigned int> m_type_head;     //!< Head list to each particle type
        GPUArray<unsigned int> m_tree_roots;    //!< Index for root node of each tree by type

        // hierarchy generation
        unsigned int m_n_leaf;                      //!< Total number of leaves in trees
        unsigned int m_n_internal;                  //!< Total number of internal nodes in trees
        unsigned int m_n_node;                      //!< Total number of leaf + internal nodes in trees

        GPUVector<uint32_t> m_morton_codes_red;     //!< Reduced capacity 30 bit morton code array (per leaf)
        GPUVector<Scalar4> m_tree_aabbs;            //!< AABBs for merged leaf nodes and internal nodes
        GPUVector<unsigned int> m_node_locks;       //!< Node locks for if node has been visited or not
        GPUVector<uint2> m_tree_parent_sib;         //!< Parents and siblings of all nodes

        //! Performs initial allocation of tree internal data structure memory
        void allocateTree();

        //! Performs all tasks needed before tree build and traversal
        void setupTree();

        //! Determines the number and head indexes for particle types and leafs
        void countParticlesAndTrees();

        //! Driver for tree multi-step tree build on the GPU
        void buildTree();

        //! Calculates 30-bit morton codes for particles
        void calcMortonCodes();

        //! Driver to sort particles by type and morton code along a Z order curve
        void sortMortonCodes();

        //! Calculates the number of bits needed to represent the largest particle type
        void calcTypeBits();
        unsigned int m_n_type_bits;     //!< the number of bits it takes to represent all the type ids

        //! Merges sorted particles into leafs based on adjacency
        void mergeLeafParticles();

        //! Generates the edges between nodes based on the sorted morton codes
        void genTreeHierarchy();

        //! Constructs enclosing AABBs from leaf to roots
        void bubbleAABBs();

        // @}
        //! \name Tree traversal
        // @{

        GPUArray<Scalar4> m_leaf_xyzf;          //!< Position and id of each particle in a leaf
        GPUArray<Scalar2> m_leaf_db;            //!< Diameter and body of each particle in a leaf

        GPUArray<Scalar3> m_image_list; //!< List of translation vectors
        unsigned int m_n_images;        //!< Number of translation vectors

        //! Computes the image vectors to query for
        void updateImageVectors();

        //! Moves particles from ParticleData order to leaf order for more efficient tree traversal
        void moveLeafParticles();

        //! Traverses the trees on the GPU
        void traverseTree();
        // @}
    };

//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUTree(pybind11::module& m);
#endif //__NEIGHBORLISTGPUTREE_H__
