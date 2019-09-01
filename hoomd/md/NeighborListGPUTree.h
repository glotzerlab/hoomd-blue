// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborListGPU.h"
#include "NeighborListGPUTree.cuh"

#include "hoomd/Autotuner.h"
#include "hoomd/neighbor/LBVH.h"

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
            }

    protected:
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

    private:
        GPUArray<unsigned int> m_types;
        GPUArray<unsigned int> m_sorted_types;
        GPUArray<unsigned int> m_indexes;
        GPUArray<unsigned int> m_sorted_indexes;

        GPUArray<unsigned int> m_type_first;
        GPUArray<unsigned int> m_type_last;

        GPUFlags<unsigned int> m_lbvh_errors;
        std::vector< std::unique_ptr<neighbor::LBVH> > m_lbvhs;

        GPUArray<Scalar3> m_image_list; //!< List of translation vectors
        unsigned int m_n_images;        //!< Number of translation vectors

        void buildTree();

        void traverseTree();

        //! Computes the image vectors to query for
        void updateImageVectors();

        //! Compute the LBVH domain from the current box
        BoxDim getLBVHBox() const
            {
            const BoxDim& box = m_pdata->getBox();

            // ghost layer padding
            Scalar ghost_layer_width(0.0);
            #ifdef ENABLE_MPI
            if (m_comm) ghost_layer_width = m_comm->getGhostLayerMaxWidth();
            #endif

            Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
            if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
            if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
            if (!box.getPeriodic().z && m_sysdef->getNDimensions() == 3) ghost_width.z = ghost_layer_width;

            return BoxDim(box.getLo()-ghost_width, box.getHi()+ghost_width, box.getPeriodic());
            }

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
            m_type_changed = true;
            }

        bool m_type_changed;        //!< Flag if types changed
        bool m_box_changed;         //!< Flag if box changed
        bool m_max_num_changed;     //!< Flag if max number of particles changed
        unsigned int m_max_types;   //!< Previous number of types
        // @}
    };

//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUTree(pybind11::module& m);
#endif //__NEIGHBORLISTGPUTREE_H__
