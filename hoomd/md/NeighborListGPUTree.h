// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborListGPU.h"
#include "NeighborListGPUTree.cuh"

#include "hoomd/Autotuner.h"

/*! \file NeighborListGPUTree.h
    \brief Declares the NeighborListGPUTree class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUTREE_H__
#define __NEIGHBORLISTGPUTREE_H__

namespace hoomd
    {
namespace md
    {
//! Efficient neighbor list build on the GPU using BVH trees
/*!
 * GPU methods mostly make use of the neighbor library to do the traversal.
 * This class acts as a wrapper around those library calls. The general idea is to
 * build one LBVH per particle type. Then, one traversal is done per-type (N^2) traversals
 * to construct the neighbor lists. To support large numbers of types, this traversal is
 * done using one CUDA stream per type to try to improve concurrency.
 *
 * The other jobs of this class are then to preprocess the particle data into a format suitable
 * for building one LBVH per-type. This mainly means sorting the particles by type. In MPI
 * simulations, this sorting can also be used to efficiently filter out ghosts that lie outside the
 * neighbor search range (e.g., those participating in bonds).
 *
 * \ingroup computes
 */
class PYBIND11_EXPORT NeighborListGPUTree : public NeighborListGPU
    {
    public:
    //! Constructs the compute
    NeighborListGPUTree(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff);

    //! Destructor
    virtual ~NeighborListGPUTree();

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        NeighborListGPU::setAutotunerParams(enable, period);

        m_mark_tuner->setPeriod(period / 10);
        m_mark_tuner->setEnabled(enable);

        m_count_tuner->setPeriod(period / 10);
        m_count_tuner->setEnabled(enable);

        m_copy_tuner->setPeriod(period / 10);
        m_copy_tuner->setEnabled(enable);

        /* These may be null pointers if the first compute has not occurred, since construction of
           these tuners is deferred until the first neighbor list build (in order to get the tuner
           parameters from the LBVHWrapper and LBVHTraverserWrapper). When initialized, the period
           and enabled must be borrowed from one of the tuners above to keep everything synced.
         */
        if (m_build_tuner)
            {
            m_build_tuner->setPeriod(period / 10);
            m_build_tuner->setEnabled(enable);
            }

        if (m_traverse_tuner)
            {
            m_traverse_tuner->setPeriod(period / 10);
            m_traverse_tuner->setEnabled(enable);
            }
        }

    protected:
    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);

    private:
    std::unique_ptr<Autotuner> m_mark_tuner;     //!< Tuner for the type mark kernel
    std::unique_ptr<Autotuner> m_count_tuner;    //!< Tuner for the type-count kernel
    std::unique_ptr<Autotuner> m_copy_tuner;     //!< Tuner for the primitive-copy kernel
    std::unique_ptr<Autotuner> m_build_tuner;    //!< Tuner for LBVH builds
    std::unique_ptr<Autotuner> m_traverse_tuner; //!< Tuner for LBVH traversers

    GPUArray<unsigned int> m_types;          //!< Particle types (for sorting)
    GPUArray<unsigned int> m_sorted_types;   //!< Sorted particle types
    GPUArray<unsigned int> m_indexes;        //!< Particle indexes (for sorting)
    GPUArray<unsigned int> m_sorted_indexes; //!< Sorted particle indexes

    unsigned int m_type_bits;            //!< Number of bits to sort based on largest type index
    GPUArray<unsigned int> m_type_first; //!< First index of each particle type in sorted list
    GPUArray<unsigned int> m_type_last;  //!< Last index of each particle type in sorted list

    GPUFlags<unsigned int> m_lbvh_errors; //!< Error flags during particle marking (e.g., off rank)
    std::vector<std::unique_ptr<kernel::LBVHWrapper>> m_lbvhs; //!< Array of LBVHs per-type
    std::vector<std::unique_ptr<kernel::LBVHTraverserWrapper>>
        m_traversers;                   //!< Array of LBVH traverers per-type
    std::vector<hipStream_t> m_streams; //!< Array of CUDA streams per-type

    GlobalVector<Scalar3> m_image_list;      //!< List of translation vectors for traversal
    unsigned int m_n_images;                 //!< Number of translation vectors for traversal
    GPUArray<unsigned int> m_traverse_order; //!< Order to traverse primitives

    //! Build the LBVHs using the neighbor library
    void buildTree();

    //! Traverse the LBVHs using the neighbor library
    void traverseTree();

    //! Computes the image vectors to query for
    void updateImageVectors();

    //! Compute the LBVH domain from the current box
    BoxDim getLBVHBox() const
        {
        const BoxDim box = m_pdata->getBox();

        // ghost layer padding
        Scalar ghost_layer_width(0.0);
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            ghost_layer_width = m_comm->getGhostLayerMaxWidth();
#endif

        Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
        if (!box.getPeriodic().x)
            ghost_width.x = ghost_layer_width;
        if (!box.getPeriodic().y)
            ghost_width.y = ghost_layer_width;
        if (!box.getPeriodic().z && m_sysdef->getNDimensions() == 3)
            ghost_width.z = ghost_layer_width;

        return BoxDim(box.getLo() - ghost_width, box.getHi() + ghost_width, box.getPeriodic());
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

    /// set to true when the type data has been allocated
    bool m_types_allocated;

    bool m_box_changed;       //!< Flag if box changed
    bool m_max_num_changed;   //!< Flag if max number of particles changed
    unsigned int m_max_types; //!< Previous number of types
    // @}
    };

namespace detail
    {
//! Exports NeighborListGPUTree to python
void export_NeighborListGPUTree(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif //__NEIGHBORLISTGPUTREE_H__
