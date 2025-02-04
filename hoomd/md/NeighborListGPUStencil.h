// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborListGPU.h"
#include "hoomd/Autotuner.h"
#include "hoomd/CellListGPU.h"
#include "hoomd/CellListStencil.h"

/*! \file NeighborListGPUStencil.h
    \brief Declares the NeighborListGPUStencil class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUSTENCIL_H__
#define __NEIGHBORLISTGPUSTENCIL_H__

namespace hoomd
    {
namespace md
    {
//! Neighbor list build on the GPU with multiple bin stencils
/*! Implements the O(N) neighbor list build on the GPU using a cell list with multiple bin stencils.

    GPU kernel methods are defined in NeighborListGPUStencil.cuh and defined in
   NeighborListGPUStencil.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListGPUStencil : public NeighborListGPU
    {
    public:
    //! Constructs the compute
    NeighborListGPUStencil(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff);

    //! Destructor
    virtual ~NeighborListGPUStencil();

    /// Notify NeighborList that a r_cut matrix value has changed
    virtual void notifyRCutMatrixChange()
        {
        m_update_cell_size = true;
        m_needs_restencil = true;
        NeighborListGPU::notifyRCutMatrixChange();
        }

    //! Change the underlying cell width
    void setCellWidth(Scalar cell_width)
        {
        m_override_cell_width = true;
        m_needs_restencil = true;
        m_cl->setNominalWidth(cell_width);
        }

    protected:
    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);

    private:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle
    uint64_t m_last_tuned_timestep;        //!< Last tuning timestep

    std::shared_ptr<CellList> m_cl;         //!< The cell list
    std::shared_ptr<CellListStencil> m_cls; //!< The cell list stencil
    bool m_override_cell_width = false;     //!< Flag to override the cell width

    //! Update the stencil radius
    void updateRStencil();
    bool m_needs_restencil = true; //!< Flag for updating the stencil

    //! Sort the particles by type
    void sortTypes();
    GPUArray<unsigned int> m_pid_map; //!< Particle indexes sorted by type
    bool m_needs_resort;              //!< Flag to resort the particles
    void slotParticleSort()
        {
        m_needs_resort = true;
        }
    void slotMaxNumChanged()
        {
        m_pid_map.resize(m_pdata->getMaxN());
        m_needs_resort = true;
        }

    /// Track when the cell size needs to be updated
    bool m_update_cell_size = false;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __NEIGHBORLISTGPUSTENCIL_H__
