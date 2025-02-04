// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborList.h"
#include "hoomd/CellList.h"
#include "hoomd/CellListStencil.h"

/*! \file NeighborListStencil.h
    \brief Declares the NeighborListStencil class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTSTENCIL_H__
#define __NEIGHBORLISTSTENCIL_H__

namespace hoomd
    {
namespace md
    {
//! Efficient neighbor list build on the CPU with multiple bin stencils
/*! Implements the O(N) neighbor list build on the CPU using a cell list with multiple bin stencils.

    \sa CellListStencil
    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListStencil : public NeighborList
    {
    public:
    //! Constructs the compute
    NeighborListStencil(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff);

    //! Destructor
    virtual ~NeighborListStencil();

    /// Notify NeighborList that a r_cut matrix value has changed
    virtual void notifyRCutMatrixChange()
        {
        m_update_cell_size = true;
        m_needs_restencil = true;
        NeighborList::notifyRCutMatrixChange();
        }

    //! Change the underlying cell width
    void setCellWidth(Scalar cell_width)
        {
        m_override_cell_width = true;
        m_needs_restencil = true;
        m_cl->setNominalWidth(cell_width);
        }

    void setDeterministic(bool deterministic)
        {
        m_cl->setSortCellList(deterministic);
        }

    bool getDeterministic()
        {
        return m_cl->getSortCellList();
        }

    Scalar getCellWidth()
        {
        return m_cl->getNominalWidth();
        }

    protected:
    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);

    private:
    std::shared_ptr<CellList> m_cl;         //!< The cell list
    std::shared_ptr<CellListStencil> m_cls; //!< The cell list stencil
    bool m_override_cell_width = false;     //!< Flag to override the cell width

    bool m_needs_restencil = true; //!< Flag for updating the stencil

    /// Track when the cell size needs to be updated
    bool m_update_cell_size = true;

    //! Update the stencil radius
    void updateRStencil();
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __NEIGHBORLISTSTENCIL_H__
