// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "NeighborList.h"
#include "hoomd/CellList.h"

/*! \file NeighborListBinned.h
    \brief Declares the NeighborListBinned class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTBINNED_H__
#define __NEIGHBORLISTBINNED_H__

//! Efficient neighbor list build on the CPU
/*! Implements the O(N) neighbor list build on the CPU using a cell list.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListBinned : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListBinned(std::shared_ptr<SystemDefinition> sysdef,
                           Scalar r_cut,
                           Scalar r_buff,
                           std::shared_ptr<CellList> cl = std::shared_ptr<CellList>());

        //! Destructor
        virtual ~NeighborListBinned();

        /// Notify NeighborList that a r_cut matrix value has changed
        virtual void notifyRCutMatrixChange()
            {
            m_update_cell_size = true;
            NeighborList::notifyRCutMatrixChange();
            }

        /// Make the neighborlist deterministic
        void setDeterministic(bool deterministic)
            {
            m_cl->setSortCellList(deterministic);
            }

        /// Get the deterministic flag
        bool getDeterministic()
            {
            return m_cl->getSortCellList();
            }

    protected:
        std::shared_ptr<CellList> m_cl;   //!< The cell list

        /// Track when the cell size needs to be updated
        bool m_update_cell_size;

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
    };

//! Exports NeighborListBinned to python
void export_NeighborListBinned(pybind11::module& m);

#endif
