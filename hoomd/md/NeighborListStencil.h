// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborList.h"
#include "hoomd/CellList.h"
#include "hoomd/CellListStencil.h"

/*! \file NeighborListStencil.h
    \brief Declares the NeighborListStencil class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTSTENCIL_H__
#define __NEIGHBORLISTSTENCIL_H__

//! Efficient neighbor list build on the CPU with multiple bin stencils
/*! Implements the O(N) neighbor list build on the CPU using a cell list with multiple bin stencils.

    \sa CellListStencil
    \ingroup computes
*/
class NeighborListStencil : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListStencil(boost::shared_ptr<SystemDefinition> sysdef,
                            Scalar r_cut,
                            Scalar r_buff,
                            boost::shared_ptr<CellList> cl = boost::shared_ptr<CellList>(),
                            boost::shared_ptr<CellListStencil> cls = boost::shared_ptr<CellListStencil>());

        //! Destructor
        virtual ~NeighborListStencil();

        //! Change the cutoff radius for all pairs
        virtual void setRCut(Scalar r_cut, Scalar r_buff);

        //! Set the cutoff radius by pair type
        virtual void setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut);

        //! Change the underlying cell width
        void setCellWidth(Scalar cell_width)
            {
            m_override_cell_width = true;
            m_needs_restencil = true;
            m_cl->setNominalWidth(cell_width);
            }

        //! Set the maximum diameter to use in computing neighbor lists
        virtual void setMaximumDiameter(Scalar d_max);

    protected:
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

    private:
        boost::shared_ptr<CellList> m_cl;           //!< The cell list
        boost::shared_ptr<CellListStencil> m_cls;   //!< The cell list stencil
        bool m_override_cell_width;                 //!< Flag to override the cell width

        boost::signals2::connection m_rcut_change_conn;     //!< Connection to the cutoff radius changing
        bool m_needs_restencil;                             //!< Flag for updating the stencil
        void slotRCutChange()
            {
            m_needs_restencil = true;
            }

        //! Update the stencil radius
        void updateRStencil();
    };

//! Exports NeighborListStencil to python
void export_NeighborListStencil();

#endif // __NEIGHBORLISTSTENCIL_H__
