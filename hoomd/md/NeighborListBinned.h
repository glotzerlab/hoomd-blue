// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "NeighborList.h"
#include "hoomd/CellList.h"

/*! \file NeighborListBinned.h
    \brief Declares the NeighborListBinned class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

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

        //! Change the cutoff radius for all pairs
        virtual void setRCut(Scalar r_cut, Scalar r_buff);

        //! Set the cutoff radius by pair type
        virtual void setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut);

        //! Set the maximum diameter to use in computing neighbor lists
        virtual void setMaximumDiameter(Scalar d_max);

    protected:
        std::shared_ptr<CellList> m_cl;   //!< The cell list

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
    };

//! Exports NeighborListBinned to python
void export_NeighborListBinned(pybind11::module& m);

#endif
