// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "NeighborListGPU.h"
#include "hoomd/CellList.h"
#include "hoomd/Autotuner.h"

/*! \file NeighborListGPUBinned.h
    \brief Declares the NeighborListGPUBinned class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUBINNED_H__
#define __NEIGHBORLISTGPUBINNED_H__

//! Neighbor list build on the GPU
/*! Implements the O(N) neighbor list build on the GPU using a cell list.

    GPU kernel methods are defined in NeighborListGPUBinned.cuh and defined in NeighborListGPUBinned.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListGPUBinned : public NeighborListGPU
    {
    public:
        //! Constructs the compute
        NeighborListGPUBinned(std::shared_ptr<SystemDefinition> sysdef,
                              Scalar r_cut,
                              Scalar r_buff,
                              std::shared_ptr<CellList> cl = std::shared_ptr<CellList>());

        //! Destructor
        virtual ~NeighborListGPUBinned();

        //! Change the cutoff radius for all pairs
        virtual void setRCut(Scalar r_cut, Scalar r_buff);

        //! Change the cutoff radius by pair type
        virtual void setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut);

        //! Set the autotuner period
        void setTuningParam(unsigned int param)
            {
            m_param = param;
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            NeighborListGPU::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period/10);
            m_tuner->setEnabled(enable);
            }

        //! Set the maximum diameter to use in computing neighbor lists
        virtual void setMaximumDiameter(Scalar d_max);

    protected:
        std::shared_ptr<CellList> m_cl;   //!< The cell list
        unsigned int m_block_size;          //!< Block size to execute on the GPU
        unsigned int m_param;               //!< Kernel tuning parameter
        bool m_use_index;                 //!< True for indirect lookup of particle data via index

        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size and threads per particle

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
    };

//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUBinned(pybind11::module& m);

#endif
