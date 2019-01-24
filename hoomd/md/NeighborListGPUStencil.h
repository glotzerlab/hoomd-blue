// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "NeighborListGPU.h"
#include "hoomd/CellList.h"
#include "hoomd/CellListStencil.h"
#include "hoomd/Autotuner.h"

/*! \file NeighborListGPUStencil.h
    \brief Declares the NeighborListGPUStencil class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUSTENCIL_H__
#define __NEIGHBORLISTGPUSTENCIL_H__

//! Neighbor list build on the GPU with multiple bin stencils
/*! Implements the O(N) neighbor list build on the GPU using a cell list with multiple bin stencils.

    GPU kernel methods are defined in NeighborListGPUStencil.cuh and defined in NeighborListGPUStencil.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListGPUStencil : public NeighborListGPU
    {
    public:
        //! Constructs the compute
        NeighborListGPUStencil(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar r_cut,
                               Scalar r_buff,
                               std::shared_ptr<CellList> cl = std::shared_ptr<CellList>(),
                               std::shared_ptr<CellListStencil> cls = std::shared_ptr<CellListStencil>());

        //! Destructor
        virtual ~NeighborListGPUStencil();

        //! Change the cutoff radius for all pairs
        virtual void setRCut(Scalar r_cut, Scalar r_buff);

        //! Change the cutoff radius by pair type
        virtual void setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut);

        //! Change the underlying cell width
        void setCellWidth(Scalar cell_width)
            {
            m_override_cell_width = true;
            m_needs_restencil = true;
            m_cl->setNominalWidth(cell_width);
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
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

    private:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size and threads per particle
        unsigned int m_last_tuned_timestep;     //!< Last tuning timestep

        std::shared_ptr<CellList> m_cl;   //!< The cell list
        std::shared_ptr<CellListStencil> m_cls;   //!< The cell list stencil
        bool m_override_cell_width;                 //!< Flag to override the cell width

        //! Update the stencil radius
        void updateRStencil();
        bool m_needs_restencil;                             //!< Flag for updating the stencil
        void slotRCutChange()
            {
            m_needs_restencil = true;
            }

        //! Sort the particles by type
        void sortTypes();
        GPUArray<unsigned int> m_pid_map;                   //!< Particle indexes sorted by type
        bool m_needs_resort;                                //!< Flag to resort the particles
        void slotParticleSort()
            {
            m_needs_resort = true;
            }
        void slotMaxNumChanged()
            {
            m_pid_map.resize(m_pdata->getMaxN());
            m_needs_resort = true;
            }
    };

//! Exports NeighborListGPUStencil to python
void export_NeighborListGPUStencil(pybind11::module& m);

#endif // __NEIGHBORLISTGPUSTENCIL_H__
