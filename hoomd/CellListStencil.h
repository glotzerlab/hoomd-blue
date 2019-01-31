// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#include "Compute.h"
#include "CellList.h"

/*! \file CellListStencil.h
    \brief Declares the CellListStencil class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


#ifndef __CELLLISTSTENCIL_H__
#define __CELLLISTSTENCIL_H__

//! Calculates a stencil for a given cell list
/*!
 * Generates a list of translation vectors to check from a CellList for a given search radius.
 *
 * A stencil is a list of offset vectors from a reference cell at (0,0,0) that must be searched for a given particle
 * type based on a set search radius. All bins within that search radius are identified based on
 * the current actual cell width. To use the stencil, the cell list bin for a given particle is identified, and then
 * the offsets are added to that current cell to identify bins to search. Periodic boundaries must be correctly
 * factored in during this step by wrapping search cells back through the boundary. The stencil generation ensures
 * that cells are not duplicated.
 *
 * The minimum distance to each cell in the stencil from the reference is also precomputed and saved during stencil
 * construction. This can be used to accelerate particle search from the cell list without distance check.
 *
 * The stencil is rebuilt any time the search radius or the box dimensions change.
 *
 * \sa NeighborListStencil
 *
 * \ingroup computes
 */
class PYBIND11_EXPORT CellListStencil : public Compute
    {
    public:
        //! Constructor
        CellListStencil(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<CellList> cl);

        //! Destructor
        virtual ~CellListStencil();

        //! Computes the stencil for each type
        virtual void compute(unsigned int timestep);

        //! Set the per-type stencil radius
        void setRStencil(const std::vector<Scalar>& rstencil)
            {
            if (rstencil.size() != m_pdata->getNTypes())
                {
                m_exec_conf->msg->error() << "nlist: number of stencils must be equal to number of particle types" << std::endl;
                throw std::runtime_error("number of stencils must equal number of particle types");
                }
            m_rstencil = rstencil;
            requestCompute();
            }

        //! Get the computed stencils
        const GPUArray<Scalar4>& getStencils() const
            {
            return m_stencil;
            }

        //! Get the size of each stencil
        const GPUArray<unsigned int>& getStencilSizes() const
            {
            return m_n_stencil;
            }

        //! Get the stencil indexer
        const Index2D& getStencilIndexer() const
            {
            return m_stencil_idx;
            }

        //! Slot to recompute the stencil
        void requestCompute()
            {
            m_compute_stencil = true;
            }

    protected:
        virtual bool shouldCompute(unsigned int timestep);

    private:
        std::shared_ptr<CellList> m_cl;               //!< Pointer to cell list operating on
        std::vector<Scalar> m_rstencil;                 //!< Per-type radius to stencil

        Index2D m_stencil_idx;                  //!< Type indexer into stencils
        GPUArray<Scalar4> m_stencil;            //!< Stencil of shifts and closest distance to bin
        GPUArray<unsigned int> m_n_stencil;     //!< Number of bins in a stencil
        bool m_compute_stencil;                 //!< Flag if stencil should be recomputed

        //! Slot for the number of types changing, which triggers a resize
        void slotTypeChange()
            {
            GPUArray<unsigned int> n_stencil(m_pdata->getNTypes(), m_exec_conf);
            m_n_stencil.swap(n_stencil);

            m_rstencil = std::vector<Scalar>(m_pdata->getNTypes(), -1.0);
            requestCompute();
            }
    };

//! Exports CellListStencil to python
#ifndef NVCC
void export_CellListStencil(pybind11::module& m);
#endif


#endif // __CELLLISTSTENCIL_H__
