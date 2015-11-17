/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: mphoward

#include "Compute.h"
#include "CellList.h"

/*! \file CellListStencil.h
    \brief Declares the CellListStencil class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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
 * \sa NeighborListMultiBinned
 *
 * \ingroup computes
 */
class CellListStencil : public Compute
    {
    public:
        //! Constructor
        CellListStencil(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<CellList> cl);

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
        boost::shared_ptr<CellList> m_cl;               //!< Pointer to cell list operating on
        std::vector<Scalar> m_rstencil;                 //!< Per-type radius to stencil

        boost::signals2::connection m_num_type_change_conn; //!< Connection to the ParticleData number of types
        boost::signals2::connection m_box_change_conn;      //!< Connection to the box size

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
void export_CellListStencil();

#endif // __CELLLISTSTENCIL_H__
