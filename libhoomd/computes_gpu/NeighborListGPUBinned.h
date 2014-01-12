/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: joaander

#include "NeighborListGPU.h"
#include "CellList.h"
#include "Autotuner.h"

/*! \file NeighborListGPUBinned.h
    \brief Declares the NeighborListGPUBinned class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTGPUBINNED_H__
#define __NEIGHBORLISTGPUBINNED_H__

//! Neighbor list build on the GPU
/*! Implements the O(N) neighbor list build on the GPU using a cell list.

    GPU kernel methods are defined in NeighborListGPUBinned.cuh and defined in NeighborListGPUBinned.cu.

    \ingroup computes
*/
class NeighborListGPUBinned : public NeighborListGPU
    {
    public:
        //! Constructs the compute
        NeighborListGPUBinned(boost::shared_ptr<SystemDefinition> sysdef,
                              Scalar r_cut,
                              Scalar r_buff,
                              boost::shared_ptr<CellList> cl = boost::shared_ptr<CellList>());

        //! Destructor
        virtual ~NeighborListGPUBinned();

        //! Change the cuttoff radius
        virtual void setRCut(Scalar r_cut, Scalar r_buff);

        //! Set the block size
        void setBlockSize(unsigned int block_size)
            {
            m_block_size = block_size;
            }

        //! Set the autotuner period
        void setTuningParam(unsigned int param)
            {
            m_param = param;
            }

        //! Set the maximum diameter to use in computing neighbor lists
        virtual void setMaximumDiameter(Scalar d_max);

        //! Enable/disable body filtering
        virtual void setFilterBody(bool filter_body);

        //! Enable/disable diameter filtering
        virtual void setFilterDiameter(bool filter_diameter);

    protected:
        boost::shared_ptr<CellList> m_cl;   //!< The cell list
        cudaArray *dca_cell_adj;            //!< CUDA array for tex2D access to d_cell_adj
        cudaArray *dca_cell_xyzf;           //!< CUDA array for tex2D access to d_cell_xyzf
        cudaArray *dca_cell_tdb;            //!< CUDA array for tex2D access to d_cell_tdb
        uint3 m_last_dim;                   //!< The last dimensions allocated for the cell list tex2D arrays
        unsigned int m_last_cell_Nmax;      //!< The last Nmax allocated for the cell list tex2D arrays
        unsigned int m_block_size;          //!< Block size to execute on the GPU
        unsigned int m_param;               //!< Kernel tuning parameter

        boost::scoped_ptr<Autotuner> m_tuner; //!< Autotuner for block size and threads per particle
        unsigned int m_last_tuned_timestep; //!< Last tuning timestep

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

        //! Test if the cuda arrays need reallocation
        bool needReallocateCudaArrays();

        //! Updates the cudaArray allocations
        void allocateCudaArrays();

    };

//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUBinned();

#endif
