/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file BinnedNeighborListGPU.h
	\brief Declares the BinnedNeighborListGPU class
*/

#include "NeighborList.h"
#include "gpu_nlist.h"

#include <boost/shared_ptr.hpp>

#ifndef __BINNED_NEIGHBORLIST_GPU_H__
#define __BINNED_NEIGHBORLIST_GPU_H__

//! Binned neighborlist on the GPU
/*! \todo document me
	\ingroup computes
*/
class BinnedNeighborListGPU : public NeighborList
	{
	public:
		//! Constructor
		BinnedNeighborListGPU(boost::shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff);

		virtual ~BinnedNeighborListGPU();

		//! Computes the NeighborList if it needs updating
		virtual void compute(unsigned int timestep);
		
		//! Prints statistics on the neighbor list
		virtual void printStats();
		
		//! Sets the block size of the calculation on the GPU
		void setBlockSize(int block_size) { m_block_size = block_size; }

	protected:
		std::vector< unsigned int > m_bin_sizes;	//!< Stores the size of each bin

		unsigned int m_Mx;		//!< Number of bins in x direction
		unsigned int m_last_Mx;	//!< Number of bins in the x direction on the last call to updateBins
		unsigned int m_My;		//!< Number of bins in y direction
		unsigned int m_last_My;	//!< Number of bins in the y direction on the last call to updateBins
		unsigned int m_Mz;		//!< Number of bins in z direction
		unsigned int m_last_Mz;	//!< Number of bins in the z direction on the last call to updateBins

		unsigned int m_Nmax; 	//!< Maximum number of particles allowed in a bin
		unsigned int m_curNmax; //!< Number of particles in the largest bin
		Scalar m_avgNmax; 		//!< Average number of particles per bin

		std::vector<gpu_bin_array> m_gpu_bin_data;	//!< The binned particle data
		unsigned int *m_host_idxlist;	//!< Host bins
		int m_block_size;				//!< Block size to use when performing the calculations on the GPU
		unsigned int *m_mem_location;	//!< Memory location of bins (Z-order curve)

		//! Puts the particles into their bins
		void updateBinsUnsorted();
	
		//! Updates the neighborlist using the binned data
		void updateListFromBins();

		//! Test if the list needs updating
		virtual bool needsUpdating(unsigned int timestep);
		
		//! Helper function to allocate bin data
		void allocateGPUBinData(unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax);
		
		//! Helper function to free bin data
		void freeGPUBinData();
	};
	
#ifdef USE_PYTHON
//! Exports the BinnedNeighborListGPU class to python
void export_BinnedNeighborListGPU();
#endif

#endif
