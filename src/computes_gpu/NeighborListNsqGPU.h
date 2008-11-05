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

/*! \file NeighborListNsqGPU.h
	\brief Declares the NeighborListNsqGPU class
*/

#include <vector>
#include "NeighborList.h"
#include "gpu_nlist.h"

#include <boost/shared_ptr.hpp>

#ifndef __NEIGHBORLIST_NSQ_GPU_H__
#define __NEIGHBORLIST_NSQ_GPU_H__

//! Computes a Neibhorlist from the particles
/*!	Calculates the same neighbor list that NeighborList does, but on the GPU.
	
	This class implements the same O(N^2) algorithm as the base class.
	
	The GPU kernel that does the calculations can be found in nlist_nsq_kernel.cu.
	\ingroup computes
*/
class NeighborListNsqGPU : public NeighborList
	{
	public:
		//! Constructs the compute
		NeighborListNsqGPU(boost::shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff);

		//! Destructor
		virtual ~NeighborListNsqGPU();
		
	private:
		//! Builds the neighbor list
		virtual void buildNlist();
		
		//! Attempts to builds the neighbor list
		virtual void buildNlistAttempt();
	};
	
//! Exports the NeighborListNsqGPU class to python
void export_NeighborListNsqGPU();

#endif

