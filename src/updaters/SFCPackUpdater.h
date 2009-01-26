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

/*! \file SFCPackUpdater.h
	\brief Declares the SFCPackUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include <vector>

#include "Updater.h"
#include "NeighborList.h"

#ifndef __SFCPACK_UPDATER_H__
#define __SFCPACK_UPDATER_H__

//! Sort the particles
/*! Impelements an algorithm that reorders particles in the ParticleData so that particles
	near each other in space become near each other in memory. This transformation improves
	cache locality in almost every other calculation in HOOMD, such as LJForceCompute,
	HarmonicBondForceCompute, and BinnedNeighborList, to name a few. As particles move
	through time, they will tend to unsort themselves at a rate depending on how diffusive
	the simulation is. Tests preformed on a Leannard-Jones liquid simulation at a temperature of 1.2 
	showed that performing the sort every 1,000 time steps is sufficient to maintain the
	benifits of the sort without significant overhead. Less diffusive systems can easily increase
	that value to 2,000 or more.

	Usage:<br>
	Constructe the SFCPackUpdater, attaching it to the ParticleData and setting reasonable
	parameters. \a bin_width should  be chosen small in principle, but there is little gain 
	to going smaller than the diameter of the typical particle in the simulation.

	Implementation details:<br>
	The rearranging is done by placing the particles into bins and then traversing the bins
	along a hilbert curve. It is very efficient, as long as the box size does not change
	quickly necessitating a rebuild of the hilbert curve.

 	\ingroup updaters
*/
class SFCPackUpdater : public Updater
	{
	public:
		//! Constructor
		SFCPackUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar bin_width);
		
		//! Take one timestep forward
		virtual void update(unsigned int timestep);
		
		//! Set the bin width
		/*! \param bin_width New bin width to set */
		void setBinWidth(Scalar bin_width) { m_bin_width = bin_width; }
		
	private:
		Scalar m_bin_width;			//!< The side length of the bins used to sort particles
		unsigned int m_lastMmax;	//!< The last value of MMax

		std::vector< std::vector<unsigned int> > m_bins;	//!< Binned particles
		std::vector< unsigned int > m_traversal_order;		//!< Generated traversal order of bins
		std::vector<unsigned int> m_sort_order;				//!< Generated sort order of the particles

		//! Helper function that actually performs the sort
		void getSortedOrder();

		//! Apply the sorted order to the particle data
		void applySortOrder();
		
	};
	
//! Export the SFCPackUpdater class to python
void export_SFCPackUpdater();
		
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

