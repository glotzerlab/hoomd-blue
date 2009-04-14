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

/*! \file BoxResizeUpdater.h
	\brief Declares an updater that resizes the simulation box of the system
*/

#include <boost/shared_ptr.hpp>

#include "Updater.h"
#include "Variant.h"

#ifndef __BOXRESIZEUPDATER_H__
#define __BOXRESIZEUPDATER_H__

//! Updates the simulation box over time
/*! This simple updater gets the box lengths from specified variants and sets those box sizes 
	over time. As an option, particles can be rescaled with the box lengths or left where they are.
	
	\ingroup updaters
*/
class BoxResizeUpdater : public Updater
	{
	public:
		//! Constructor
		BoxResizeUpdater(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<Variant> Lx, boost::shared_ptr<Variant> Ly, boost::shared_ptr<Variant> Lz);
		
		//! Sets parameter flags
		void setParams(bool scale_particles);
		
		//! Take one timestep forward
		virtual void update(unsigned int timestep);
		
	private:
		boost::shared_ptr<Variant> m_Lx;	//!< Box Lx vs time
		boost::shared_ptr<Variant> m_Ly;	//!< Box Ly vs time
		boost::shared_ptr<Variant> m_Lz;	//!< Box Lz vs time
		bool m_scale_particles; //!< Set to true if particle positions are to be scaled as well


	};
	
//! Export the BoxResizeUpdater to python
void export_BoxResizeUpdater();
	
#endif
