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
// Maintainer: joaander

/*! \file NVEUpdater.h
    \brief Declares an updater that implements NVE dynamics
*/

#include "Updater.h"
#include "Integrator.h"
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __NVEUPDATER_H__
#define __NVEUPDATER_H__

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant energy (NVE) dynamics. Particle positions and 
	velocities are updated according to the velocity verlet algorithm. The forces that drive this motion are defined 
	external to this class in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be 
	summed to produce a net force on each particle.

    \ingroup updaters
*/
class NVEUpdater : public Integrator
    {
    public:
        //! Constructor
        NVEUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);
        
        //! Sets the movement limit
        void setLimit(Scalar limit);
        
        //! Removes the limit
        void removeLimit();
        
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
    protected:
        bool m_accel_set;   //!< Flag to tell if we have set the accelleration yet
        bool m_limit;       //!< True if we should limit the distance a particle moves in one step
        Scalar m_limit_val; //!< The maximum distance a particle is to move in one step
        
        boost::shared_ptr<class NVERigidUpdater> m_rigid_updater;   //! The updater for rigid bodies, if any
    };

//! Exports the NVEUpdater class to python
void export_NVEUpdater();

#endif

