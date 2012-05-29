/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*! \file RemoveParticlesUpdater.h
    \brief Declares an updater that renders certain particles inert and updates the degrees of freedom
    of the system.
*/

#include <boost/shared_ptr.hpp>

#include "Updater.h"
#include "ParticleData.h"
#include "ComputeThermo.h"
#include "Index1D.h"
#include "NeighborList.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <sstream>

#ifndef __REMOVEPARTICLESUPDATER_H__
#define __REMOVEPARTICLESUPDATER_H__

//! Removes particles from the system.
/*! This updater checks whether particles have crossed the z-boundary and changes the type of any
    particles that have.  The new type will exert no forces on other particles.  The updater also
    sets the velocity of each removed particle to zero so that it does not contribute to the system
    temperature.  Finally, the updater updates the number of degrees of freedom of the system so the
    temperature is computed accurately.

    \ingroup updaters
*/
class RemoveParticlesUpdater : public Updater
{
    public:
        //! Constructor
        RemoveParticlesUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                               boost::shared_ptr<ParticleGroup> groupA,
							   boost::shared_ptr<ParticleGroup> groupB,
                               boost::shared_ptr<ComputeThermo> thermo,
                               boost::shared_ptr<NeighborList> nlist,
							   Scalar r_cut,
                               std::string filename);

        //! Destructor
        virtual ~RemoveParticlesUpdater() 
		{
			m_exec_conf->msg->notice(5) << "Destroying RemoveParticlesUpdater" << endl;
		};

        //! Perform the updater
        virtual void update(unsigned int timestep);

    private:
        //! Helper function for determining the particle type ID
        unsigned int getTypeId(const std::string& name);

    protected:
        boost::shared_ptr<ParticleGroup> m_groupA; //!< primary particle group to which the updater is applied
		boost::shared_ptr<ParticleGroup> m_groupB; //!< secondary particle group to which the updater is applied
						    //!< this group uses special handling 
        const boost::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities
        boost::shared_ptr<NeighborList> m_nlist; //!< The neighbor list to use for capturing groups of particles
		const Scalar m_rcut; //!< Cutoff radius for determining neighbors to remove
        std::string m_filename; //!< file name to which particle data will be written
        unsigned int num_particles; //!< number of particles used to compute the number of degrees of freedom
};

//! Export the class to python
void export_RemoveParticlesUpdater();

#endif
