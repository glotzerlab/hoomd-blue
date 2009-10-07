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

/*! \file Integrator.h
    \brief Declares the Integrator base class
*/

#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "Updater.h"
#include "ForceCompute.h"
#include <string>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif


//! Base class that defines an integrator
/*! An Integrator steps the entire simulation forward one time step in time.
    Prior to calling update(timestep), the system is at time step \a timestep.
    After the call to update completes, the system is at \a timestep + 1.

    Many integrators have the common property that they add up many forces to
    get the net force on each particle. This task is performed by the
    base class Integrator. Similarly, all integrators share the
    property that they have a time step, \a deltaT.

    Derived integrators can of course add additional parameters and
    properties.

    Any number of ForceComputes can be used to specify the net force
    for use with this integrator. They are added via calling
    addForceCompute(). Although there is a current, a maximum of 32 ForceComputes
    supported on the GPU. If there is ever a need for this to be increased, it can
    be done without too much trouble, but 32 should be much more than sufficient
    for any simulation.

    Integrators take "ownership" of the particle's accellerations. Any other updater
    that modifies the particles accelerations will produce undefined results. If
    accelerations are to be modified, they must be done through forces, and added to
    an Integrator via addForceCompute().

    No such ownership is taken of the particle positions and velocities. Other Updaters
    can modify particle positions and velocities as they wish. Those updates will be taken
    into account by the Integrator. It would probably make the most sense to have such updaters
    run BEFORE the Integrator, since an Integrator actually moves the particles to the next time
    step. This is handled correctly by System.

    \ingroup updaters
*/
class Integrator : public Updater
    {
    public:
        //! Constructor
        Integrator(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);
        
        //! Destructor
        ~Integrator();
        
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
        
        //! Add a ForceCompute to the list
        virtual void addForceCompute(boost::shared_ptr<ForceCompute> fc);
        
        //! Removes all ForceComputes from the list
        virtual void removeForceComputes();
        
        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);
        
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
        //! helper function to compute temperature
        virtual Scalar computeTemperature(unsigned int timestep);
        
        //! helper function to compute pressure
        virtual Scalar computePressure(unsigned int timestep);
        
        //! helper function to compute kinetic energy
        virtual Scalar computeKineticEnergy(unsigned int timestep);
        
        //! helper function to compute potential energy
        virtual Scalar computePotentialEnergy(unsigned int timestep);
        
        //! helper function to compute total momentum
        virtual Scalar computeTotalMomentum(unsigned int timestep);
        
    protected:
        Scalar m_deltaT;    //!< The time step
        std::vector< boost::shared_ptr<ForceCompute> > m_forces;    //!< List of all the force computes
        
        //! helper function to compute accelerations
        void computeAccelerations(unsigned int timestep, const std::string& profile_name);
        
#ifdef ENABLE_CUDA
        //! helper function to compute accelerations on the GPU
        void computeAccelerationsGPU(unsigned int timestep, const std::string& profile_name, bool sum_accel);
        
        //! Force data pointers on the device
        vector<float4 **> m_d_force_data_ptrs;
        
#endif
    };

//! Exports the NVEUpdater class to python
void export_Integrator();

#endif

