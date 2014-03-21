/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

// Maintainer: joaander

/*! \file Integrator.h
    \brief Declares the Integrator base class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "Updater.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "ParticleGroup.h"
#include <string>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

//! Base class that defines an integrator
/*! An Integrator steps the entire simulation forward one time step in time.
    Prior to calling update(timestep), the system is at time step \a timestep.
    After the call to update completes, the system is at \a timestep + 1.

    All integrators have the common property that they add up many forces to
    get the net force on each particle. This task is performed by the
    base class Integrator. Similarly, all integrators share the
    property that they have a time step, \a deltaT.

    Any number of ForceComputes can be used to specify the net force
    for use with this integrator. They are added via calling
    addForceCompute(). Any number of forces can be added in this way.

    All forces added via addForceCompute() are computed independantly and then totaled up to calculate the net force
    and enrgy on each particle. Constraint forces (ForceConstraint) are unique in that they need to be computed
    \b after the net forces is already available. To implement this behavior, call addForceConstraint() to add any
    number of constraint forces. All constraint forces will be computed independantly and will be able to read the
    current unconstrained net force. Separate constraint forces should not overlap. Degrees of freedom removed
    via the constraint forces can be totaled up with a call to getNDOFRemoved for convenience in derived classes
    implementing correct counting in getNDOF().

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
        virtual ~Integrator();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Add a ForceCompute to the list
        virtual void addForceCompute(boost::shared_ptr<ForceCompute> fc);

        //! Add a ForceConstraint to the list
        virtual void addForceConstraint(boost::shared_ptr<ForceConstraint> fc);

        //! Removes all ForceComputes from the list
        virtual void removeForceComputes();

        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);

        //! Return the timestep
        Scalar getDeltaT();

        //! Get the number of degrees of freedom granted to a given group
        /*! \param group Group over which to count degrees of freedom.
            Base class Integrator returns 0. Derived classes should override.
        */
        virtual unsigned int getNDOF(boost::shared_ptr<ParticleGroup> group)
            {
            return 0;
            }

        //! Count the total number of degrees of freedom removed by all constraint forces
        unsigned int getNDOFRemoved();

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! helper function to compute total momentum
        virtual Scalar computeTotalMomentum(unsigned int timestep);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm The Communicator
         */
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm);

        //! Callback for pre-computing the forces
        void computeCallback(unsigned int timestep);
        #endif

    protected:
        Scalar m_deltaT;                                            //!< The time step
        std::vector< boost::shared_ptr<ForceCompute> > m_forces;    //!< List of all the force computes

        std::vector< boost::shared_ptr<ForceConstraint> > m_constraint_forces;    //!< List of all the constraints

        //! helper function to compute initial accelerations
        void computeAccelerations(unsigned int timestep);

        //! helper function to compute net force/virial
        void computeNetForce(unsigned int timestep);

#ifdef ENABLE_CUDA
        //! helper function to compute net force/virial on the GPU
        void computeNetForceGPU(unsigned int timestep);
#endif

#ifdef ENABLE_MPI
        //! helper function to determine the ghost communciation flags
        CommFlags determineFlags(unsigned int timestep);
#endif

    private:
        #ifdef ENABLE_MPI
        boost::signals2::connection m_request_flags_connection;     //!< Connection to Communicator to request communication flags
        boost::signals2::connection m_callback_connection;          //!< Connection to Commmunicator for compute callback
        #endif
    };

//! Exports the NVEUpdater class to python
void export_Integrator();

#endif
