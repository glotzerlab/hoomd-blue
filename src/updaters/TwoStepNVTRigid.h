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
// Maintainer: ndtrung

/*! \file TwoStepNVTRigid.h
    \brief Declares an updater that implements NVT dynamics for rigid bodies
*/

#include "Variant.h"
#include "TwoStepNVERigid.h"
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __TWO_STEP_NVT_RIGID_H__
#define __TWO_STEP_NVT_RIGID_H__

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant temperature (NVT) dynamics. Particle positions and velocities are
    updated according to the velocity verlet algorithm. The forces that drive this motion are defined external to this class
    in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be summed to produce a net force on
    each particle.

    \ingroup updaters
*/

class TwoStepNVTRigid : public TwoStepNVERigid
    {
    public:
        //! Constructor
        TwoStepNVTRigid(boost::shared_ptr<SystemDefinition> sysdef, 
                        boost::shared_ptr<ParticleGroup> group,
                        boost::shared_ptr<Variant> temperature);
        
        //! Setup the initial net forces, torques and angular momenta
        void setup();
        
        //! First step of velocit Verlet integration
        void integrateStepOne(unsigned int timestep);
        
        //! Second step of velocit Verlet integration
        void integrateStepTwo(unsigned int timestep);
            
    protected:
        //! Update thermostats
        void update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep);
        
        //! Update thermostat momenta and positions
        void no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt);
        
        //! Quaternion multiply
        void quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
        
        //! Inverse quaternion multiply
        void inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
        
        //! Matrix multiply
        void matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
        
        //! Transposed matrix multiply
        void transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
        
        //! Maclaurin expansion
        inline Scalar maclaurin_series(Scalar x);
        
        boost::shared_ptr<Variant> m_temperature;   //!< External temperature
        Scalar boltz;                               //!< Boltzmann constant
        Scalar t_freq;                              //!< Coupling frequency
        Scalar nf_t;                                //!< Translational degrees of freedom        
        Scalar nf_r;                                //!< Rotational degrees of freedom 
        unsigned int chain;                         //!< Number of thermostat chains
        unsigned int iter;                          //!< Number of iterations
        unsigned int order;                         //!< Number of thermostat per chain
        
        //!< Thermostat fictious positions and velocities
        GPUArray<Scalar>    q_t;
        GPUArray<Scalar>    q_r;
        GPUArray<Scalar>    eta_t;
        GPUArray<Scalar>    eta_r;
        GPUArray<Scalar>    eta_dot_t;
        GPUArray<Scalar>    eta_dot_r;
        GPUArray<Scalar>    f_eta_t;
        GPUArray<Scalar>    f_eta_r;
        
        GPUArray<Scalar>    w;
        GPUArray<Scalar>    wdti1;
        GPUArray<Scalar>    wdti2;
        GPUArray<Scalar>    wdti4;
        GPUArray<Scalar4>   conjqm;

    };

//! Exports the TwoStepNVTRigid class to python
void export_TwoStepNVTRigid();

#endif

