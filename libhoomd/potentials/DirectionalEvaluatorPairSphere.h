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

// $Id$
// $URL$
// Maintainer: grva

#ifndef __DirectionalEvaluatorPairSphere__
#define __DirectionalEvaluatorPairSphere__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file DirectionalEvaluatorPairSphere.h
    \brief Defines the pair evaluator class for spheres. It is templated on
           a struct that allows it to do different types of directional
	   modulation for pair potentials. E.g. by templating in the right
	   fashion one can make Janus spheres, triblock Janus spheres, etc.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Class for evaluating the directional part of anisotropic pair interations
/*! <b>General Overview</b>

    Provides class for generalized diblock Janus spheres. It uses preexisting isotropic pair evaluators to do much of
    the work.
*/
template <EvaluatorPairSphereStruct>
class DirectionalEvaluatorPairSphere
    {
    public:
	typedef typename DirectionalEvaluatorPair.params param_type;

	//! Constructor
	DirectionalEvaluatorPairSphere(Scalar3 _dr, Scalar4 _quat_i,
			Scalar4 _quat_j, Scalar _rcutsq, param_type _params)
	    {
	    // call constructor for underlying struct
	    s = DirectionalEvaluatorPairSphere(_dr,_quat_i,_quat_j,
			    _rcutsq,_params);
	    }

        //! uses diameter
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! whether pair potential requires charges
        //! This function is pure virtual
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }
        
        //! Evaluate the force and energy
        //! This function is pure virtual
        /*! \param force Output parameter to write the computed force.
            \param isoModulator Output parameter to write the amount of modulation of the isotropic part
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPairJanusSphere.
            
            \return Always true
        */
        DEVICE bool evalPair(Scalar3& force, Scalar& isoModulator, Scalar3& torque_i, Scalar3& torque_j)
            {
            // common calculations
            Scalar modi = s.Modulatori();
            Scalar modj = s.Modulatorj();
	    Scalar modPi = s.ModulatorPrimei();
	    Scalar modPj = s.ModulatorPrimej();

	    // the overall modulation
	    isoModulator = modi*modj;

	    // intermediate calculations
	    Scalar iPj = modPi*modj;
	    Scalar jPi = modPi*modj;
            
	    // torque on ith
	    torque_i.x = iPj*(s.dr.y*s.ei.z-s.dr.z*s.ei.y);
	    torque_i.y = iPj*(s.dr.z*s.ei.x-s.dr.x*s.ei.z);
	    torque_i.z = iPj*(s.dr.x*s.ei.y-s.dr.y*s.ei.x);

	    // torque on jth - note sign is opposite ith!
	    torque_j.x = jPi*(s.dr.z*s.ej.y-s.dr.y*s.ej.z);
	    torque_j.y = jPi*(s.dr.x*s.ej.z-s.dr.z*s.ej.x);
	    torque_j.z = jPi*(s.dr.y*s.ej.x-s.dr.x*s.ej.y);

	    // compute force contribution
	    force.x = (iPj*(s.ei.x/s.magdr-s.doti*s.dr.x/s.drsq)
			    -jPi*(s.ej.x/s.magdr-s.dotj*s.dr.x/s.drsq));
	    force.y = (iPj*(s.ei.y/s.magdr-s.doti*s.dr.y/s.drsq)
			    -jPi*(s.ej.y/s.magdr-s.dotj*s.dr.y/s.drsq));
	    force.z = (iPj*(s.ei.z/s.magdr-s.doti*s.dr.z/s.drsq)
			    -jPi*(s.ej.z/s.magdr-s.dotj*s.dr.z/s.drsq));
	    

            return true;
            }
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName() { return std::string("deps") }
        #endif

    private:
	EvaluatorPairSphereStruct s;

    };

//! Class for evaluating the directional part of anisotropic pair interations
/*! <b>General Overview</b>

    Provides class for generalized diblock Janus spheres. It uses preexisting isotropic pair evaluators to do much of
    the work. This does the "non-decorated" part of the sphere.
*/
template <typename EvaluatorPairSphereStruct>
class DirectionalEvaluatorPairJanusSphereComplement
    {
    public:
	typedef typename DirectionalEvaluatorPair.params param_type;

	//! Constructor
	DirectionalEvaluatorPairSphere(Scalar3 _dr, Scalar4 _quat_i,
			Scalar4 _quat_j, Scalar _rcutsq, param_type _params)
	    {
	    // call constructor for underlying struct
	    s = DirectionalEvaluatorPairSphere(_dr,_quat_i,_quat_j,
			    _rcutsq,_params);
	    }

        //! uses diameter
        //! 
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! whether pair potential requires charges
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }
        
        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
            \param isoModulator Output parameter to write the amount of modulation of the isotropic part
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPairJanusSphere.
            
            \return Always true
        */
        DEVICE bool evalPair(Scalar3& force, Scalar& isoModulator, Scalar3& torque_i, Scalar3& torque_j)
            {
            // common calculations
            Scalar modi = s.Modulatori();
            Scalar modj = s.Modulatorj();
	    Scalar modPi = s.ModulatorPrimei();
	    Scalar modPj = s.ModulatorPrimej();

	    // the overall modulation
	    isoModulator = Scalar(1.0)-modi*modj;

	    // intermediate calculations
	    Scalar iPj = modPi*modj;
	    Scalar jPi = modPi*modj;
            
	    // torque on ith
	    torque_i.x = iPj*(s.dr.z*s.ei.y-s.dr.y*s.ei.z);
	    torque_i.y = iPj*(s.dr.x*s.ei.z-s.dr.z*s.ei.x);
	    torque_i.z = iPj*(s.dr.y*s.ei.x-s.dr.x*s.ei.y);

	    // torque on jth - note sign is opposite ith!
	    torque_j.x = jPi*(s.dr.y*s.ej.z-s.dr.z*s.ej.y);
	    torque_j.y = jPi*(s.dr.z*s.ej.x-s.dr.x*s.ej.z);
	    torque_j.z = jPi*(s.dr.x*s.ej.y-s.dr.y*s.ej.x);

	    // compute force contribution
	    force.x = -(iPj*(s.ei.x/s.magdr-s.doti*s.dr.x/s.drsq)
			    -jPi*(s.ej.x/s.magdr-s.dotj*s.dr.x/s.drsq));
	    force.y = -(iPj*(s.ei.y/s.magdr-s.doti*s.dr.y/s.drsq)
			    -jPi*(s.ej.y/s.magdr-s.dotj*s.dr.y/s.drsq));
	    force.z = -(iPj*(s.ei.z/s.magdr-s.doti*s.dr.z/s.drsq)
			    -jPi*(s.ej.z/s.magdr-s.dotj*s.dr.z/s.drsq));
	    

            return true;
            }
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName() { return std::string("depjsc") }
        #endif

    private:
	EvaluatorPairSphereStruct s;

    };

#endif // __DirectionalEvaluatorPairSphere__

