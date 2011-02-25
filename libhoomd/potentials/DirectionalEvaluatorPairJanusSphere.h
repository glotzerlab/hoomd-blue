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

#ifndef __DirectionalEvaluatorPairJanusSphere__
#define __DirectionalEvaluatorPairJanusSphere__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"
#include "DirectionalEvaluatorPair.h"

/*! \file DirectionalEvaluatorPairJanusSphere.h
    \brief Defines the pair evaluator class for Janus spheres
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

struct EvaluatorPairJanusSphereStruct
    {
    EvaluatorPairJanusSphereStruct(Scalar3& _dr, Scalar3& _ei, Scalar3& _ej, Scalar _w, Scalar _costheta) :
        dr(_dr), ei(_ei), ej(_ej), w(_w) costheta(_costheta)
        {
	drsq = dr.x*dr.x+dr.y*dr.y+dr.z*dr.z;
	magdr = sqrt(drsq);
	doti =  (dr.x*ei.x+dr.y*ei.y+dr.z*ei.z)/magdr;
	dotj = -(dr.x*ej.x+dr.y*ej.y+dr.z*ej.z)/magdr;
	}

    DEVICE inline Scalar Modulatori(void)
        {
	return Scalar(1.0)/(1.0+exp(-w*(doti-costheta)));
	}

    DEVICE inline Scalar Modulatorj(void)
        {
	return Scalar(1.0)/(1.0+exp(-w*(dotj-costheta)));
	}

    DEVICE Scalar ModulatorPrimei(void)
        {
	Scalar fact = Modulatori();
	return w*exp(-w*(doti-costheta))*fact*fact;
	}

    DEVICE Scalar ModulatorPrimej(void)
        {
	Scalar fact = Modulatorj();
	return w*exp(-w*(dotj-costheta))*fact*fact;
	}


    Scalar3 dr;
    Scalar3 ei;
    Scalar3 ej;
    Scalar w;
    Scalar costheta;
    Scalar drsq;
    Scalar magdr;
    Scalar doti;
    Scalar dotj;
    };

//! Class for evaluating the directional part of anisotropic pair interations
/*! <b>General Overview</b>

    Provides class for generalized diblock Janus spheres. It uses preexisting isotropic pair evaluators to do much of
    the work.
*/
class DirectionalEvaluatorPairJanusSphere : public DirectionalEvaluatorPair<EvaluatorPairJanusSphereStruct>
    {
    public:
        //! uses diameter
        //! 
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        //! This function is pure virtual
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
            Scalar modi = param.Modulatori();
            Scalar modj = param.Modulatorj();
	    Scalar modPi = param.ModulatorPrimei();
	    Scalar modPj = param.ModulatorPrimej();

	    // the overall modulation
	    isoModulator = modi*modj;

	    // intermediate calculations
	    Scalar iPj = modPi*modj;
	    Scalar jPi = modPi*modj;
            
	    // torque on ith
	    torque_i.x = iPj*(param.dr.y*param.ei.z-param.dr.z*param.ei.y);
	    torque_i.y = iPj*(param.dr.z*param.ei.x-param.dr.x*param.ei.z);
	    torque_i.z = iPj*(param.dr.x*param.ei.y-param.dr.y*param.ei.x);

	    // torque on jth - note sign is opposite ith!
	    torque_j.x = jPi*(param.dr.z*param.ej.y-param.dr.y*param.ej.z);
	    torque_j.y = jPi*(param.dr.x*param.ej.z-param.dr.z*param.ej.x);
	    torque_j.z = jPi*(param.dr.y*param.ej.x-param.dr.x*param.ej.y);

	    // compute force contribution
	    force.x = (iPj*(param.ei.x/param.magdr-param.doti*param.dr.x/param.drsq)
			    -jPi*(param.ej.x/param.magdr-param.dotj*param.dr.x/param.drsq));
	    force.y = (iPj*(param.ei.y/param.magdr-param.doti*param.dr.y/param.drsq)
			    -jPi*(param.ej.y/param.magdr-param.dotj*param.dr.y/param.drsq));
	    force.z = (iPj*(param.ei.z/param.magdr-param.doti*param.dr.z/param.drsq)
			    -jPi*(param.ej.z/param.magdr-param.dotj*param.dr.z/param.drsq));
	    

            return true;
            }
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName() { return std::string("depjs") }
        #endif

    };

//! Class for evaluating the directional part of anisotropic pair interations
/*! <b>General Overview</b>

    Provides class for generalized diblock Janus spheres. It uses preexisting isotropic pair evaluators to do much of
    the work. This does the "non-decorated" part of the sphere.
*/
class DirectionalEvaluatorPairJanusSphereComplement
    : public DirectionalEvaluatorPair<EvaluatorPairJanusSphereStruct>
    {
    public:
        //! uses diameter
        //! 
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        //! This function is pure virtual
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
        DEVICE bool evalPairJanusSphere(Scalar3& force, Scalar& isoModulator, Scalar3& torque_i, Scalar3& torque_j)
            {
            // common calculations
            Scalar modi = param.Modulatori();
            Scalar modj = param.Modulatorj();
	    Scalar modPi = param.ModulatorPrimei();
	    Scalar modPj = param.ModulatorPrimej();

	    // the overall modulation
	    isoModulator = Scalar(1.0)-modi*modj;

	    // intermediate calculations
	    Scalar iPj = modPi*modj;
	    Scalar jPi = modPi*modj;
            
	    // torque on ith
	    torque_i.x = iPj*(param.dr.z*param.ei.y-param.dr.y*param.ei.z);
	    torque_i.y = iPj*(param.dr.x*param.ei.z-param.dr.z*param.ei.x);
	    torque_i.z = iPj*(param.dr.y*param.ei.x-param.dr.x*param.ei.y);

	    // torque on jth - note sign is opposite ith!
	    torque_j.x = jPi*(param.dr.y*param.ej.z-param.dr.z*param.ej.y);
	    torque_j.y = jPi*(param.dr.z*param.ej.x-param.dr.x*param.ej.z);
	    torque_j.z = jPi*(param.dr.x*param.ej.y-param.dr.y*param.ej.x);

	    // compute force contribution
	    force.x = -(iPj*(param.ei.x/param.magdr-param.doti*param.dr.x/param.drsq)
			    -jPi*(param.ej.x/param.magdr-param.dotj*param.dr.x/param.drsq));
	    force.y = -(iPj*(param.ei.y/param.magdr-param.doti*param.dr.y/param.drsq)
			    -jPi*(param.ej.y/param.magdr-param.dotj*param.dr.y/param.drsq));
	    force.z = -(iPj*(param.ei.z/param.magdr-param.doti*param.dr.z/param.drsq)
			    -jPi*(param.ej.z/param.magdr-param.dotj*param.dr.z/param.drsq));
	    

            return true;
            }
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName() { return std::string("depjs") }
        #endif

    };


#endif // __DirectionalEvaluatorPairJanusSphere__

