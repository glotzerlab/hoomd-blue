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
// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_DPD_H__
#define __PAIR_EVALUATOR_DPD_H__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

#ifdef NVCC
#include "saruprngCUDA.h"
#else
#include "saruprng.h"
#endif


/*! \file EvaluatorPairDPDThermo.h
    \brief Defines the pair evaluator class for the DPD conservative potential
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

// call different division functions on the host / device
#ifdef NVCC
#define FDIV(x,y) __fdiv_rn(x,y)
#else
#define FDIV(x,y) x/y
#endif


// call different Saru PRNG initializers on the host / device
//! is SaruGPU when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define SARU(ix,iy,iz) SaruGPU saru( (ix) , (iy) , (iz) )
#define CALL_SARU(x,y) saru.f( (x), (y))
#else
#define SARU(ix, iy, iz) Saru saru( (ix) , (iy) , (iz))
#define CALL_SARU(x,y) saru.f( (x), (y))
#endif

//! Class for evaluating the DPD Thermostat pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ
    
    <b>DPD Conservative specifics</b>
    
    EvaluatorPairDPDThermo evaluates the function:
    \f[ V_{\mathrm{DPD-C}}(r) = a \cdot \left( r_{\mathrm{cut}} - r \right) 
						- \frac{1}{2} \cdot \frac{a}{r_{\mathrm{cut}}} \cdot \left(r_{\mathrm{cut}}^2 - r^2 \right)\f]
    
    The DPD Conservative potential does not need charge or diameter. One parameters is specified and stored in a Scalar. 
    \a a is placed in \a param.
    
    These are related to the standard lj parameters sigma and epsilon by:
    - \a a = \f$ a \f$    
*/
class EvaluatorPairDPDThermo
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;
        
        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDPDThermo(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), a(_params.x), gamma(_params.y)
            {
            }

        //! Set i and j, (particle indices, or should it be tags), and the timestep
        DEVICE void set_seed_ij_timestep(unsigned int seed, unsigned int i, unsigned int j, unsigned int timestep) 
            {
            m_seed = seed;
            m_i = i;
            m_j = j;
            m_timestep = timestep;
            }
            
        //! Set the timestep size
        DEVICE void setDeltaT(Scalar dt) 
            {
            m_deltaT = dt;
            }

        //! Set the velocity term
        DEVICE void setRDotV(Scalar dot) 
            {
            m_dot = dot;
            }
            
        //! Set the temperature
        DEVICE void setT(Scalar Temp) 
            {
            m_T = Temp;
            }
        
        //! Does not use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Yukawa doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }
        
        //! Evaluate the force and energy using the conservative force only
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
             // Tried using more Exact versions of these functions, no difference seen   
             //   Scalar r = SQRT(rsq);
             //   Scalar rinv = FDIV(Scalar(1.0), r);
             //   Scalar rcut = SQRT(rcutsq);
             //   Scalar rcutinv = FDIV(Scalar(1.0), rcut);
               
                Scalar rinv = RSQRT(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar rcutinv = RSQRT(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;

                // force is easy to calculate
                //force_divr = FDIV(a,r)*(Scalar(1.0) - r*rcutinv);
                force_divr = a*(rinv - rcutinv);
                pair_eng = a * (rcut - r) - Scalar(1.0/2.0) * a * rcutinv * (rcutsq - rsq);

                //if (energy_shift)
                    //{
                    // do nothing in energy_shift mode: DPD-C goes to 0 at the cutoff
                    //}
                return true;
                }
            else
                return false;
            }
            
        //! Evaluate the force and energy using the thermostat
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceEnergyThermo(Scalar& force_divr, Scalar& pair_eng)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                Scalar rinv = RSQRT(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar rcutinv = RSQRT(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;

                // force calculation
                
                // initialize the RNG
                
                // Mix the two indices to generate a third indice using the Cantor Pairing Function, a primitive 
                // recursive bijection that maps two natural numbers to a third unique natural number.  This method will
                // start to overflow the 32-bit integer around i, j ~ 46,000.  Note, that given
                // the limited size of integer that can be held in a GPU/CPU, how the GPU/CPU handles overflow is important.
                // Per the ANSI.C spec.  "A computation involving unsigned operands can never overflow, because a result
                // that cannot be represented by the resulting unsigned integer type is reduced modulo the number that is
                // one greater than the largest value that can be represented by the resulting unsigned integer type"
                // We also assess the the possibility of collision  (e.g. i1, j1 map to same as i2 j2, assuming the unsigned
                // integers wrap) on any given timestep is slim for systems where the number of particles contained in
                // a sphere of r_cut is a small percent of the entire system.  Also, the larger the number of particles 
                // within r_cut, the less the impact of a "collision". 
                
                unsigned int mixij = ((m_i + m_j)*(m_i + m_j + 1))/2;  //needs final term

                if (m_i > m_j)
                    mixij += m_j;
                else    
                    mixij += m_i;
                
                SARU(mixij, m_seed, m_timestep);
                
                // Generate a single random number
                Scalar alpha = CALL_SARU(-1,1) ;
                
                // conservative dpd
                //force_divr = FDIV(a,r)*(Scalar(1.0) - r*rcutinv);
                force_divr = a*(rinv - rcutinv);
                
                //  Drag Term 
                force_divr -=  gamma*m_dot*(rinv - rcutinv)*(rinv - rcutinv);
                
                //  Random Force 
                force_divr += RSQRT(m_deltaT/(m_T*gamma*Scalar(6.0)))*(rinv - rcutinv)*alpha;
                
                //conservative energie only
                pair_eng = a * (rcut - r) - Scalar(1.0/2.0) * a * rcutinv * (rcutsq - rsq);  

                //if (energy_shift)
                    //{
                    // do nothing in energy_shift mode: DPD-C goes to 0 at the cutoff
                    //}
                return true;
                }
            else
                return false;
            }
        
        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("dpd");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar a;       //!< a parameter for potential extracted from params by constructor
        Scalar gamma;   //!< gamma parameter for potential extracted from params by constructor
        unsigned int m_seed; //!< User set seed for thermostat PRNG
        unsigned int m_i;   //!< index of first particle (should it be tag?).  For use in PRNG
        unsigned int m_j;   //!< index of second particle (should it be tag?). For use in PRNG
        unsigned int m_timestep; //!< timestep for use in PRNG
        Scalar m_T;         //!< Temperature for Themostat
        Scalar m_dot;       //! < Velocity difference dotted with displacement vector
        Scalar m_deltaT;   //!<  timestep size stored from constructor
    };


#endif // __PAIR_EVALUATOR_DPD_H__

