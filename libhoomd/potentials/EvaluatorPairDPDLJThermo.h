/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: phillicl

#ifndef __PAIR_EVALUATOR_DPDLJ_H__
#define __PAIR_EVALUATOR_DPDLJ_H__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

#ifdef NVCC
#include "saruprngCUDA.h"
#else
#include "saruprng.h"
#endif


/*! \file EvaluatorPairDPDLJThermo.h
    \brief Defines the pair evaluator class for the DPD Thermostat with a Lennard Jones conservative potential
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

// call different Saru PRNG initializers on the host / device
//! SARU is SaruGPU Class when included in nvcc and Saru Class when included into the host compiler
#ifdef NVCC
#define SARU(ix,iy,iz) SaruGPU saru( (ix) , (iy) , (iz) )
#else
#define SARU(ix, iy, iz) Saru saru( (ix) , (iy) , (iz) )
#endif

// use different Saru PRNG returns on the host / device
//! CALL_SARU is currently define to return a random float for both the GPU and Host.  By changing saru.f to saru.d, a double could be returned instead.
#ifdef NVCC
#define CALL_SARU(x,y) saru.f( (x), (y))
#else
#define CALL_SARU(x,y) saru.f( (x), (y))
#endif



//! Class for evaluating the DPD Thermostat pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ
    
    <b>DPD Thermostat and Conservative LJ specifics</b>
    
    EvaluatorPairDPDLJThermo::evalForceAndEnergy evaluates the Lennard-Jones function (see EvaluatorPairLJ.  However it is not intended to be used.  
    It is written for completeness sake only.
            
    EvaluatorPairDPDLJThermo::evalForceEnergyThermo evaluates the function:
    \f{eqnarray*}  
    F =   F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +  F_{\mathrm{D,ij}}(v_{ij}) \\
    \f}

    \f{eqnarray*}
    F_{\mathrm{C}}(r) = & \partial V_{\mathrm{LJ}} / \partial r    \\
    F_{\mathrm{R, ij}}(r_{ij}) = & - \theta_{ij}\sqrt{3} \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot w(r_{ij})  \\
    F_{\mathrm{D, ij}}(r_{ij}) = & - \gamma w^2(r_{ij})\left( \hat r_{ij} \circ v_{ij} \right)  \\
    \f}

    where
     \f{eqnarray*}
    V_{\mathrm{LJ}}(r) = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
                                            \alpha \left( \frac{\sigma}{r} \right)^{6} \right]  & r < r_{\mathrm{cut}} \\
                                            = & 0 & r \ge r_{\mathrm{cut}} \\
    \f}
    and
    \f{eqnarray*}
    w(r_{ij}) = &\left( 1 - r/r_{\mathrm{cut}} \right)  & r < r_{\mathrm{cut}} \\
                     = & 0 & r \ge r_{\mathrm{cut}} \\
    \f}
    where \f$\hat r_{ij} \f$ is a normalized vector from particle i to particle j, \f$ v_{ij} = v_i - v_j \f$, and \f$ \theta_{ij} \f$ is a uniformly distributed
    random number in the range [-1, 1].                        
 
    The LJ potential does not need diameter or charge. Three parameters are specified and stored in a Scalar4. \a lj1 is
    placed in \a params.x, \a lj2 is in \a params.y and \a gamma in \a params.z. The final parameter \a params.w is not used and set to zero.
    
    lj1 and lj2 are related to the standard lj parameters sigma and epsilon by:
    - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);   
     
*/
class EvaluatorPairDPDLJThermo
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;
        
        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDPDLJThermo(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.x), lj2(_params.y), gamma(_params.z)
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
        
        //! LJ does not use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }
                
        
        //! Evaluate the force and energy
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
            if (rsq < rcutsq && lj1 != 0)
                {
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                force_divr= r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);
                
                pair_eng = r6inv * (lj1*r6inv - lj2);
                
                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= rcut6inv * (lj1*rcut6inv - lj2);
                    }
                return true;
                }
            else
                return false;
            }
        
        //! Evaluate the force and energy using the thermostat
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff            
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
                
        DEVICE bool evalForceEnergyThermo(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && lj1!= 0)
                {
                Scalar rinv = RSQRT(rsq);
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                Scalar rcutinv = RSQRT(rcutsq);

                // force calculation
                
                unsigned int m_oi, m_oj;
                // initialize the RNG
                if (m_i > m_j)
                   {
                   m_oi = m_j;
                   m_oj = m_i;
                   }    
                else
                   {
                   m_oi = m_i;
                   m_oj = m_j;
                   }                  
                    
                SARU(m_oi, m_oj, m_seed + m_timestep);
                
                
                // Generate a single random number
                Scalar alpha = CALL_SARU(-1,1) ;
                
                // conservative lj
                force_divr = r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);
                                
                //  Drag Term 
                force_divr -=  gamma*m_dot*(rinv - rcutinv)*(rinv - rcutinv);
                
                //  Random Force 
                force_divr += RSQRT(m_deltaT/(m_T*gamma*Scalar(6.0)))*(rinv - rcutinv)*alpha;
                
                //conservative energy only
                pair_eng = r6inv * (lj1*r6inv - lj2);


                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= rcut6inv * (lj1*rcut6inv - lj2);
                    }
                    
                      
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
            return std::string("dpdlj");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
        Scalar gamma;   //!< gamma parameter for potential extracted from params by constructor
        unsigned int m_seed; //!< User set seed for thermostat PRNG
        unsigned int m_i;   //!< index of first particle (should it be tag?).  For use in PRNG
        unsigned int m_j;   //!< index of second particle (should it be tag?). For use in PRNG
        unsigned int m_timestep; //!< timestep for use in PRNG
        Scalar m_T;         //!< Temperature for Themostat
        Scalar m_dot;       //!< Velocity difference dotted with displacement vector
        Scalar m_deltaT;   //!<  timestep size stored from constructor
    };

#undef SARU

#endif // __PAIR_EVALUATOR_DPDLJ_H__

