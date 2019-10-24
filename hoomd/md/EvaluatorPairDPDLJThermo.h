// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

#ifndef __PAIR_EVALUATOR_DPDLJ_H__
#define __PAIR_EVALUATOR_DPDLJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"


/*! \file EvaluatorPairDPDLJThermo.h
    \brief Defines the pair evaluator class for the DPD Thermostat with a Lennard Jones conservative potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
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
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
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

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
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
        /*! \param force_divr Output parameter to write the computed total force divided by r.
            \param force_divr_cons Output parameter to write the computed conservative force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \note The conservative part \b only must be output to \a force_divr_cons so that the virial may be
                  computed correctly.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */

        DEVICE bool evalForceEnergyThermo(Scalar& force_divr, Scalar& force_divr_cons, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && lj1!= 0)
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                Scalar rcutinv = fast::rsqrt(rcutsq);

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

                hoomd::RandomGenerator rng(hoomd::RNGIdentifier::EvaluatorPairDPDThermo, m_seed, m_oi, m_oj, m_timestep);


                // Generate a single random number
                Scalar alpha = hoomd::UniformDistribution<Scalar>(-1,1)(rng);

                // conservative lj
                force_divr = r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);
                force_divr_cons = force_divr;

                //  Drag Term
                force_divr -=  gamma*m_dot*(rinv - rcutinv)*(rinv - rcutinv);

                //  Random Force
                force_divr += fast::rsqrt(m_deltaT/(m_T*gamma*Scalar(6.0)))*(rinv - rcutinv)*alpha;

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

       std::string getShapeSpec() const
          {
          throw std::runtime_error("Shape definition not supported for this pair potential.");
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

#undef DEVICE

#endif // __PAIR_EVALUATOR_DPDLJ_H__
