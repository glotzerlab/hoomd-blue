// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_DPD_H__
#define __PAIR_EVALUATOR_DPD_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

/*! \file EvaluatorPairDPDThermo.h
    \brief Defines the pair evaluator class for the DPD conservative potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating the DPD Thermostat pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>DPD Thermostat and Conservative specifics</b>

    EvaluatorPairDPDThermo::evalForceAndEnergy evaluates the function:
    \f[ V_{\mathrm{DPD-C}}(r) = A \cdot \left( r_{\mathrm{cut}} - r \right)
                        - \frac{1}{2} \cdot \frac{A}{r_{\mathrm{cut}}} \cdot
   \left(r_{\mathrm{cut}}^2 - r^2 \right)\f]

    The DPD Conservative potential does not need charge. One parameter is specified and
   stored in a Scalar. \a A is placed in \a param.

    EvaluatorPairDPDThermo::evalForceEnergyThermo evaluates the function:
    \f{eqnarray*}
    F =   F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +  F_{\mathrm{D,ij}}(v_{ij}) \\
    \f}

    \f{eqnarray*}
    F_{\mathrm{C}}(r) = & A \cdot  w(r_{ij}) \\
    F_{\mathrm{R, ij}}(r_{ij}) = & - \theta_{ij}\sqrt{3} \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot
   w(r_{ij})  \\
    F_{\mathrm{D, ij}}(r_{ij}) = & - \gamma w^2(r_{ij})\left( \hat r_{ij} \circ v_{ij} \right)  \\
    \f}

    \f{eqnarray*}
    w(r_{ij}) = &\left( 1 - r/r_{\mathrm{cut}} \right)  & r < r_{\mathrm{cut}} \\
                     = & 0 & r \ge r_{\mathrm{cut}} \\
    \f}
    where \f$\hat r_{ij} \f$ is a normalized vector from particle i to particle j, \f$ v_{ij} = v_i
   - v_j \f$, and \f$ \theta_{ij} \f$ is a uniformly distributed random number in the range [-1, 1].

    The DPD Thermostat potential does not need charge. Two parameters are specified and
   stored in a Scalar. \a A and \a gamma are placed in \a param.

    These are related to the standard lj parameters sigma and epsilon by:
    - \a A = \f$ A \f$
    - \a gamma = \f$ \gamma \f$

*/
class EvaluatorPairDPDThermoDPD
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar A;
        Scalar gamma;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // CUDA memory hints
        void set_memory_hints() const { }
#endif
#ifndef __HIPCC__
        param_type() : A(0), gamma(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            A = v["A"].cast<Scalar>();
            // protect against a user setting gamma to 0 in dpd
            if (v.contains("gamma"))
                {
                auto gam = v["gamma"].cast<Scalar>();
                if (gam == 0)
                    throw std::invalid_argument(
                        "Cannot set gamma to 0 in DPD, try using DPDConservative instead.");
                else
                    gamma = gam;
                }
            else
                gamma = 0;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["A"] = A;
            if (gamma != 0.0)
                {
                v["gamma"] = gamma;
                }
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairDPDThermoDPD(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), a(_params.A), gamma(_params.gamma)
        {
        }

    //! Set i and j, (particle tags), and the timestep
    DEVICE void
    set_seed_ij_timestep(uint16_t seed, unsigned int i, unsigned int j, uint64_t timestep)
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

    //! Yukawa doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy using the conservative force only
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

            // force is easy to calculate
            force_divr = a * (rinv - rcutinv);
            pair_eng = a * (rcut - r) - Scalar(1.0 / 2.0) * a * rcutinv * (rcutsq - rsq);

            return true;
            }
        else
            return false;
        }

    //! Evaluate the force and energy using the thermostat
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param force_divr_cons Output parameter to write the computed conservative force divided by
       r. \param pair_eng Output parameter to write the computed pair energy \param energy_shift
       Ignored. DPD always goes to 0 at the cutoff. \note There is no need to check if rsq < rcutsq
       in this method. Cutoff tests are performed in PotentialPair.

        \note The conservative part \b only must be output to \a force_divr_cons so that the virial
       may be computed correctly.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceEnergyThermo(Scalar& force_divr,
                                      Scalar& force_divr_cons,
                                      Scalar& pair_eng,
                                      bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

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

            hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::RNGIdentifier::EvaluatorPairDPDThermo, m_timestep, m_seed),
                hoomd::Counter(m_oi, m_oj));

            // Generate a single random number
            Scalar alpha = hoomd::UniformDistribution<Scalar>(-1, 1)(rng);

            // conservative dpd
            // force_divr = FDIV(a,r)*(Scalar(1.0) - r*rcutinv);
            force_divr = a * (rinv - rcutinv);

            //  conservative force only
            force_divr_cons = force_divr;

            //  Drag Term
            force_divr -= gamma * m_dot * (rinv - rcutinv) * (rinv - rcutinv);

            //  Random Force
            force_divr
                += fast::rsqrt(m_deltaT / (m_T * gamma * Scalar(6.0))) * (rinv - rcutinv) * alpha;

            // conservative energy only
            pair_eng = a * (rcut - r) - Scalar(1.0 / 2.0) * a * rcutinv * (rcutsq - rsq);

            return true;
            }
        else
            return false;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("dpd");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;          //!< Stored rsq from the constructor
    Scalar rcutsq;       //!< Stored rcutsq from the constructor
    Scalar a;            //!< a parameter for potential extracted from params by constructor
    Scalar gamma;        //!< gamma parameter for potential extracted from params by constructor
    uint16_t m_seed;     //!< User set seed for thermostat PRNG
    unsigned int m_i;    //!< index of first particle (should it be tag?).  For use in PRNG
    unsigned int m_j;    //!< index of second particle (should it be tag?). For use in PRNG
    uint64_t m_timestep; //!< timestep for use in PRNG
    Scalar m_T;          //!< Temperature for Themostat
    Scalar m_dot;        //!< Velocity difference dotted with displacement vector
    Scalar m_deltaT;     //!<  timestep size stored from constructor
    };

#undef DEVICE

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_DPD_H__
