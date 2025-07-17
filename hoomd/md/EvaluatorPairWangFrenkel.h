// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_WANGFRENKEL_H__
#define __PAIR_EVALUATOR_WANGFRENKEL_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairWangFrenkel.h
    \brief Defines the pair evaluator class for Wang-Frenkel potentials
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
//! Class for evaluating the Wang-Frenkel pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Wang-Frenkel specifics</b>

    EvaluatorPairWang-Frenkel evaluates the function:
    \f[ \epsilon \alpha \left( \left[\frac{\sigma}{r}\right]^{2\mu}
   -1\right)\left(\left[\frac{R}{r}\right]^{2\mu} -1 \left)^{2\nu} \f]
*/
class EvaluatorPairWangFrenkel
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar prefactor;
        Scalar sigma_pow_2m;
        Scalar R_pow_2m;
        int mu;
        int nu;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // set CUDA memory hints
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type() : prefactor(0.0), sigma_pow_2m(0.0), R_pow_2m(0.0), mu(0.0), nu(0.0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            // should probably send a warning if exponents are large (eg >256)
            mu = v["mu"].cast<unsigned int>();
            nu = v["nu"].cast<unsigned int>();

            if (nu == 0 || mu == 0)
                throw std::invalid_argument("Cannot set exponents nu or mu to zero");

            Scalar epsilon = v["epsilon"].cast<Scalar>();
            Scalar sigma = v["sigma"].cast<Scalar>();
            Scalar sigma_sq = sigma * sigma;
            Scalar rcut = v["R"].cast<Scalar>();
            Scalar rcutsq = rcut * rcut;
            Scalar left = (1 + 2 * nu) / (2 * nu * (fast::pow(rcutsq / sigma_sq, mu) - 1));
            Scalar alpha = 2 * nu * fast::pow(rcutsq / sigma_sq, mu) * fast::pow(left, 2 * nu + 1);

            prefactor = epsilon * alpha;
            R_pow_2m = fast::pow(rcutsq, mu);
            sigma_pow_2m = fast::pow(sigma_sq, mu);
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["mu"] = mu;
            v["nu"] = nu;
            Scalar sigma = fast::pow(sigma_pow_2m, 1.0 / (2.0 * Scalar(mu)));
            v["sigma"] = sigma;
            Scalar sigma_sq = sigma * sigma;
            v["R"] = fast::pow(R_pow_2m, 1 / Scalar(2 * mu));
            Scalar rcutsq = fast::pow(R_pow_2m, 1 / Scalar(mu));
            Scalar left = (1 + 2 * nu) / (2 * nu * (fast::pow(rcutsq / sigma_sq, mu) - 1));
            Scalar alpha = 2 * nu * fast::pow(rcutsq / sigma_sq, mu) * fast::pow(left, 2 * nu + 1);

            v["epsilon"] = prefactor / alpha;

            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairWangFrenkel(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), prefactor(_params.prefactor),
          sigma_pow_2m(_params.sigma_pow_2m), R_pow_2m(_params.R_pow_2m), mu(_params.mu),
          nu(_params.nu)
        {
        }

    //! Mie doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
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
        if (rsq < rcutsq && prefactor != 0)
            {
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar rinv_pow_2m = fast::pow(r2inv, mu);
            Scalar sigma_over_r_pow_2m = sigma_pow_2m * rinv_pow_2m;
            Scalar R_over_r_pow_2m = R_pow_2m * rinv_pow_2m;

            Scalar sigma_term = sigma_over_r_pow_2m - 1;
            Scalar R_term = R_over_r_pow_2m - 1;

            Scalar R_term_2num1 = fast::pow(R_term, 2 * nu - 1);
            Scalar R_term_2nu = R_term_2num1 * R_term;

            pair_eng = prefactor * sigma_term * R_term_2nu;
            force_divr = 2 * prefactor * mu * R_term_2num1
                         * (2 * nu * R_over_r_pow_2m * sigma_term + R_term * sigma_over_r_pow_2m)
                         * r2inv;

            if (energy_shift)
                {
                Scalar rcinv_pow_2m = fast::pow(Scalar(1.0) / rcutsq, mu);
                Scalar sigma_over_rc_pow_2m = sigma_pow_2m * rcinv_pow_2m;
                Scalar R_over_rc_pow_2m = R_pow_2m * rcinv_pow_2m;
                Scalar rc_R_term_2nu = fast::pow(R_over_rc_pow_2m - 1, 2 * nu);
                pair_eng -= prefactor * (sigma_over_rc_pow_2m - 1) * rc_R_term_2nu;
                }
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
        return std::string("wangfrenkel");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor

    Scalar prefactor;    //!< Prefactor (epsilon * alpha)
    Scalar sigma_pow_2m; //!< sigma^(2m) stored
    Scalar R_pow_2m;     //!< R^(2m) stored
    unsigned int mu;     //!< mu exponent
    unsigned int nu;     //!< nu exponent
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_WANGFRENKEL_H__
