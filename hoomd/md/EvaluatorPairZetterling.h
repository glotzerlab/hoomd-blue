// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_ZETTERLING_H__
#define __PAIR_EVALUATOR_ZETTERLING_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairZetterling.h
    \brief Defines the pair evaluator class for Zetterling potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc DEVICE is __host__ __device__ when included in nvcc and blank when
// included into the host compiler
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
//! Class for evaluating the Zetterling pair potential
/*! <b>General Overview</b>

    <b>OPP specifics</b>

    EvaluatorPairZetterling evaluates the function:
    \f{equation*}
    V_{\mathrm{Zetterling}}(r) =
    A \frac{\exp{(\alpha r)\cos{(2 k_F r)}}}{r^3}
    + B \left( \frac{\sigma}{r} \right)^n)
    \f}

*/
class EvaluatorPairZetterling
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar A;
        Scalar alpha;
        Scalar kf;
        Scalar B;
        Scalar sigma;
        Scalar n;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : A(0), alpha(0), kf(0), B(0), sigma(0), n(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            A = v["A"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            kf = v["kf"].cast<Scalar>();
            B = v["B"].cast<Scalar>();
            sigma = v["sigma"].cast<Scalar>();
            n = v["n"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["A"] = A;
            v["alpha"] = alpha;
            v["kf"] = kf;
            v["B"] = B;
            v["sigma"] = sigma;
            v["n"] = n;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairZetterling(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! Zetterling doesn't use charge
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
    /*! \param force_divr Output parameter to write the computed force
     * divided by r.
     *  \param pair_eng Output parameter to write the computed pair energy
     *  \param energy_shift If true, the potential must be shifted so that
     *      V(r) is continuous at the cutoff

     *  \return True if they are evaluated or false if they are not because
     *  we are beyond the cutoff
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        if (rsq < rcutsq)
            {
            // Get quantities need for both energy and force calculation
            Scalar r(fast::sqrt(rsq));
            Scalar eval_sin, eval_cos;
            fast::sincos(Scalar(2.0) * params.kf * r, eval_sin, eval_cos);
            Scalar screening(fast::exp(params.alpha * r));
            Scalar inv_r(Scalar(1.0) / r);
            Scalar inv_r_2(Scalar(1.0) / rsq);
            Scalar inv_r_3(inv_r_2 * inv_r);
            Scalar inv_r_5(inv_r_3 * inv_r_2);
            Scalar power_sigma_over_r(fast::pow(params.sigma * inv_r, params.n));

            // Compute energy
            Scalar term1(params.A * screening * eval_cos * inv_r_3);
            Scalar term2(params.B * power_sigma_over_r);
            pair_eng = term1 + term2;

            // Compute force
            Scalar deriv_term1(
                -params.A * screening
                * ((params.alpha * r - Scalar(3.0)) * eval_cos - 2 * params.kf * r * eval_sin)
                * inv_r_5);
            Scalar deriv_term2(params.B * params.n * power_sigma_over_r * inv_r_2);
            force_divr = deriv_term1 + deriv_term2;

            if (energy_shift)
                {
                Scalar r_cut(fast::sqrt(rcutsq));
                Scalar screening_r_cut(fast::exp(params.alpha * r_cut));
                Scalar inv_rcut(Scalar(1.0) / r_cut);
                Scalar inv_rcut_3(inv_rcut * inv_rcut * inv_rcut);
                Scalar term1_rcut(params.A * screening_r_cut
                                  * fast::cos(Scalar(2.0) * params.kf * r_cut) * inv_rcut_3);
                Scalar term2_rcut(params.B * fast::pow(params.sigma * inv_rcut, params.n));
                pair_eng -= term1_rcut + term2_rcut;
                }

            return true;
            }
        else
            {
            return false;
            }
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
        return std::string("zetterling");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;        /// Stored rsq from the constructor
    Scalar rcutsq;     /// Stored rcutsq from the constructor
    param_type params; /// Stored pair parameters for a given type pair
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_ZETTERLING_H__
