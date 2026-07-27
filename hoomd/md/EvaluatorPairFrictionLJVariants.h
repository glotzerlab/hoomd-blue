// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__
#define __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__

/*! \file EvaluatorPairFrictionLJVariants.h
    \brief Defines the different variants of frictional contact interactions
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc.  HOSTDEVICE is __host__ __device__ when included in nvcc and blank
// when included into the host compiler

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairFrictionLJLinear
    : public EvaluatorPairFrictionLJBase<EvaluatorPairFrictionLJLinear>
    {
    public:
    using EvaluatorPairFrictionLJBase<EvaluatorPairFrictionLJLinear>::EvaluatorPairFrictionLJBase;

    HOSTDEVICE void eval_factors(Scalar& factor_f, Scalar& factor_r, Scalar w, Scalar du)
        {
        Scalar f = gamma;

        factor_f = w * f;
        factor_r = fast::sqrt(factor_f);
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "FrictionLJLinear";
        }
#endif
    };

class EvaluatorPairFrictionLJCoulomb
    : public EvaluatorPairFrictionLJBase<EvaluatorPairFrictionLJCoulomb>
    {
    public:
    using EvaluatorPairFrictionLJBase<EvaluatorPairFrictionLJCoulomb>::EvaluatorPairFrictionLJBase;

    HOSTDEVICE void eval_factors(Scalar& factor_f, Scalar& factor_r, Scalar w, Scalar du)
        {
        Scalar f = kappa / du;
        Scalar D = Scalar(0.0);
        if (pair_temp > 0.0)
            D = kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                * exp(du * du / (2 * pair_temp * nu_ij))
                * erfc(du / fast::sqrt(Scalar(2.0) * pair_temp * nu_ij));

        factor_f = w * f;
        factor_r = fast::sqrt(w * D);
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "FrictionLJCoulomb";
        }
#endif
    };

class EvaluatorPairFrictionLJCoulombNewton
    : public EvaluatorPairFrictionLJBase<EvaluatorPairFrictionLJCoulombNewton>
    {
    public:
    using EvaluatorPairFrictionLJBase<
        EvaluatorPairFrictionLJCoulombNewton>::EvaluatorPairFrictionLJBase;

    HOSTDEVICE void eval_factors(Scalar& factor_f, Scalar& factor_r, Scalar w, Scalar du)
        {
        Scalar f = Scalar(0.0);
        Scalar D = Scalar(0.0);

        if (du < (w * kappa / gamma))
            {
            if (pair_temp > 0.0)
                D = w * kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                        * exp(du * du / (Scalar(2.0) * pair_temp * nu_ij))
                        * erfc(w * kappa / (gamma * fast::sqrt(Scalar(2.0) * pair_temp * nu_ij)))
                    + gamma
                          * (Scalar(1.0)
                             - exp((du * du - (w * kappa / gamma) * (w * kappa / gamma))
                                   / (Scalar(2.0) * pair_temp * nu_ij)));
            f = gamma;
            }
        else
            {
            if (pair_temp > 0.0)
                D = w * kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                    * exp(du * du / (Scalar(2.0) * pair_temp * nu_ij))
                    * erfc(du / fast::sqrt(Scalar(2.0) * pair_temp * nu_ij));
            f = w * kappa / du;
            }

        factor_f = f;
        factor_r = fast::sqrt(D);
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "FrictionLJCoulombNewton";
        }
#endif
    };
    } // namespace md
    } // namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__
