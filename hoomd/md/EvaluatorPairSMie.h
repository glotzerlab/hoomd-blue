// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_SMie_H__
#define __PAIR_EVALUATOR_SMie_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "EvaluatorPairLJ.h"

/*! \file EvaluatorPairSMie.h
    \brief Defines the pair evaluator class for shifted Mie potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Shifter Mie pair potential
/*! <b>General Overview</b>

    See EvaluatorPairSLJ and EvaluatorPairMie

    <b>SMie specifics</b>

    EvaluatorPairSMie evaluates the function:
    \f{eqnarray*}
    V_{\mathrm{SMie}}(r)  = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m} \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left( \frac{\sigma}{r-\Delta} \right)^{n} -
                \left( \frac{\sigma}{r-\Delta} \right)^{m} \right] r \l (r_{\mathrm{cut}} + \Delta)\\
                = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
    \f}
    where  \f$ \Delta is specified by the user, conventionally calcualted as, \f$ \Delta = (d_i + d_j)/2 - \sigma \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.

    The SMie potential does not need charge, but does need diameter. Two parameters are specified and stored in a
    Scalar5. \a sm1 \a sm2 \a sm3 \a sm4 \a sm5 are stored in \a params.x \a params.y \a params.z and \a params.w \a params.v respectively.


    These are related to the standard lj parameters sigma and epsilon and the variable exponents n and m by:
    - \a sm1 = epsilon * pow(sigma,n) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a sm2 = epsilon * pow(sigma,m) * (n/(n-m)) * power(n/m,m/(n-m))
    - \a sm3 = n
    - \a sm4 = m
    - \a sm5 = Delta

    Due to the way that SMie modifies the cutoff condition, it will not function properly with the xplor shifting mode.
*/
class EvaluatorPairSMie
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            Scalar sm1;
            Scalar sm2;
            Scalar sm3;
            Scalar sm4;
            Scalar sm5;

            #ifndef ENABLE_HIP
            //! set CUDA memory hints
            void set_memory_hint() const {}
            #endif

            #ifndef __HIPCC__
            param_type() : sm1(0), sm2(0), sm3(0), sm4(0), sm5(0) {}

            param_type(pybind11::dict v)
                {
                sm3 = v["n"].cast<Scalar>();
                sm4 = v["m"].cast<Scalar>();

                auto sigma(v["sigma"].cast<Scalar>());
                auto epsilon(v["epsilon"].cast<Scalar>());

                Scalar outFront = (sm3/(sm3-sm4)) * fast::pow(sm3/sm4, sm4/(sm3-sm4));
                sm1 = outFront * epsilon * fast::pow(sigma, sm3);
                sm2 = outFront * epsilon * fast::pow(sigma, sm4);

                sm5 = v["Delta"].cast<Scalar>();
                }

            pybind11::dict asDict()
                {
                pybind11::dict v;
                v["n"] = sm3;
                v["m"] = sm4;

                Scalar sigma = fast::pow(sm1 / sm2, 1 / (sm3 - sm4));
                Scalar epsilon = sm1 / fast::pow(sigma, sm3) * (sm3 - sm4) / sm3 * fast::pow(sm3 / sm4, sm4 / (sm4 - sm3));

                v["epsilon"] = epsilon;
                v["sigma"] = sigma;

                v["Delta"] = sm5;
                return v;
                }
            #endif
            }
            #ifdef SINGLE_PRECISION
            __attribute__((aligned(8)));
            #else
            __attribute__((aligned(16)));
            #endif

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _n First, larger exponent that captures hard-core repulsion
            \param _m Second, smaller exponent that captures attraction
            \param _params Per type pair parameters of this potential
            \param _Delta Horizontal shift in r
        */
        DEVICE EvaluatorPairSMie(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), sm1(_params.sm1), sm2(_params.sm2), sm3(_params.sm3), sm4(_params.sm4), sm5(_params.sm5)
            {
            }

        //! SMie doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! SMie doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
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
            // precompute some quantities
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;
            Scalar r = fast::sqrt(rsq);
            Scalar rinv = fast::rsqrt(rsq);

            // compute the force divided by r in force_divr
            if (r < (rcut + sm5) && sm1 != 0)
                {
                Scalar rmd = r - sm5;
                Scalar rmdinv = Scalar(1.0) / rmd;
                Scalar rmd2inv = rmdinv * rmdinv;
                Scalar rmdninv = fast::pow(rmd2inv,sm3/Scalar(2.0));
                Scalar rmdminv = fast::pow(rmd2inv,sm4/Scalar(2.0));
                force_divr= rinv * rmdinv * (sm3*sm1*rmdninv-sm4*sm2*rmdminv);

                pair_eng = sm1 * rmdninv - sm2 * rmdminv;

                if (energy_shift)
                    {
                    Scalar rcutninv = fast::pow(rcutsq,-sm3/Scalar(2.0));
                    Scalar rcutminv = fast::pow(rcutsq,-sm4/Scalar(2.0));
                    pair_eng -= sm1 * rcutninv - sm2 * rcutminv;
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("smie");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar sm1;     //!< mie1 parameter extracted from the params passed to the constructor
        Scalar sm2;     //!< mie2 parameter extracted from the params passed to the constructor
        Scalar sm3;     //!< mie3 parameter extracted from the params passed to the constructor
        Scalar sm4;     //!< mie4 parameter extracted from the params passed to the constructor
        Scalar sm5;   //!< Delta parameter extracted from the call to setDiameter
    };

#endif // __PAIR_EVALUATOR_SMIE_H__
