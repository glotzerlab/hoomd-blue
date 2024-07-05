// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_TETHER_H__
#define __BOND_EVALUATOR_TETHER_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorBondTether.h
    \brief Defines the bond evaluator class for Tethering potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
struct tether_params
    {
    Scalar k_b;
    Scalar l_min;
    Scalar l_c1;
    Scalar l_c0;
    Scalar l_max;

#ifndef __HIPCC__
    tether_params()
        {
        k_b = 10;
        l_min = 0.9;
        l_c1 = 1.2;
        l_c0 = 1.8;
        l_max = 2.1;
        }

    tether_params(Scalar k_b, Scalar l_min, Scalar l_c1, Scalar l_c0, Scalar l_max)
        : k_b(k_b), l_min(l_min), l_c1(l_c1), l_c0(l_c0), l_max(l_max)
        {
        }

    tether_params(pybind11::dict v)
        {
        k_b = v["k_b"].cast<Scalar>();
        l_min = v["l_min"].cast<Scalar>();
        l_c1 = v["l_c1"].cast<Scalar>();
        l_c0 = v["l_c0"].cast<Scalar>();
        l_max = v["l_max"].cast<Scalar>();

        if (!(l_min < l_c1 && l_min < l_c0 && l_min < l_max && l_c1 < l_c0 && l_c1 < l_max
              && l_c0 < l_max))
            {
            throw std::invalid_argument("The input parameters should follow l_min < l_c1 < l_c0"
                                        " < l_max");
            }
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k_b"] = k_b;
        v["l_min"] = l_min;
        v["l_c1"] = l_c1;
        v["l_c0"] = l_c0;
        v["l_max"] = l_max;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

//! Class for evaluating the tethering bond potential
/*! The parameters are:
    - \a k_b (param.x) Bond stiffness
    - \a l_min (param.y) minimum bond length
    - \a l_c1 (param.w) cutoff length of repulsive part
    - \a l_c0 (param.z) cutoff length of attractive part
    - \a l_max (param.a) maximum bond length
*/
class EvaluatorBondTether
    {
    public:
    //! Define the parameter type used by this bond potential evaluator
    typedef tether_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorBondTether(Scalar _rsq, const param_type& _params)
        : rsq(_rsq), k_b(_params.k_b), l_min(_params.l_min), l_c1(_params.l_c1), l_c0(_params.l_c0),
          l_max(_params.l_max)
        {
        }

    //! Tether doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values
    /*! \param qa Charge of particle a
        \param qb Charge of particle b
    */
    DEVICE void setCharge(Scalar qa, Scalar qb) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param bond_eng Output parameter to write the computed bond energy

        \return True if they are evaluated or false if the bond
                energy is not defined
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        Scalar r = sqrt(rsq);
        Scalar U_att = Scalar(0.0);
        Scalar F_att = Scalar(0.0);
        Scalar U_rep = Scalar(0.0);
        Scalar F_rep = Scalar(0.0);

        if (r > l_c0)
            {
            U_att = k_b * (exp(Scalar(1.0) / (l_c0 - r)) / (l_max - r));
            F_att = k_b
                    * (((r - l_max) * exp(Scalar(1.0) / (l_c0 - r)) / (l_c0 - r) / (l_c0 - r)
                        - exp(Scalar(1.0) / (l_c0 - r)))
                       / (l_max - r) / (l_max - r));
            }

        if (r < l_c1)
            {
            U_rep = k_b * (exp(Scalar(1.0) / (r - l_c1)) / (r - l_min));
            F_rep = k_b
                    * (((r - l_min) * exp(Scalar(1.0) / (r - l_c1)) / (r - l_c1) / (r - l_c1)
                        + exp(Scalar(1.0) / (r - l_c1)))
                       / (r - l_min) / (r - l_min));
            }
        if (k_b != Scalar(0.0))
            {
            // Check if bond length restriction is violated
            if (rsq >= l_max * l_max)
                return false;
            if (rsq <= l_min * l_min)
                return false;

            force_divr = (F_att + F_rep) / r;
            bond_eng = U_att + U_rep;
            }

        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("tether");
        }
#endif

    protected:
    Scalar rsq;   //!< Stored rsq from the constructor
    Scalar k_b;   //!< k_b parameter
    Scalar l_min; //!< l_min parameter
    Scalar l_c1;  //!< l_c1 parameter
    Scalar l_c0;  //!< l_c0 parameter
    Scalar l_max; //!< l_max parameter
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_TETHER_H__
