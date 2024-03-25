// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_HARMONIC_H__
#define __BOND_EVALUATOR_HARMONIC_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorBondHarmonic.h
    \brief Defines the bond evaluator class for harmonic potentials
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
struct harmonic_params
    {
    Scalar k;
    Scalar r_0;

#ifndef __HIPCC__
    harmonic_params()
        {
        k = 0;
        r_0 = 0;
        }

    harmonic_params(Scalar k, Scalar r_0) : k(k), r_0(r_0) { }

    harmonic_params(pybind11::dict v)
        {
        k = v["k"].cast<Scalar>();
        r_0 = v["r0"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["r0"] = r_0;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Class for evaluating the harmonic bond potential
/*! Evaluates the harmonic bond potential in an identical manner to EvaluatorPairLJ for pair
   potentials. See that class for a full motivation and design specifics.

    params.x is the K stiffness parameter, and params.y is the r_0 equilibrium rest length.
*/
class EvaluatorBondHarmonic
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    typedef harmonic_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorBondHarmonic(Scalar _rsq, const param_type& _params)
        : rsq(_rsq), K(_params.k), r_0(_params.r_0)
        {
        }

    //! Harmonic doesn't use charge
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
        force_divr = K * (r_0 / r - Scalar(1.0));

// if the result is not finite, it is likely because of a division by 0, setting force_divr to 0
// will correctly result in a 0 force in this case
#ifdef __HIPCC__
        if (!isfinite(force_divr))
#else
        if (!std::isfinite(force_divr))
#endif
            {
            force_divr = Scalar(0);
            }
        bond_eng = Scalar(0.5) * K * (r_0 - r) * (r_0 - r);

        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("harmonic");
        }
#endif

    protected:
    Scalar rsq; //!< Stored rsq from the constructor
    Scalar K;   //!< K parameter
    Scalar r_0; //!< r_0 parameter
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_HARMONIC_H__
