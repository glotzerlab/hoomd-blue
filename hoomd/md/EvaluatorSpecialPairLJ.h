// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_LJ_H__
#define __BOND_EVALUATOR_LJ_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorSpecialPairLJ.h
    \brief Defines the bond evaluator class for LJ interactions

    The LJ bond represents a means of injecting specified pairs of particles
    into the force computation, e.g. based on topology. This is designed
    to be used e.g. for the scaled 1-4 interaction in all-atom force fields
    such as GROMOS or OPLS.
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
struct special_lj_params
    {
    Scalar sigma_6;
    Scalar epsilon_x_4;
    Scalar r_cutsq;

#ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        // default implementation does nothing
        }
#endif

#ifndef __HIPCC__
    special_lj_params() : sigma_6(0.), epsilon_x_4(0.), r_cutsq(0.) { }

    special_lj_params(pybind11::dict v)
        {
        auto sigma(v["sigma"].cast<Scalar>());
        auto epsilon(v["epsilon"].cast<Scalar>());
        sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma;
        epsilon_x_4 = Scalar(4.0) * epsilon;

        // r_cutsq is set later in PotentialSpecialPair<evaluator>::setRCut
        r_cutsq = 0.;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["sigma"] = pow(sigma_6, 1. / 6.);
        v["epsilon"] = epsilon_x_4 / 4.0;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the LJ bond potential
/*! See the EvaluatorPairLJ class for the meaning of the parameters
 */
class EvaluatorSpecialPairLJ
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    typedef special_lj_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorSpecialPairLJ(Scalar _rsq, const param_type& _params)
        : rsq(_rsq), lj1(_params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6),
          lj2(_params.epsilon_x_4 * _params.sigma_6), rcutsq(_params.r_cutsq)
        {
        }

    //! LJ doesn't use charge
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
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && lj1 != 0)
            {
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r6inv = r2inv * r2inv * r2inv;
            force_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);

            bond_eng = r6inv * (lj1 * r6inv - lj2);
            }
        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("lj");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar lj1;    //!< lj1 parameter extracted from the params passed to the constructor
    Scalar lj2;    //!< lj2 parameter extracted from the params passed to the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_LJ_H__
