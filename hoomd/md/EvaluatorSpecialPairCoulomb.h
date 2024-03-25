// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_COULOMB_H__
#define __BOND_EVALUATOR_COULOMB_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorSpecialPairCoulomb.h
    \brief Defines the bond evaluator class for Coulomb interactions

    This is designed to be used e.g. for the scaled 1-4 interaction in all-atom force fields such as
   GROMOS or OPLS. It implements Coulombs law, as both LJ and Coulomb interactions are scaled by the
   same amount in the OPLS force field for 1-4 interactions. Thus it is intended to be used along
   with special_pair.LJ.
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
struct special_coulomb_params
    {
    Scalar alpha;
    Scalar r_cutsq;

#ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        // default implementation does nothing
        }
#endif

#ifndef __HIPCC__
    special_coulomb_params() : alpha(0.), r_cutsq(0.) { }

    special_coulomb_params(pybind11::dict v)
        {
        alpha = v["alpha"].cast<Scalar>();
        r_cutsq = 0.;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["alpha"] = alpha;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Class for evaluating the Coulomb bond potential
/*! See the EvaluatorPairLJ class for the meaning of the parameters
 */
class EvaluatorSpecialPairCoulomb
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    typedef special_coulomb_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorSpecialPairCoulomb(Scalar _rsq, const param_type& _params)
        : rsq(_rsq), scale(_params.alpha), rcutsq(_params.r_cutsq)
        {
        }

    //! Coulomb use charge
    DEVICE static bool needsCharge()
        {
        return true;
        }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
        qiqj = qi * qj;
        }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param bond_eng Output parameter to write the computed bond energy

        \return True if they are evaluated or false if the bond
                energy is not defined. Based on EvaluatorSpecialPairLJ which
                returns true regardless whether it is evaluated or not.
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && qiqj != 0)
            {
            Scalar r1inv = Scalar(1.0) / fast::sqrt(rsq);
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r3inv = r2inv * r1inv;

            Scalar scaledQ = qiqj * scale;

            force_divr = scaledQ * r3inv;
            bond_eng = scaledQ * r1inv;
            }
        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("coul");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar qiqj;   //!< product of charges from setCharge(qa, qb)
    Scalar scale;  //!< scaling factor to apply to Coulomb interaction
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_COULOMB_H__
