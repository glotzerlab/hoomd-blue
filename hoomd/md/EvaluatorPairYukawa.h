// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_YUKAWA_H__
#define __PAIR_EVALUATOR_YUKAWA_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairYukawa.h
    \brief Defines the pair evaluator class for Yukawa potentials
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
//! Class for evaluating the Yukawa pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>Yukawa specifics</b>

    EvaluatorPairYukawa evaluates the function:
    \f[ V_{\mathrm{yukawa}}(r) = \varepsilon \frac{ \exp \left( -\kappa r \right) }{r} \f]
*/
class EvaluatorPairYukawa
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar epsilon;
        Scalar kappa;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type()
            {
            epsilon = 0;
            kappa = 0;
            }

        param_type(pybind11::dict v, bool managed = false)
            {
            epsilon = v["epsilon"].cast<Scalar>();
            kappa = v["kappa"].cast<Scalar>();
            }

        // this constructor facilitates unit testing
        param_type(Scalar eps, Scalar kap, bool managed = false)
            {
            epsilon = eps;
            kappa = kap;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["epsilon"] = epsilon;
            v["kappa"] = kappa;
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
    DEVICE EvaluatorPairYukawa(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.epsilon), kappa(_params.kappa)
        {
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
        if (rsq < rcutsq && epsilon != 0)
            {
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar r2inv = Scalar(1.0) / rsq;

            Scalar exp_val = fast::exp(-kappa * r);

            force_divr = epsilon * exp_val * r2inv * (rinv + kappa);
            pair_eng = epsilon * exp_val * rinv;

            if (energy_shift)
                {
                Scalar rcutinv = fast::rsqrt(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;
                pair_eng -= epsilon * fast::exp(-kappa * rcut) * rcutinv;
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
        return std::string("yukawa");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;     //!< Stored rsq from the constructor
    Scalar rcutsq;  //!< Stored rcutsq from the constructor
    Scalar epsilon; //!< epsilon parameter extracted from the params passed to the constructor
    Scalar kappa;   //!< kappa parameter extracted from the params passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_YUKAWA_H__
