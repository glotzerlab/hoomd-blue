// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_TWF_H__
#define __PAIR_EVALUATOR_TWF_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include <cmath>

/*! \file EvaluatorPairTWF.h
    \brief Defines the pair potential evaluator for the TWF potential
    \details The potential was designed for simulating globular proteins and is
    a modification of the LJ potential with harder interactions and a variable
    well width. For more information see the Python documentation.
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included
// into the host compiler
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
//! Class for evaluating the TWF pair potential
class EvaluatorPairTWF
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar sigma;
        Scalar alpha;
        Scalar prefactor;

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
        param_type() : sigma(1), alpha(1), prefactor(1) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            sigma = v["sigma"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            prefactor = 4.0 * v["epsilon"].cast<Scalar>() / (alpha * alpha);
            }

        param_type(Scalar sigma, Scalar epsilon, Scalar alpha, bool managed = false)
            : sigma(sigma), alpha(alpha), prefactor(4 * epsilon / (alpha * alpha))
            {
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = sigma;
            v["alpha"] = alpha;
            v["epsilon"] = alpha * alpha / 4 * prefactor;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairTWF(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! TWF doesn't use charge
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
        \param energy_shift If true, the potential must be shifted so that
        V(r) is continuous at the cutoff

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            // Compute common terms to equations
            Scalar sigma2 = params.sigma * params.sigma;
            // If particles are overlapping return maximum possible energy
            // and force since this is an invalid state and should be
            // infinite energy and force.
            if (rsq <= sigma2)
                {
                // Since std::numeric_limit<>::max cannot be used for GPU
                // code, we use the INFINITY macros instead.
                pair_eng = INFINITY;
                force_divr = INFINITY;
                return true;
                }

            Scalar common_term = 1.0 / (rsq / sigma2 - 1.0);
            Scalar common_term3 = common_term * common_term * common_term;
            Scalar common_term6 = common_term3 * common_term3;
            // Compute force and energy
            pair_eng = params.prefactor * (common_term6 - params.alpha * common_term3);
            // The force term is -(dE / dr) * (1 / r).
            Scalar force_term = 6 * common_term / sigma2;
            force_divr
                = params.prefactor * force_term * (2 * common_term6 - params.alpha * common_term3);

            if (energy_shift)
                {
                Scalar common_term_shift = 1.0 / (rcutsq / sigma2 - 1.0);
                Scalar common_term3_shift
                    = common_term_shift * common_term_shift * common_term_shift;
                Scalar common_term6_shift = common_term3_shift * common_term3_shift;
                pair_eng
                    -= params.prefactor * (common_term6_shift - params.alpha * common_term3_shift);
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
        return std::string("twf");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;        //!< Stored rsq from the constructor
    Scalar rcutsq;     //!< Stored rcutsq from the constructor
    param_type params; //!< parameters passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd
#endif // __PAIR_EVALUATOR_TWF_H__
