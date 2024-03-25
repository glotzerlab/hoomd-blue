// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_LJ1208_H__
#define __PAIR_EVALUATOR_LJ1208_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairLJ1208.h
    \brief Defines the pair evaluator class for LJ 12-8 potentials
    \details A modified Lennard-Jones potential where the attractive portion
        has a power of 8 rather than the normal power of 6.
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
//! Class for evaluating the LJ-12-8 pair potential
/*! <b>General Overview</b>

    EvaluatorPairLJ1208 is a low level computation class that computes the LJ-12-8 pair potential
   V(r).

    <b>LJ 12-8 specifics</b>

    EvaluatorPairLJ1208 evaluates the function:
    \f[ V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                                            \left( \frac{\sigma}{r} \right)^{8} \right] \f]
    broken up as follows for efficiency
    \f[ V_{\mathrm{LJ}}(r) = r^{-8} \cdot \left( 4 \varepsilon \sigma^{12} \cdot r^{-4} -
                                            4 \varepsilon \sigma^{8} \right) \f]
    . Similarly,
    \f[ -\frac{1}{r} \frac{\partial V_{\mathrm{LJ}}}{\partial r} = r^{-2} \cdot r^{-8} \cdot
            \left( 12 \cdot 4 \varepsilon \sigma^{12} \cdot r^{-4} - 8 \cdot 4 \varepsilon
   \sigma^{8} \right) \f]
*/
class EvaluatorPairLJ1208
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar sigma_4;
        Scalar epsilon_x_4;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifndef ENABLE_HIP
        //! set CUDA memory hints
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type() : sigma_4(0), epsilon_x_4(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            auto sigma(v["sigma"].cast<Scalar>());
            auto epsilon(v["epsilon"].cast<Scalar>());

            sigma_4 = sigma * sigma * sigma * sigma;
            epsilon_x_4 = Scalar(4.0) * epsilon;

            // parameters used in implementation
            // lj1 = 4.0 * epsilon * pow(sigma, Scalar(12.0));
            // -> lj1 = epsilon_x_4 * sigma_4 * sigma_4 * sigma_4
            // lj2 = 4.0 * epsilon * pow(sigma, Scalar(8.0));
            // -> lj2 = epsilon_x_4 * sigma_4 * sigma_4
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = pow(sigma_4, 1. / 4.);
            v["epsilon"] = epsilon_x_4 / 4.0;
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
    DEVICE EvaluatorPairLJ1208(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq),
          lj1(_params.epsilon_x_4 * _params.sigma_4 * _params.sigma_4 * _params.sigma_4),
          lj2(_params.epsilon_x_4 * _params.sigma_4 * _params.sigma_4)
        {
        }

    //! LJ doesn't use charge
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
        if (rsq < rcutsq && lj1 != 0)
            {
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r4inv = r2inv * r2inv;
            Scalar r8inv = r4inv * r4inv;

            force_divr = r2inv * r8inv * (Scalar(12.0) * lj1 * r4inv - Scalar(8.0) * lj2);

            pair_eng = r8inv * (lj1 * r4inv - lj2);

            if (energy_shift)
                {
                Scalar rcut2inv = Scalar(1.0) / rcutsq;
                Scalar rcut4inv = rcut2inv * rcut2inv;
                Scalar rcut8inv = rcut4inv * rcut4inv;
                pair_eng -= rcut8inv * (lj1 * rcut4inv - lj2);
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
        return std::string("lj1208");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar lj1;    //!< lj1 parameter extracted from the params passed to the constructor
    Scalar lj2;    //!< lj2 parameter extracted from the params passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_LJ1208_H__
