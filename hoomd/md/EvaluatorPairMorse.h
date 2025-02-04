// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_MORSE_H__
#define __PAIR_EVALUATOR_MORSE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairMorse.h
    \brief Defines the pair evaluator class for Morse potential
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
//! Class for evaluating the Morse pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Morse specifics</b>

    EvaluatorPairMorse evaluates the function:
    \f[ V_{\mathrm{Morse}}(r) = D_0 \left[ \exp \left(-2\alpha \left(r - r_0\right) \right)
                                           -2\exp \left(-\alpha \left(r-r_0\right) \right)  \right]
   \f]
*/
class EvaluatorPairMorse
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar D0;
        Scalar alpha;
        Scalar r0;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        // CUDA memory hints
        void set_memory_hints() const { }
#endif

#ifndef __HIPCC__
        param_type() : D0(0), alpha(0), r0(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            D0 = v["D0"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            r0 = v["r0"].cast<Scalar>();
            }

        param_type(Scalar d, Scalar a, Scalar r, bool managed = false)
            {
            D0 = d;
            alpha = a;
            r0 = r;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["D0"] = D0;
            v["alpha"] = alpha;
            v["r0"] = r0;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairMorse(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), D0(_params.D0), alpha(_params.alpha), r0(_params.r0)
        {
        }

    //! Morse doesn't use charge
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
        if (rsq < rcutsq)
            {
            Scalar r = fast::sqrt(rsq);
            Scalar Exp_factor = fast::exp(-alpha * (r - r0));

            pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
            force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r;

            if (energy_shift)
                {
                Scalar rcut = fast::sqrt(rcutsq);
                Scalar Exp_factor_cut = fast::exp(-alpha * (rcut - r0));
                pair_eng -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));
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
        return std::string("morse");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar D0;     //!< Depth of the Morse potential at its minimum
    Scalar alpha;  //!< Controls width of the potential well
    Scalar r0;     //!< Offset, i.e., position of the potential minimum
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_MORSE_H__
