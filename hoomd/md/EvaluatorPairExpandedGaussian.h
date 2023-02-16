// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_EXPANDEDGAUSSIAN_H__
#define __PAIR_EVALUATOR_EXPANDEDGAUSSIAN_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairShiftedGauss.h
    \brief Defines the pair evaluator class for shifted Gaussian potentials
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
//! Class for evaluating the expanded Gaussian pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>Gauss specifics</b>

    EvaluatorPairExpandedGaussian evaluates the function:
    \f[ V_{\mathrm{expanded_gauss}}(r) = \varepsilon \exp \left[ -
    \frac{1}{2}\left( \frac{r-r_{0}}{\sigma}\right)^2 \right] \f]

    The expanded Gaussian potential does not need diameter or charge. Three
    parameters are specified and stored in a Scalar3.
    \a epsilon is placed in \a params.x, \a sigma is in \a params.y, and
    \a r_{0} is in \a params.z.

    \a epsilon and \a sigma are related to the standard lj parameters sigma and
    epsilon by:
    - \a epsilon = \f$ \varepsilon \f$
    - \a sigma = \f$ \sigma \f$

    \a r_0 is the shifted distance of the gaussian potential.

*/
class EvaluatorPairExpandedGaussian
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar epsilon;
        Scalar sigma;
        Scalar r_0;

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
        param_type() : epsilon(0), sigma(0), r_0(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            sigma = v["sigma"].cast<Scalar>();
            epsilon = v["epsilon"].cast<Scalar>();
            r_0 = v["r_0"].cast<Scalar>();
            }

        // used to facilitate unit testing
        param_type(Scalar eps, Scalar sig, Scalar r, bool managed = false)
            {
            sigma = sig;
            epsilon = eps;
            r_0 = r;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = sigma;
            v["epsilon"] = epsilon;
            v["r_0"] = r_0;
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
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairShiftedGauss(Scalar _rsq, Scalar _rcutsq,
                                     const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.epsilon),
          sigma(_params.sigma), r_0(_params.r_0)
        {
        }

    //! Gauss doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! Gauss doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional diameter values
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
            Scalar sigma_sq = sigma * sigma;
            Scalar rinv = fast::rsqrt(rsq);

            Scalar r_over_sigma_sq = (r - r_0) * (r - r_0) / sigma_sq;
            Scalar exp_val = fast::exp(-Scalar(1.0) / Scalar(2.0) * r_over_sigma_sq);

            force_divr = epsilon / sigma_sq * exp_val * (Scalar(1.0) - r_0 / r);
            pair_eng = epsilon * exp_val;

            if (energy_shift)
                {
                pair_eng -= epsilon * fast::exp(-Scalar(1.0) / Scalar(2.0) * rcutsq / sigma_sq);
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
        return std::string("shiftedgauss");
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
    Scalar sigma;   //!< sigma parameter extracted from the params passed to the constructor
    Scalar r_0;     //!< r_0 parameter extracted from the params passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_SHIFTEDGAUSS_H__
