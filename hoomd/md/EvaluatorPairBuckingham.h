// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_BUCKINGHAM_H__
#define __PAIR_EVALUATOR_BUCKINGHAM_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairBuckingham.h
    \brief Defines the pair evaluator class for Buckingham potentials
    \details Buckingham potential uses an exp() based repulsive contribution
        paired with an exp-6 attractive term.
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
//! Class for evaluating the Buckingham pair potential
/*! <b>General Overview</b>

    EvaluatorPairBuckingham is a low level computation class that computes the Buckingham pair
   potential V(r).

    <b>Buckingham specifics</b>

    EvaluatorPairBuckingham evaluates the function:
    \f[ V_{\mathrm{Buckingham}}(r) = A \exp\left( \frac{-r}{\rho} \right) -
                                            \frac{C}{r^{6}} \f]
   Split into:
   \f[ V_{\mathrm{Buckingham}}(r) = Exp_factor - \frac{C}{r^6} \f]

   Similarly,
    \f[ -\frac{1}{r} \frac{\partial V_{\mathrm{Buckingham}}}{\partial r} = \frac{Exp_factor}{\rho
   \cdot r} - \frac{6 \cdot C}{r^{8}} \f]

    The Buckingham potential does not need charge. Three parameters are specified and
   stored in a Scalar4. \a A is placed in \a params.x, \a rho is in \a params.y and \a C is placed
   in \a params.z.

    These are the standard Buckingham parameters:
    - \a A = the exponent pre-factor: A * exp(-r / rho);
    - \a rho = the well width in the exponent: exp(-r / rho);
    - \a C = the attractive contribution: - C / r^6;

*/
class EvaluatorPairBuckingham
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar A;
        Scalar rho;
        Scalar C;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! set CUDA memory hint
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type() : A(0), rho(0), C(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            A = v["A"].cast<Scalar>();
            rho = v["rho"].cast<Scalar>();
            C = v["C"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["A"] = A;
            v["rho"] = rho;
            v["C"] = C;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairBuckingham(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), A(_params.A), rho(_params.rho), C(_params.C)
        {
        }

    //! Buckingham doesn't use charge
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
        if (rsq < rcutsq && rho > Scalar(0.0))
            {
            Scalar r = fast::sqrt(rsq);
            Scalar Exp_factor = A * fast::exp(-r / rho);

            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r6inv = r2inv * r2inv * r2inv;

            force_divr = (Exp_factor / (rho * r)) - (r2inv * r6inv * Scalar(6.0) * C);

            pair_eng = Exp_factor - r6inv * C;

            if (energy_shift)
                {
                Scalar rcut = fast::sqrt(rcutsq);
                Scalar Exp_factor_cut = A * fast::exp(-rcut / rho);

                Scalar rcut2inv = Scalar(1.0) / rcutsq;
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                pair_eng -= Exp_factor_cut - rcut6inv * C;
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
        return std::string("buck");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar A;      //!< Buckingham parameter extracted from the params passed to the constructor
    Scalar rho;    //!< Buckingham parameter extracted from the params passed to the constructor
    Scalar C;      //!< Buckingham parameter extracted from the params passed to the constructor
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_BUCKINGHAM_H__
