// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_OPP_H__
#define __PAIR_EVALUATOR_OPP_H__

#ifndef __HIPCC__
#include <string>
#endif
 
#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairOPP.h
    \brief Defines the pair evaluator class for OPP potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc DEVICE is __host__ __device__ when included in nvcc and blank when
// included into the host compiler
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
//! Class for evaluating the oscillating pair potential
/*! <b>General Overview</b>

    <b>OPP specifics</b>

    EvaluatorPairOPP evaluates the function:
    \f{equation*}
    V_{\mathrm{OPP}}(r)  = - C_1 r^{\eta_1} + C_2 r^{\eta_2}
                             \cos(k ( r - b) - \phi)
    \f}

*/
class EvaluatorPairOPP
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar C1;
        Scalar C2;
        Scalar eta1;
        Scalar eta2;
        Scalar k;
        Scalar phi;

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
        param_type() : C1(0), C2(0), eta1(0), eta2(0), k(0), phi(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            C1 = v["C1"].cast<Scalar>();
            C2 = v["C2"].cast<Scalar>();
            eta1 = v["eta1"].cast<Scalar>();
            eta2 = v["eta2"].cast<Scalar>();
            k = v["k"].cast<Scalar>();
            phi = v["phi"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["C1"] = C1;
            v["C2"] = C2;
            v["eta1"] = eta1;
            v["eta2"] = eta2;
            v["k"] = k;
            v["phi"] = phi;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairOPP(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! OPP doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Get the number of alchemical parameters
    static const unsigned int num_alchemical_parameters = 6;

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force
     * divided by r.
     *  \param pair_eng Output parameter to write the computed pair energy
     *  \param energy_shift If true, the potential must be shifted so that
     *      V(r) is continuous at the cutoff

     *  \return True if they are evaluated or false if they are not because
     *  we are beyond the cutoff
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        if (rsq < rcutsq)
            {
            // Get quantities need for both energy and force calculation
            Scalar r(fast::sqrt(rsq));
            Scalar eval_sin, eval_cos;
            fast::sincos(params.k * r - params.phi, eval_sin, eval_cos);

            // Compute energy
            Scalar r_eta1_arg(params.C1 * fast::pow(r, -params.eta1));
            Scalar r_to_eta2(fast::pow(r, -params.eta2));
            Scalar r_eta2_arg(params.C2 * r_to_eta2 * eval_cos);
            pair_eng = r_eta1_arg + r_eta2_arg;

            // Compute force
            force_divr = ((r_eta1_arg * params.eta1 + r_eta2_arg * params.eta2) / r
                          + (params.C2 * params.k * r_to_eta2 * eval_sin))
                         / r;

            if (energy_shift)
                {
                Scalar r_cut(fast::sqrt(rcutsq));
                Scalar r_cut_eta1_arg(params.C1 * fast::pow(r_cut, -params.eta1));
                Scalar r_cut_eta2_arg(params.C2 * fast::pow(r_cut, -params.eta2)
                                      * fast::cos(params.k * r_cut - params.phi));
                pair_eng -= r_cut_eta1_arg + r_cut_eta2_arg;
                }

            return true;
            }
        else
            {
            return false;
            }
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

    /** Get the index of an alchemical parameter based on the string name.
     */
    static int getAlchemicalParameterIndex(std::string param_name)
        {
        if (param_name == "C1")
            return 0;
        else if (param_name == "C2")
            return 1;
        else if (param_name == "eta1")
            return 2;
        else if (param_name == "eta2")
            return 3;
        else if (param_name == "k")
            return 4;
        else if (param_name == "phi")
            return 5;
        throw std::runtime_error("Unknown alchemical parameter name.");
        }

    /** Update parameters with alchemical degrees of freedom.

        Interoperate with PotentialPairAlchemical to modify the given potential parameters:
        p -> p * alpha, where p is a parameter.
    */
    DEVICE void updateAlchemyParams(const std::array<Scalar, num_alchemical_parameters>& alphas)
        {
        params.C1 *= alphas[0];
        params.C2 *= alphas[1];
        params.eta1 *= alphas[2];
        params.eta2 *= alphas[3];
        params.k *= alphas[4];
        params.phi *= alphas[5];
        }

    /** Update parameters with alchemical degrees of freedom.

        Interoperate with PotentialPairAlchemical to modify the given potential parameters:
        p -> p * alpha, where p is a parameter.
    */
    DEVICE static param_type
    updateAlchemyParams(const param_type& initial_params,
                        std::array<Scalar, num_alchemical_parameters>& alphas)
        {
        param_type params(initial_params);
        params.C1 *= alphas[0];
        params.C2 *= alphas[1];
        params.eta1 *= alphas[2];
        params.eta2 *= alphas[3];
        params.k *= alphas[4];
        params.phi *= alphas[5];
        return params;
        }

    /** Calculate derivative of the alchemical potential with repsect to alpha.

        Interoperate with PotentialPairAlchemical to compute dU/d alpha.
    */
    DEVICE void
    evalAlchemyDerivatives(std::array<Scalar, num_alchemical_parameters>& alchemical_derivatives,
                           const std::array<Scalar, num_alchemical_parameters>& alphas)
        {
            {
            Scalar r = fast::sqrt(rsq);
            Scalar r_to_eta1 = fast::pow(r, -alphas[2] * params.eta1);
            Scalar r_to_eta2 = fast::pow(r, -alphas[3] * params.eta2);
            Scalar log_r = slow::log(r);

            Scalar eval_sin, eval_cos;
            fast::sincos(alphas[4] * params.k * r - alphas[5] * params.phi, eval_sin, eval_cos);

            alchemical_derivatives[0] = r_to_eta1 * params.C1;
            alchemical_derivatives[1] = r_to_eta2 * eval_cos * params.C2;
            alchemical_derivatives[2] = -alphas[0] * params.C1 * r_to_eta1 * log_r * params.eta1;
            alchemical_derivatives[3]
                = -alphas[1] * params.C2 * r_to_eta2 * log_r * eval_cos * params.eta2;
            alchemical_derivatives[4]
                = -alphas[1] * params.C2 * r_to_eta2 * eval_sin * params.k * r;
            alchemical_derivatives[5] = alphas[1] * params.C2 * r_to_eta2 * eval_sin * params.phi;
            }
        }

    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("opp");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;        /// Stored rsq from the constructor
    Scalar rcutsq;     /// Stored rcutsq from the constructor
    param_type params; /// Stored pair parameters for a given type pair
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_OPP_H__
