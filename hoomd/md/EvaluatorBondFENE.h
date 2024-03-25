// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_FENE_H__
#define __BOND_EVALUATOR_FENE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorBondFENE.h
    \brief Defines the bond evaluator class for FENE potentials
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
struct fene_params
    {
    Scalar k;
    Scalar r_0;
    Scalar epsilon_x_4;
    Scalar sigma_6;
    Scalar delta;

#ifndef __HIPCC__
    fene_params()
        {
        k = 0;
        r_0 = 0;
        epsilon_x_4 = 0;
        sigma_6 = 0;
        }

    fene_params(pybind11::dict v)
        {
        k = v["k"].cast<Scalar>();
        r_0 = v["r0"].cast<Scalar>();
        delta = v["delta"].cast<Scalar>();
        Scalar epsilon = v["epsilon"].cast<Scalar>();
        Scalar sigma = v["sigma"].cast<Scalar>();
        sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma;
        epsilon_x_4 = Scalar(4.0) * epsilon;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["r0"] = r_0;
        v["sigma"] = pow(sigma_6, 1. / 6.);
        v["epsilon"] = epsilon_x_4 / 4.0;
        v["delta"] = delta;
        return v;
        }
#endif
    } __attribute__((aligned(16)));

//! Class for evaluating the FENE bond potential
/*! The parameters are:
    - \a K (params.x) Stiffness parameter for the force computation
    - \a r_0 (params.y) maximum bond length for the force computation
    - \a lj1 (params.z) Value of lj1 = 4.0*epsilon*pow(sigma,12.0)
       of the WCA potential in the force calculation
    - \a lj2 (params.w) Value of lj2 = 4.0*epsilon*pow(sigma,6.0)
       of the WCA potential in the force calculation
*/
class EvaluatorBondFENE
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    typedef fene_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorBondFENE(Scalar _rsq, const param_type& _params)
        : rsq(_rsq), K(_params.k), r_0(_params.r_0),
          lj1(_params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6),
          lj2(_params.epsilon_x_4 * _params.sigma_6), delta(_params.delta)
        {
        }

    //! FENE  doesn't use charge
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
        Scalar rmdoverr = Scalar(1.0);

        // Correct the rsq for particles that are not unit in size.
        Scalar rtemp = sqrt(rsq) - delta;
        rmdoverr = rtemp / sqrt(rsq);
        rsq = rtemp * rtemp;

        // compute the force magnitude/r in forcemag_divr (FLOPS: 9)
        Scalar r2inv = Scalar(1.0) / rsq;
        Scalar r6inv = r2inv * r2inv * r2inv;

        Scalar WCAforcemag_divr = Scalar(0.0);
        Scalar pair_eng = Scalar(0.0);

        Scalar sigma6inv = lj2 / lj1;
        Scalar epsilon = lj2 * lj2 / Scalar(4.0) / lj1;

        if (lj1 != 0 && r6inv > sigma6inv / Scalar(2.0)) // wcalimit 2^(1/6))^6 sigma^6
            {
            WCAforcemag_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);
            pair_eng = (r6inv * (lj1 * r6inv - lj2) + epsilon);
            }

        force_divr = WCAforcemag_divr * rmdoverr;
        bond_eng = pair_eng;

        // need to check that K is nonzero, to avoid division by zero
        if (K != Scalar(0.0))
            {
            // Check if bond length restriction is violated
            if (rsq >= r_0 * r_0)
                return false;

            force_divr += -K / (Scalar(1.0) - rsq / (r_0 * r_0)) * rmdoverr;
            bond_eng += -Scalar(0.5) * K * (r_0 * r_0) * log(Scalar(1.0) - rsq / (r_0 * r_0));
            }

        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("fene");
        }
#endif

    protected:
    Scalar rsq;   //!< Stored rsq from the constructor
    Scalar K;     //!< K parameter
    Scalar r_0;   //!< r_0 parameter
    Scalar lj1;   //!< lj1 parameter
    Scalar lj2;   //!< lj2 parameter
    Scalar delta; //!< Radial shift
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_FENE_H__
