// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __EVALUATOR_TERSOFF__
#define __EVALUATOR_TERSOFF__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorTersoff.h
    \brief Defines the evaluator class for the three-body Tersoff potential
*/

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

//! Parameter type for this potential
struct tersoff_params
    {
    Scalar cutoff_thickness; //!< Thickness of the cutoff shell (2D)
    Scalar2 coeffs; //!< Contains the coefficients for the repulsive (x) and attractive (y) terms
    Scalar2 exp_consts; //!< Gives the coefficients in the exponential functions for the repulsive (x) and attractive (y) terms
    Scalar dimer_r; //!< Dimer separation of the type-pair
    Scalar tersoff_n; //!< \a n in Tersoff potential
    Scalar gamman; //!< \a gamma raised to the \a n power in the Tersoff potential
    Scalar lambda_cube; //!< \a lambda^3 in the exponential term of \a chi
    Scalar3 ang_consts; //!< Constants \a c^2, \a d^2, and \a m in the bond-angle function of the Tersoff potential
    Scalar alpha; //!< \a alpha in the exponential cutoff-smoothing function
};

//! Function to make the parameter type
HOSTDEVICE inline tersoff_params make_tersoff_params(Scalar cutoff_thickness,
                                                     Scalar2 coeffs,
                                                     Scalar2 exp_consts,
                                                     Scalar dimer_r,
                                                     Scalar tersoff_n,
                                                     Scalar gamman,
                                                     Scalar lambda_cube,
                                                     Scalar3 ang_consts,
                                                     Scalar alpha)
    {
    tersoff_params retval;
    retval.cutoff_thickness = cutoff_thickness;
    retval.coeffs = coeffs;
    retval.exp_consts = exp_consts;
    retval.dimer_r = dimer_r;
    retval.tersoff_n = tersoff_n;
    retval.gamman = gamman;
    retval.lambda_cube = lambda_cube;
    retval.ang_consts = ang_consts;
    retval.alpha = alpha;
    return retval;
    }

//! Class for evaluating the Tersoff three-body potential
class EvaluatorTersoff
    {
    public:
        //! Define the parameter type used by this evaluator
        typedef tersoff_params param_type;

        //! Constructs the evaluator
        /*! \param _rij_sq Squared distance between particles i and j
            \param _rcutsq Squared distance at which the potential goes to zero
            \param _params Per type-pair parameters for this potential
        */
        DEVICE EvaluatorTersoff(Scalar _rij_sq, Scalar _rcutsq, const param_type& _params)
            : rij_sq(_rij_sq), rcutsq(_rcutsq), cutoff_shell_thickness(_params.cutoff_thickness),
              tersoff_A(_params.coeffs.x), tersoff_B(_params.coeffs.y), lambda_R(_params.exp_consts.x), lambda_A(_params.exp_consts.y),
              dimer_separation(_params.dimer_r), tersoff_n(_params.tersoff_n), gamman(_params.gamman), lambda_h3(_params.lambda_cube),
              tersoff_c2(_params.ang_consts.x), tersoff_d2(_params.ang_consts.y), tersoff_m(_params.ang_consts.z),
              cutoff_alpha(_params.alpha)
            {
            }

        //! Set the square distance between particles i and j
        DEVICE void setRij(Scalar rsq)
            {
            rij_sq = rsq;
            }

        //! Set the square distance between particles i and k
        DEVICE void setRik(Scalar rsq)
            {
            rik_sq = rsq;
            }

        //! This is a pure pair potential
        DEVICE static bool hasPerParticleEnergy() { return false; }

        //! We need chi
        DEVICE static bool needsChi() { return true; }

        //! We have ik-forces
        DEVICE static bool hasIkForce() { return true; }

        //! The Tersoff potential needs the bond angle
        DEVICE static bool needsAngle() { return true; }

        //! Set the bond angle value
        //! \param _cos_th Cosine of the angle between ij and ik
        DEVICE void setAngle(Scalar _cos_th)
            {
            cos_th = _cos_th;
            }

        //! Check whether a pair of particles is interactive
        DEVICE bool areInteractive()
            {
            if (tersoff_A > Scalar(0.0) || tersoff_B > Scalar(0.0))
                return true;
            else return false;
            }

        //! Evaluate the repulsive and attractive terms of the force
        DEVICE bool evalRepulsiveAndAttractive(Scalar& fR, Scalar& fA)
            {
            if (rij_sq < rcutsq && (tersoff_A > Scalar(0.0) || tersoff_B > Scalar(0.0)))
                {
                // compute rij
                Scalar rij = fast::sqrt(rij_sq);

                // compute the repulsive potential
                fR = tersoff_A * fast::exp( lambda_R * (dimer_separation - rij) );

                // compute the attractive potential
                fA = tersoff_B * fast::exp( lambda_A * (dimer_separation - rij) );

                return true;
                }
            else return false;
            }

        //! Evaluate chi for this triplet
        DEVICE void evalChi(Scalar& chi)
            {
            if (rik_sq < rcutsq && gamman != 0)
                {
                // compute rij, rik, rcut, and r_shell_inner
                Scalar rij = fast::sqrt(rij_sq);
                Scalar rik = fast::sqrt(rik_sq);
                Scalar rcut = fast::sqrt(rcutsq);
                Scalar r_shell_inner = rcut - cutoff_shell_thickness;

                // compute the rik cutoff function
                Scalar fcut_ik = Scalar(1.0);
                if (rik > r_shell_inner)
                    {
                    Scalar cutoff_x = (rik - r_shell_inner) / cutoff_shell_thickness;
                    Scalar cutoff_x2 = cutoff_x * cutoff_x;
                    Scalar cutoff_x3 = cutoff_x2 * cutoff_x;
                    Scalar inv_denom = Scalar(1.0) / (cutoff_x3 - Scalar(1.0));

                    fcut_ik = fast::exp( cutoff_alpha * cutoff_x3 * inv_denom);

//                    Scalar r_shell_mid = rcut - Scalar(0.5) * cutoff_shell_thickness;
//                    Scalar cutoff_x = Scalar(M_PI) * (rik - r_shell_mid)
//                        / cutoff_shell_thickness;
//
//                    fcut_ik = Scalar(0.5) - Scalar(0.5) * fast::sin(cutoff_x);
                    }

                // compute the h function
                Scalar delta_r = rij - rik;
                Scalar delta_r3 = delta_r * delta_r * delta_r;
                Scalar h = fast::exp( lambda_h3 * delta_r3 );

                // compute the g function
                Scalar ang_difference = tersoff_m - cos_th;
                Scalar gdenom = tersoff_d2 + ang_difference * ang_difference;
                Scalar g = Scalar(1.0) + tersoff_c2 / tersoff_d2 - tersoff_c2 / gdenom;

                chi += fcut_ik * g * h;
                }
            }

        //! We don't have a scalar ij contribution
        DEVICE void evalPhi(Scalar& chi)
            { }

        //! Evaluate the force and potential energy due to ij interactions
        DEVICE void evalForceij(Scalar fR,
                                Scalar fA,
                                Scalar chi,
                                Scalar phi,
                                Scalar& bij,
                                Scalar& force_divr,
                                Scalar& potential_eng)
            {
            // compute rij, rcut, and r_shell_inner
            Scalar rij = fast::sqrt(rij_sq);
            Scalar rcut = fast::sqrt(rcutsq);
            Scalar r_shell_inner = rcut - cutoff_shell_thickness;

            // compute the ij cutoff function and its derivative
            Scalar fcut_ij = Scalar(1.0);
            Scalar dfcut_ij = Scalar(0.0);
            if (rij > r_shell_inner)
                {
                Scalar cutoff_x = (rij - r_shell_inner) / cutoff_shell_thickness;
                Scalar cutoff_x2 = cutoff_x * cutoff_x;
                Scalar cutoff_x3 = cutoff_x2 * cutoff_x;
                Scalar inv_denom = Scalar(1.0) / (cutoff_x3 - Scalar(1.0));

                fcut_ij = fast::exp( cutoff_alpha * cutoff_x3 * inv_denom );
                dfcut_ij = Scalar(-3.0) * cutoff_alpha * cutoff_x2 * inv_denom * inv_denom
                    / cutoff_shell_thickness * fcut_ij;

//                Scalar cutoff_x = Scalar(M_PI) * (rij - r_shell_mid)
//                    / cutoff_shell_thickness;
//
//                fcut_ij = Scalar(0.5) - Scalar(0.5) * fast::sin(cutoff_x);
//                dfcut_ij = Scalar(-M_PI / 2.0) / cutoff_shell_thickness
//                    * fast::cos(cutoff_x);
                }

            // compute the derivative of the base repulsive and attractive terms
            Scalar dfR = Scalar(-1.0) * lambda_R * fR;
            Scalar dfA = Scalar(-1.0) * lambda_A * fA;

            // compute chi^n and (1 + gamma^n * chi^n)
            Scalar chin = fast::pow(chi, tersoff_n);
            Scalar sum_gamma_chi = Scalar(1.0) + gamman * chin;

            // compute bij
            bij = fast::pow( sum_gamma_chi, Scalar(-0.5) / tersoff_n );

            // compute the ij force
            force_divr = Scalar(-0.5)
                * ( dfcut_ij * ( fR - bij * fA ) + fcut_ij * ( dfR - bij * dfA ) ) / rij;

            // compute the potential energy
            potential_eng = Scalar(0.5) * fcut_ij * (fR - bij * fA);
            }

        DEVICE void evalSelfEnergy(Scalar& energy, Scalar phi)
            { }

        //! Evaluate the forces due to ijk interactions
        DEVICE bool evalForceik(Scalar fR,
                                Scalar fA,
                                Scalar chi,
                                Scalar bij,
                                Scalar3& force_divr_ij,
                                Scalar3& force_divr_ik)
            {
            if (rik_sq < rcutsq && chi != Scalar(0.0))
                {
                // compute rij, rik, rcut, and r_shell_inner
                Scalar rij = fast::sqrt(rij_sq);
                Scalar rik = fast::sqrt(rik_sq);
                Scalar rcut = fast::sqrt(rcutsq);
                Scalar r_shell_inner = rcut - cutoff_shell_thickness;
                // compute the dot product of rij and rik
//                Scalar rdot = cos_th * rij * rik;

                // compute the ij cutoff function
                Scalar fcut_ij = Scalar(1.0);
                if (rij > r_shell_inner)
                    {
                    Scalar cutoff_x = (rij - r_shell_inner) / cutoff_shell_thickness;
                    Scalar cutoff_x2 = cutoff_x * cutoff_x;
                    Scalar cutoff_x3 = cutoff_x2 * cutoff_x;
                    Scalar inv_denom = Scalar(1.0) / (cutoff_x3 - Scalar(1.0));

                    fcut_ij = fast::exp( cutoff_alpha * cutoff_x3 * inv_denom );

//                    Scalar r_shell_mid = rcut - Scalar(0.5) * cutoff_shell_thickness;
//                    Scalar cutoff_x = Scalar(M_PI) * (rij - r_shell_mid)
//                        / cutoff_shell_thickness;
//
//                    fcut_ij = Scalar(0.5) - Scalar(0.5) * fast::sin(cutoff_x);
                    }

                // compute the ik cutoff function and its derivative
                Scalar fcut_ik = Scalar(1.0);
                Scalar dfcut_ik = Scalar(0.0);
                if (rik > r_shell_inner)
                    {
                    Scalar cutoff_x = (rik - r_shell_inner) / cutoff_shell_thickness;
                    Scalar cutoff_x2 = cutoff_x * cutoff_x;
                    Scalar cutoff_x3 = cutoff_x2 * cutoff_x;
                    Scalar inv_denom = Scalar(1.0) / (cutoff_x3 - Scalar(1.0));

                    fcut_ik = fast::exp( cutoff_alpha * cutoff_x3 * inv_denom );
                    dfcut_ik = Scalar(-3.0) * cutoff_alpha * cutoff_x2 * inv_denom * inv_denom
                        / cutoff_shell_thickness * fcut_ik;

//                    Scalar r_shell_mid = rcut - Scalar(0.5) * cutoff_shell_thickness;
//                    Scalar cutoff_x = Scalar(M_PI) * (rik - r_shell_mid)
//                        / cutoff_shell_thickness;
//
//                    fcut_ik = Scalar(0.5) - Scalar(0.5) * fast::sin(cutoff_x);
//                    dfcut_ik = Scalar(-M_PI) / (Scalar(2.0) * cutoff_shell_thickness)
//                        * fast::cos(cutoff_x);
                    }

                // h function and its derivatives
                Scalar delta_r = rij - rik;
                Scalar delta_r2 = delta_r * delta_r;
                Scalar delta_r3 = delta_r2 * delta_r;
                Scalar h = fast::exp( lambda_h3 * delta_r3 );
                Scalar dhj = Scalar(3.0) * lambda_h3 * delta_r2 * h;
                Scalar dhk = -dhj;

                // g function and its derivatives
                Scalar ang_diff = tersoff_m - cos_th;
                Scalar gdenom = tersoff_d2 + ang_diff * ang_diff;
                Scalar g = Scalar(1.0) + tersoff_c2 / tersoff_d2 - tersoff_c2 / gdenom;
                Scalar dg = Scalar(-2.0) * tersoff_c2 / (gdenom * gdenom) * ang_diff;
                Scalar dg_ij_i = dg * ( Scalar(1.0) / rik - cos_th / rij );
                Scalar dg_ik_i = dg * ( Scalar(1.0) / rij - cos_th / rik );
                Scalar dg_ij_j = dg * ( cos_th / rij );
                Scalar dg_ik_j = dg * (Scalar(-1.0) / rij);
                Scalar dg_ij_k = dg * (Scalar(-1.0) / rik);
                Scalar dg_ik_k = dg * ( cos_th / rik );

                // derivatives of chi
                Scalar dchi_ij_i = fcut_ik * dg_ij_i * h + fcut_ik * g * dhj;
                Scalar dchi_ik_i = dfcut_ik * g * h + fcut_ik * dg_ik_i * h + fcut_ik * g * dhk;

                Scalar dchi_ij_j = fcut_ik * dg_ij_j * h - fcut_ik * g * dhj;
                Scalar dchi_ik_j = fcut_ik * dg_ik_j * h;

                Scalar dchi_ij_k = fcut_ik * dg_ij_k * h;
                Scalar dchi_ik_k = -dfcut_ik * g * h + fcut_ik * dg_ik_k * h - fcut_ik * g * dhk;

                // derivative of bij
                Scalar chin = fast::pow( chi, tersoff_n );
                Scalar sum_gamma_chi = Scalar(1.0) + gamman * chin;
                Scalar dbij = Scalar(-0.5) * fast::pow( chi, tersoff_n - Scalar(1.0) )
                    * gamman * fast::pow( sum_gamma_chi, Scalar(-0.5) / tersoff_n - Scalar(1.0) );

                // compute the forces and energy
                Scalar F = Scalar(0.5) * fcut_ij * dbij * fA;
                // assign the ij forces
                force_divr_ij.x = F * dchi_ij_i / rij;
                force_divr_ij.y = F * dchi_ij_j / rij;
                force_divr_ij.z = F * dchi_ij_k / rij;
                // assign the ik forces
                force_divr_ik.x = F * dchi_ik_i / rik;
                force_divr_ik.y = F * dchi_ik_j / rik;
                force_divr_ik.z = F * dchi_ik_k / rik;

                return true;
                }
            else return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name.  Must be short and all lowercase, as this is the name
            energies will be logged as via analyze.log.
        */
        static std::string getName()
            {
            return std::string("tersoff");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rij_sq; //!< Stored rij_sq from the constructor
        Scalar rik_sq; //!< Stored rik_sq from the constructor
        Scalar rcutsq; //!< Stored rcutsq from the constructor
        Scalar cos_th; //!< Cosine of the angle between rij and rik
        Scalar cutoff_shell_thickness; //!< Thickness of the cutoff shell in which the cutoff function applies
        Scalar tersoff_A; //!< Coefficient for the repulsive term of the Tersoff potential
        Scalar tersoff_B; //!< Coefficient for the attractive term of the Tersoff potential
        Scalar lambda_R; //!< Exponential coefficient for the repulsive term of the Tersoff potential
        Scalar lambda_A; //!< Exponential coefficient for the attractive term of the Tersoff potential
        Scalar dimer_separation; //!< Dimer separation for the type-pair
        Scalar tersoff_n; //!< \a n parameter for the Tersoff potential
        Scalar gamman; //!< \a gamma^n in the modifier for the attractive term of the Tersoff potential
        Scalar lambda_h3; //!< \a lambda^3 in the h function
        Scalar tersoff_c2; //!< \a c^2 in the \a g(theta) portion of the Tersoff potential
        Scalar tersoff_d2; //!< \a d^2 in the \a g(theta) portion of the Tersoff potential
        Scalar tersoff_m; //!< Cosine of the minimum-energy bonding angle
        Scalar cutoff_alpha; //!< \a alpha in the cutoff smoothing function
    };

#endif
