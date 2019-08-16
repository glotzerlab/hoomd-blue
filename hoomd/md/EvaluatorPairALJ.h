// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: thivo

#ifndef __EVALUATOR_PAIR_ALJ_H__
#define __EVALUATOR_PAIR_ALJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/VectorMath.h"
#include "hoomd/ManagedArray.h"
#include "ALJData.h"
#include "GJK.h"
#include <iostream>

/*! \file EvaluatorPairALJ.h
    \brief Defines a an evaluator class for the anisotropic LJ table potential.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE

#if defined (__SSE__)
#include <immintrin.h>
#endif
#endif

/*!
 * Anisotropic LJ potential (assuming analytical kernel and (temporarily) sigma = 1.0)
 */

template <unsigned int ndim>
class EvaluatorPairALJ
    {
    public:
        typedef shape_table param_type;

        //! Constructs the pair potential evaluator.
        /*! \param _dr Displacement vector between particle centers of mass.
            \param _rcutsq Squared distance at which the potential goes to 0.
            \param _q_i Quaternion of i^th particle.
            \param _q_j Quaternion of j^th particle.
            \param _params Per type pair parameters of this potential.
        */
        DEVICE EvaluatorPairALJ(Scalar3& _dr,
                                Scalar4& _qi,
                                Scalar4& _qj,
                                Scalar _rcutsq,
                                const param_type& _params)
            : dr(_dr),rcutsq(_rcutsq),qi(_qi),qj(_qj), _params(_params)
            {
            }

        //! uses diameter
        DEVICE static bool needsDiameter()
            {
            return true;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            dia_i = di;
            dia_j = dj;
            }

        //! whether pair potential requires charges
        DEVICE static bool needsCharge( )
            {
            return false;
            }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Evaluate the force and energy.
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        DEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            // Define relevant distance parameters (rsqr, r, directional vector)
            Scalar rsq = dot(dr,dr);
            Scalar r = sqrt(rsq);
            vec3<Scalar> unitr = dr/r;

            // Interaction cutoff is scaled by the max kernel value scaled by
            // the insphere radius, which is the max vertex distance
            // k[ij]_maxsq.
            if ( (rsq/_params.ki_maxsq < rcutsq) | (rsq/_params.kj_maxsq < rcutsq) )
                {
                // Call gjk
                bool success, overlap;
                vec3<Scalar> v = vec3<Scalar>(), a = vec3<Scalar>(), b = vec3<Scalar>();
                    {
                    vec3<Scalar> v1 = vec3<Scalar>(), a1 = vec3<Scalar>(), b1 = vec3<Scalar>();
                    vec3<Scalar> v2 = vec3<Scalar>(), a2 = vec3<Scalar>(), b2 = vec3<Scalar>();
                    bool success1, overlap1;
                    bool success2, overlap2;

                    gjk<ndim>(_params.verts_i, _params.verts_j, v1, a1, b1, success1, overlap1, qi, qj, dr);
                    gjk<ndim>(_params.verts_j, _params.verts_i, v2, a2, b2, success2, overlap2, qj, qi, -dr);

                    if (dot(v1, v1) < dot(v2, v2))
                        {
                            v = v1;
                            a = a1;
                            b = b1;
                            success = success1;
                            overlap = overlap1;
                        }
                    else
                        {
                        v = -v2;
                        a = b2;
                        b = a2;
                        success = success2;
                        overlap = overlap2;
                        }
                    }
                if (ndim == 2)
                    {
                    v.z = 0;
                    a.z = 0;
                    b.z = 0;
                    }

                // Get kernel
                Scalar sigma12 = (_params.sigma_i + _params.sigma_j)*Scalar(0.5);

                Scalar sub_sphere = 0.15;
                const Scalar two_p_16 = 1.12246204831;  // 2^(1/6)

                vec3<Scalar> f;
                vec3<Scalar> rvect;

                Scalar k1 = sqrt(dot(a,a));
                Scalar k2 = sqrt(dot(b,b));
                Scalar rho = sigma12 / (r - Scalar(0.5)*(k1/_params.sigma_i - 1.0) - Scalar(0.5)*(k2/_params.sigma_j - 1.0));
                Scalar invr_rsq = rho*rho;
                Scalar invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar numer = (invr_6*invr_6 - invr_6);

                invr_rsq = sigma12*sigma12/rsq;
                invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar invr_12 = invr_6*invr_6;
                Scalar denom = invr_12 - invr_6;

                Scalar epsilon = _params.epsilon;
                Scalar scaled_epsilon = epsilon*(numer/denom);

                // Define relevant vectors
                rvect = Scalar(-1.0)*v;
                Scalar vsq = dot(v,v);
                Scalar rcheck_isq = fast::rsqrt(vsq);
                rvect = rvect*rcheck_isq;
                Scalar f_scalar = 0;
                Scalar f_scalar_contact = 0;

                // Check repulsion vs attraction for center particle
                if (_params.alpha < 1.0)
                    {
                    if (r < two_p_16*sigma12)
                        {
                        // Center force and energy
                        pair_eng = Scalar(4.0) * scaled_epsilon * (invr_12 - invr_6);
                        f_scalar = Scalar(4.0) * scaled_epsilon * ( Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6 ) / (r);

                        // Shift energy
                        rho = 1.0 / two_p_16;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng -= Scalar(4.0) * scaled_epsilon * (invr_6*invr_6 - invr_6);
                        }
                    else
                        {
                        pair_eng = 0.0;
                        }
                    }
                else
                    {
                    // Center force and energy
                    pair_eng = Scalar(4.0) * scaled_epsilon * (invr_12 - invr_6);
                    f_scalar = Scalar(4.0) * scaled_epsilon * ( Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6 ) / r;
                    }

                // Check repulsion attraction for contact point
                // No overlap
                if (_params.alpha*0.0 < 1.0)
                    {
                    if (1/rcheck_isq  < two_p_16 *sub_sphere*sigma12)
                        {
                        // Contact force and energy
                        rho = sub_sphere * sigma12 * rcheck_isq;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                        f_scalar_contact = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) * rcheck_isq;

                        // Shift energy
                        rho = 1.0 / two_p_16;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                        }
                    }
                else
                    {
                    // Contact force and energy
                    rho = sub_sphere * sigma12 * rcheck_isq;
                    invr_rsq = rho*rho;
                    invr_6 = invr_rsq*invr_rsq*invr_rsq;
                    pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                    f_scalar_contact = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) * rcheck_isq;
                    }

                // Net force
                f = f_scalar * unitr - f_scalar_contact * rvect;
                if (ndim == 2)
                    {
                    f.z = 0;
                    }
                force = vec_to_scalar3(f);

                // Torque
                vec3<Scalar> lever = 0.5*sub_sphere*sigma12*rvect;
                torque_i = vec_to_scalar3(cross(a - lever + rvect/rcheck_isq, f));
                torque_j = vec_to_scalar3(cross(dr + a + lever, Scalar(-1.0)*f));

                return true;
                }
              else
                {
                return false;
                }
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "alj_table";
            }
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        Scalar dia_i;
        Scalar dia_j;
        const param_type& _params;
    };


#endif // __EVALUATOR_PAIR_ALJ_H__
