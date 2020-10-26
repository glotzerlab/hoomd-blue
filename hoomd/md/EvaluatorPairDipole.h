// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// $Id$
// $URL$
// Maintainer: ndtrung

#ifndef __PAIR_EVALUATOR_DIPOLE_H__
#define __PAIR_EVALUATOR_DIPOLE_H__

#ifndef NVCC
#include <string>
#endif

#include "QuaternionMath.h"

#include <iostream>
/*! \file EvaluatorPairDipole.h
    \brief Defines the dipole potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

#ifdef NVCC
#define _POW powf
#else
#define _POW pow
#endif

#ifdef NVCC
#define _SQRT sqrtf
#else
#define _SQRT sqrt
#endif

#ifdef SINGLE_PRECISION
#define _EXP(x) expf( (x) )
#else
#define _EXP(x) exp( (x) )
#endif

struct pair_dipole_params
    {
    Scalar mu;         //! The magnitude of the magnetic moment.
    Scalar A;          //! The electrostatic energy scale.
    Scalar kappa;      //! The inverse screening length.

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        // No-op for this struct since it contains no arrays.
        }
    };

// Nullary structure required by AnisoPotentialPair.
struct dipole_shape_params
    {
    HOSTDEVICE dipole_shape_params() {}

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const {}
    #endif
    };

class EvaluatorPairDipole
    {
    public:
        typedef pair_dipole_params param_type;
        typedef dipole_shape_params shape_param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centers of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _quat_i Quaternion of i^{th} particle
            \param _quat_j Quaternion of j^{th} particle
            \param _mu Dipole magnitude of particles
            \param _A Electrostatic energy scale
            \param _kappa Inverse screening length
            \param _params Per type pair parameters of this potential
        */
        HOSTDEVICE EvaluatorPairDipole(Scalar3& _dr, Scalar4& _quat_i, Scalar4& _quat_j, Scalar _rcutsq, const param_type& _params)
            :dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j), params(_params)
            {
            }

        //! uses diameter
        HOSTDEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Whether the pair potential uses shape.
        HOSTDEVICE static bool needsShape()
            {
            return false;
            }

        //! Whether the pair potential needs particle tags.
        HOSTDEVICE static bool needsTags()
            {
            return false;
            }

        //! whether pair potential requires charges
        HOSTDEVICE static bool needsCharge()
            {
            return true;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        HOSTDEVICE void setDiameter(Scalar di, Scalar dj){}

        //! Accept the optional shape values
        /*! \param shape_i Shape of particle i
            \param shape_j Shape of particle j
        */
        HOSTDEVICE void setShape(const shape_param_type *shapei, const shape_param_type *shapej) {}

        //! Accept the optional tags
        /*! \param tag_i Tag of particle i
            \param tag_j Tag of particle j
        */
        HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) {}

        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        HOSTDEVICE void setCharge(Scalar qi, Scalar qj)
            {
            q_i = qi;
            q_j = qj;
            }

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exerted on the i^th particle.
            \param torque_j The torque exerted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        HOSTDEVICE  bool
      evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            vec3<Scalar> rvec(dr);
            Scalar rsq = dot(rvec, rvec);

            if(rsq > rcutsq)
                return false;

            Scalar rinv =  RSQRT(rsq);
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r3inv = r2inv*rinv;
            Scalar r5inv = r3inv*r2inv;

            // convert dipole vector in the body frame of each particle to space frame
            vec3<Scalar> p_i = rotate(quat<Scalar>(quat_i), vec3<Scalar>(params.mu, 0, 0));
            vec3<Scalar> p_j = rotate(quat<Scalar>(quat_j), vec3<Scalar>(params.mu, 0, 0));

            vec3<Scalar> f;
            vec3<Scalar> t_i;
            vec3<Scalar> t_j;
            Scalar e = Scalar(0.0);

            Scalar r = Scalar(1.0)/rinv;
            Scalar prefactor = params.A*_EXP(-params.kappa*r);

            // dipole-dipole
            if (params.mu != Scalar(0.0))
                {
                Scalar r7inv = r5inv*r2inv;
                Scalar pidotpj = dot(p_i, p_j);
                Scalar pidotr = dot(p_i, rvec);
                Scalar pjdotr = dot(p_j, rvec);

                Scalar pre1 = prefactor*(Scalar(3.0)*r5inv*pidotpj - Scalar(15.0)*r7inv*pidotr*pjdotr);
                Scalar pre2 = prefactor*Scalar(3.0)*r5inv*pjdotr;
                Scalar pre3 = prefactor*Scalar(3.0)*r5inv*pidotr;
                Scalar pre4 = prefactor*Scalar(-1.0)*r3inv;
                Scalar pre5 = prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr)*params.kappa*rinv;

                f += pre1*rvec + pre2*p_i + pre3*p_j + pre5*rvec;

                vec3<Scalar> scaledpicrosspj(pre4*cross(p_i, p_j));

                t_i += scaledpicrosspj + pre2*cross(p_i, rvec);
                t_j += -scaledpicrosspj + pre3*cross(p_j, rvec);

                e += prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr);
                }
            // dipole i - electrostatic j
            if (params.mu != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar pidotr = dot(p_i, rvec);
                Scalar pre1 = prefactor*Scalar(3.0)*q_j*r5inv * pidotr;
                Scalar pre2 = prefactor*q_j*r3inv;
                Scalar pre3 = prefactor*q_j*r3inv*pidotr*params.kappa*rinv;

                f += pre2*p_i - pre1*rvec - pre3*rvec;

                t_i += pre2*cross(p_i, rvec);

                e -= pidotr*pre2;
                }
            // electrostatic i - dipole j
            if (q_i != Scalar(0.0) && params.mu != Scalar(0.0))
                {
                Scalar pjdotr = dot(p_j, rvec);
                Scalar pre1 = prefactor*Scalar(3.0)*q_i*r5inv * pjdotr;
                Scalar pre2 = prefactor*q_i*r3inv;
                Scalar pre3 = prefactor*q_i*r3inv*pjdotr*params.kappa*rinv;

                f += pre1*rvec - pre2*p_j + pre3*rvec;

                t_j += -pre2*cross(p_j, rvec);

                e += pjdotr*pre2;
                }
            // electrostatic-electrostatic
            if (q_i != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar fforce = prefactor*q_i*q_j*(params.kappa+rinv)*r2inv;

                f += fforce*rvec;

                e += prefactor*q_i*q_j*rinv;
                }

            force = vec_to_scalar3(f);
            torque_i = vec_to_scalar3(t_i);
            torque_j = vec_to_scalar3(t_j);
            pair_eng = e;
            return true;
            }


       #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "dipole";
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar3 dr;                 //!< Stored vector pointing between particle centers of mass
        Scalar rcutsq;              //!< Stored rcutsq from the constructor
        Scalar q_i, q_j;            //!< Stored particle charges
        Scalar4 quat_i,quat_j;      //!< Stored quaternion of ith and jth particle from constructor
        const param_type &params;   //!< The pair potential parameters
    };


#endif // __PAIR_EVALUATOR_DIPOLE_H__
