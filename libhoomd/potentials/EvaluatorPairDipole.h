/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: ndtrung

#ifndef __PAIR_EVALUATOR_DIPOLE_H__
#define __PAIR_EVALUATOR_DIPOLE_H__

#ifndef NVCC
#include <string>
#include <boost/python.hpp>
#endif

#include "QuaternionMath.h"

#include <iostream>
/*! \file EvaluatorPairDipole.h
    \brief Defines the dipole potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
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

#ifndef NVCC
using namespace boost::python;
#endif

class EvaluatorPairDipole
    {
    public:
        typedef Scalar3 param_type;
        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _quat_i Quaterion of i^{th} particle
            \param _quat_j Quaterion of j^{th} particle
            \param _mu Dipole magnitude of particles
            \param _A Electrostatic energy scale
            \param _kappa Inverse screening length
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDipole(Scalar3& _dr, Scalar4& _quat_i, Scalar4& _quat_j, Scalar _rcutsq, param_type& params)
            :dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j),
             mu(params.x), A(params.y), kappa(params.z)
            {
            }

        //! uses diameter
        DEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj){}

        //! whether pair potential requires charges
        DEVICE static bool needsCharge()
            {
            return true;
            }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj)
            {
            q_i = qi;
            q_j = qj;
            }

        //! Evaluate the force and energy
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
            vec3<Scalar> rvec(dr);
            Scalar rsq = dot(rvec, rvec);

            if(rsq > rcutsq)
                return false;

            Scalar rinv =  RSQRT(rsq);
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r3inv = r2inv*rinv;
            Scalar r5inv = r3inv*r2inv;

            // convert dipole vector in the body frame of each particle to space frame
            vec3<Scalar> p_i = rotate(quat<Scalar>(quat_i), vec3<Scalar>(mu, 0, 0));
            vec3<Scalar> p_j = rotate(quat<Scalar>(quat_j), vec3<Scalar>(mu, 0, 0));

            vec3<Scalar> f;
            vec3<Scalar> t_i;
            vec3<Scalar> t_j;
            Scalar e = Scalar(0.0);

            Scalar r = Scalar(1.0)/rinv;
            Scalar prefactor = A*_EXP(-kappa*r);

            // dipole-dipole
            if (mu != Scalar(0.0))
                {
                Scalar r7inv = r5inv*r2inv;
                Scalar pidotpj = dot(p_i, p_j);
                Scalar pidotr = dot(p_i, rvec);
                Scalar pjdotr = dot(p_j, rvec);

                Scalar pre1 = prefactor*(Scalar(3.0)*r5inv*pidotpj - Scalar(15.0)*r7inv*pidotr*pjdotr);
                Scalar pre2 = prefactor*Scalar(3.0)*r5inv*pjdotr;
                Scalar pre3 = prefactor*Scalar(3.0)*r5inv*pidotr;
                Scalar pre4 = prefactor*Scalar(-1.0)*r3inv;
                Scalar pre5 = prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr)*kappa*rinv;

                f += pre1*rvec + pre2*p_i + pre3*p_j + pre5*rvec;

                vec3<Scalar> scaledpicrosspj(pre4*cross(p_i, p_j));

                t_i += scaledpicrosspj + pre2*cross(p_i, rvec);
                t_j += -scaledpicrosspj + pre3*cross(p_j, rvec);

                e += prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr);
                }
            // dipole i - electrostatic j
            if (mu != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar pidotr = dot(p_i, rvec);
                Scalar pre1 = prefactor*Scalar(3.0)*q_j*r5inv * pidotr;
                Scalar pre2 = prefactor*q_j*r3inv;
                Scalar pre3 = prefactor*q_j*r3inv*pidotr*kappa*rinv;

                f += pre2*p_i - pre1*rvec + pre3*rvec;

                t_i += pre2*cross(p_i, rvec);

                e -= pidotr*pre2;
                }
            // electrostatic i - dipole j
            if (q_i != Scalar(0.0) && mu != Scalar(0.0))
                {
                Scalar pjdotr = dot(p_j, rvec);
                Scalar pre1 = prefactor*Scalar(3.0)*q_i*r5inv * pjdotr;
                Scalar pre2 = prefactor*q_i*r3inv;
                Scalar pre3 = prefactor*q_i*r3inv*pjdotr*kappa*rinv;

                f += pre1*rvec - pre2*p_j + pre3*rvec;

                t_j += -pre2*cross(p_j, rvec);

                e += pjdotr*pre2;
                }
            // electrostatic-electrostatic
            if (q_i != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar fforce = prefactor*q_i*q_j*(kappa+rinv)*r2inv;

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

        #endif

    protected:
        Scalar3 dr;                 //!< Stored vector pointing between particle centres of mass
        Scalar rcutsq;              //!< Stored rcutsq from the constructor
        Scalar q_i, q_j;            //!< Stored particle charges
        Scalar4 quat_i,quat_j;      //!< Stored quaternion of ith and jth particle from constuctor
        Scalar mu, A, kappa;        //!< Stored dipole magnitude, electrostatic magnitude and inverse screeing length
    };


#endif // __PAIR_EVALUATOR_DIPOLE_H__
