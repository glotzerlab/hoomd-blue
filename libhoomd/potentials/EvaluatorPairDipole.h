/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

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

//! Class for evaluating dipole-dipole, dipole-charge and charge-charge interations in polyelectrolyte solutions (i.e. screened electrostatics)
/*! <b>General Overview</b>
    Evaluate screened Coulombic energy, force and torque between two interacting particles with charge and dipole.

*/

struct EvaluatorPairDipoleParams
    {
    DEVICE inline  EvaluatorPairDipoleParams()
        {
        }
    DEVICE inline  EvaluatorPairDipoleParams(Scalar3 _mu_i, Scalar3 _mu_j, Scalar _A, Scalar _kappa, Scalar _qqrd2e)
      : mu_i(_mu_i), mu_j(_mu_j), A(_A), kappa(_kappa), qqrd2e(_qqrd2e)
        {
        }

      Scalar3 mu_i , mu_j;
      Scalar A, kappa, qqrd2e;
    };


class EvaluatorPairDipole
    {
    public:
        typedef EvaluatorPairDipoleParams param_type;
        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _quat_i Quaterion of i^{th} particle
            \param _quat_j Quaterion of j^{th} particle
            \param _mu_i Dipole magnitude of particles of type I
            \param _mu_j Dipole magnitude of particles of type J
            \param _A Electrostatic energy scale
            \param _kappa Inverse screening length
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairDipole(Scalar3& _dr, Scalar4& _quat_i, Scalar4& _quat_j, Scalar _rcutsq, param_type& params)
            :dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j), mu_i(params.mu_i), mu_j(params.mu_j),
            A(params.A), kappa(params.kappa)
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
            Scalar dx = dr.x;
            Scalar dy = dr.y;
            Scalar dz = dr.z;
            Scalar rsq = dx * dx + dy * dy + dz * dz;
            Scalar rinv =  RSQRT(rsq);
            Scalar r2inv = Scalar(1.0) / rsq;
            Scalar r3inv = r2inv*rinv;
            Scalar r5inv = r3inv*r2inv;

            // convert dipole vector in the body frame of each particle to space frame
            Scalar3 mubody_i = mu_i;
            Scalar3 p_i;
            quatrot(mubody_i, quat_i, p_i);

            Scalar3 mubody_j = mu_j;
            Scalar3 p_j;
            quatrot(mubody_j, quat_j, p_j);

            Scalar3 f = make_scalar3(0.0, 0.0, 0.0);
            Scalar3 t_i = make_scalar3(0.0, 0.0, 0.0);
            Scalar3 t_j = make_scalar3(0.0, 0.0, 0.0);
            Scalar e = Scalar(0.0);

            Scalar r = Scalar(1.0)/rinv;
            Scalar prefactor = A*_EXP(-kappa*r);

	    Scalar mu_i_sq = mu_i.x*mu_i.x + mu_i.y*mu_i.y + mu_i.z*mu_i.z;
	    Scalar mu_j_sq = mu_j.x*mu_j.x + mu_j.y*mu_j.y + mu_j.z*mu_j.z;

            // dipole-dipole
            if (mu_i_sq != Scalar(0.0) && mu_j_sq != Scalar(0.0))
                {
                Scalar r7inv = r5inv*r2inv;
                Scalar pidotpj = p_i.x*p_j.x + p_i.y*p_j.y + p_i.z*p_j.z;
                Scalar pidotr = p_i.x*dx + p_i.y*dr.y + p_i.z*dz;
                Scalar pjdotr = p_j.x*dx + p_j.y*dr.y + p_j.z*dz;

                Scalar pre1 = prefactor*(Scalar(3.0)*r5inv*pidotpj - Scalar(15.0)*r7inv*pidotr*pjdotr);
                Scalar pre2 = prefactor*Scalar(3.0)*r5inv*pjdotr;
                Scalar pre3 = prefactor*Scalar(3.0)*r5inv*pidotr;
                Scalar pre4 = prefactor*Scalar(-1.0)*r3inv;
                Scalar pre5 = prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr)*kappa*rinv;

                f.x += pre1*dx + pre2*p_i.x + pre3*p_j.x + pre5*dx;
                f.y += pre1*dy + pre2*p_i.y + pre3*p_j.y + pre5*dy;
                f.z += pre1*dz + pre2*p_i.z + pre3*p_j.z + pre5*dz;

                Scalar crossx = pre4 * (p_i.y*p_j.z - p_i.z*p_j.y);
                Scalar crossy = pre4 * (p_i.z*p_j.x - p_i.x*p_j.z);
                Scalar crossz = pre4 * (p_i.x*p_j.y - p_i.y*p_j.x);

                t_i.x += crossx + pre2 * (p_i.y*dz - p_i.z*dy);
                t_i.y += crossy + pre2 * (p_i.z*dx - p_i.x*dz);
                t_i.z += crossz + pre2 * (p_i.x*dy - p_i.y*dx);

                t_j.x += -crossx + pre3 * (p_j.y*dz - p_j.z*dy);
                t_j.y += -crossy + pre3 * (p_j.z*dx - p_j.x*dz);
                t_j.z += -crossz + pre3 * (p_j.x*dy - p_j.y*dx);

                e += prefactor*(r3inv*pidotpj - Scalar(3.0)*r5inv*pidotr*pjdotr);
                }
            // dipole i - electrostatic j
            if (mu_i_sq != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar pidotr = p_i.x*dx + p_i.y*dy + p_i.z*dz;
                Scalar pre1 = prefactor*Scalar(3.0)*q_j*r5inv * pidotr;
                Scalar pre2 = prefactor*q_j*r3inv;
                Scalar pre3 = prefactor*q_j*r3inv*pidotr*kappa*rinv;

                f.x += pre2*p_i.x - pre1*dx + pre3*dx;
                f.y += pre2*p_i.y - pre1*dy + pre3*dy;
                f.z += pre2*p_i.z - pre1*dz + pre3*dz;

                t_i.x += pre2 * (p_i.y*dz - p_i.z*dy);
                t_i.y += pre2 * (p_i.z*dx - p_i.x*dz);
                t_i.z += pre2 * (p_i.x*dy - p_i.y*dx);

                e += -q_j*r3inv*pidotr*prefactor;
                }
            // electrostatic i - dipole j
            if (q_i != Scalar(0.0) && mu_j_sq != Scalar(0.0))
                {
                Scalar pjdotr = p_j.x*dx + p_j.y*dy + p_j.z*dz;
                Scalar pre1 = prefactor*Scalar(3.0)*q_i*r5inv * pjdotr;
                Scalar pre2 = prefactor*q_i*r3inv;
                Scalar pre3 = prefactor*q_i*r3inv*pjdotr*kappa*rinv;

                f.x += pre1*dx - pre2*p_j.x + pre3*dx;
                f.y += pre1*dy - pre2*p_j.y + pre3*dy;
                f.z += pre1*dz - pre2*p_j.z + pre3*dz;

                t_j.x += -pre2 * (p_j.y*dz - p_j.z*dy);
                t_j.y += -pre2 * (p_j.z*dx - p_j.x*dz);
                t_j.z += -pre2 * (p_j.x*dy - p_j.y*dx);

                e += q_i*r3inv*pjdotr*prefactor;
                }
            // electrostatic-electrostatic
            if (q_i != Scalar(0.0) && q_j != Scalar(0.0))
                {
                Scalar fforce = prefactor*q_i*q_j*(kappa+rinv)*r2inv;

                f.x += fforce*dx;
                f.y += fforce*dy;
                f.z += fforce*dz;

                e += prefactor*q_i*q_j*rinv;
                }

            force = f;
            torque_i = t_i;
            torque_j = t_j;
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

        // make this classes parameters available to python
        static void export_params()
            {
            class_<EvaluatorPairDipoleParams>("EvaluatorPairDipoleParams",init<Scalar3, Scalar3, Scalar, Scalar, Scalar>());
            }
        #endif

    protected:
        Scalar3 dr;                 //!< Stored vector pointing between particle centres of mass
        Scalar rcutsq;              //!< Stored rcutsq from the constructor
        Scalar q_i, q_j;            //!< Stored particle charges
        Scalar4 quat_i,quat_j;      //!< Stored quaternion of ith and jth particle from constuctor
        Scalar3 mu_i,mu_j;           //!< Stored dipole magnitude of ith and jth particle from constuctor
        Scalar A, kappa;            //!< Stored magnitude and inverse screeing length
        Scalar qqrd2e;              //!< Stored conversion factor q^2/r to energy w/ dielectric constant
    };


#endif // __PAIR_EVALUATOR_DIPOLE_H__
