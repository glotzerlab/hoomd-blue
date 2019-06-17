// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: thivo

#ifndef __EVALUATOR_PAIR_2DALJ_H__
#define __EVALUATOR_PAIR_2DALJ_H__

#ifndef NVCC
#include <string>
#endif

#define HOOMD_2DALJ_MIN(i,j) ((i > j) ? j : i)
#define HOOMD_2DALJ_MAX(i,j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"
#include "hoomd/ManagedArray.h"
#include "ALJ2DData.h"
#include "GJK_vyas.h"
#include <iostream>

#include <limits.h>
#include <utility>

/*! \file EvaluatorPair2DALJ.h
    \brief Defines a an evaluator class for the anisotropic LJ potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

/*!
 * Anisotropic LJ potential (assuming analytical kernel and (temporarily) sigma = 1.0)
 */

class EvaluatorPair2DALJ
    {
    public:
        typedef shape_2D param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _q_i Quaterion of i^th particle
            \param _q_j Quaterion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPair2DALJ(Scalar3& _dr,
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
            
            // Define relevant distance parameters (rsqr, r, directional vector)
            Scalar rsq = dot(dr,dr);
            Scalar r = sqrt(rsq);
            vec3<Scalar> unitr = dr/sqrt(dot(dr,dr));

            Scalar rho = 0.0;
            Scalar invr_rsq = 0.0;
            Scalar invr_6 = 0.0; 
            Scalar sub_sphere = 0.15;
            
            vec3<Scalar> f;

            vec3<Scalar> r1_store;
            vec3<Scalar> rvect;
            vec3<Scalar> rvect1;
            vec3<Scalar> rvect2;

            // // Define vectors a priori
            // ManagedArray<float> xi_internal;
            // ManagedArray<float> yi_internal;
            // ManagedArray<float> zi_internal;
            // ManagedArray<float> xj_internal;
            // ManagedArray<float> yj_internal;
            // ManagedArray<float> zj_internal;         
            
            // int Ni_internal;
            // int Nj_internal;
            // if (fabs(sigma_i - dia_i) < 0.01)
            //     {
            //     Ni_internal = int(Ni + 0.001);
            //     xi_internal = xi;
            //     yi_internal = yi;
            //     zi_internal = zi;
            //     }
            // else
            //     {
            //     Ni_internal = int(Nj + 0.001);
            //     xi_internal = xj;
            //     yi_internal = yj;
            //     zi_internal = zj;
            //     }
            // if (fabs(sigma_j - dia_j) < 0.01)
            //     {
            //     Nj_internal = int(Nj + 0.001);
            //     xj_internal = xj;
            //     yj_internal = yj;
            //     zj_internal = zj;
            //     }
            // else
            //     {
            //     Nj_internal = int(Ni + 0.001);
            //     xj_internal = xi;
            //     yj_internal = yi;
            //     zj_internal = zi;
            //     }

            // Create rotate structures
            vec3<Scalar> vertsi[20];
            vec3<Scalar> vertsj[20];
            Scalar xi_final[20];
            Scalar yi_final[20];
            Scalar zi_final[20];
            Scalar xj_final[20];
            Scalar yj_final[20];
            Scalar zj_final[20];
            // float xi_final[Ni_internal];
            // float yi_final[Ni_internal];
            // float zi_final[Ni_internal];
            // float xj_final[Nj_internal];
            // float yj_final[Nj_internal];
            // float zj_final[Nj_internal];

            Scalar factor = 1.0;
            vec3<Scalar> tmp;            
            // Define internal shape_i
            for (unsigned int i = 0; i < _params.Ni; i=i+1)
              {
              tmp.x = factor*_params.xi[i];
              tmp.y = factor*_params.yi[i];
              tmp.z = 0.0;
              tmp = rotate(qi,tmp);
              xi_final[i] = tmp.x;
              yi_final[i] = tmp.y;
              zi_final[i] = 0.0;
              }

            // Define internal shape_j
            for (unsigned int i = 0; i < _params.Nj; i=i+1)
              {
              tmp.x = factor*_params.xj[i];
              tmp.y = factor*_params.yj[i];
              tmp.z = 0.0;
              tmp = rotate(qj,tmp);
              xj_final[i] = tmp.x + Scalar(-1.0)*dr.x;
              yj_final[i] = tmp.y + Scalar(-1.0)*dr.y;
              zj_final[i] = 0.0;
              }

            for (unsigned int i = 0; i < _params.Ni; i++)
            {
              vertsi[i] = vec3<Scalar>(xi_final[i], yi_final[i], zi_final[i]);
            }
            for (unsigned int i = 0; i < _params.Nj; i++)
            {
              vertsj[i] = vec3<Scalar>(xj_final[i], yj_final[i], zj_final[i]);
            }
            

            // Distance
            if ( (r/_params.ki_max < sqrt(rcutsq)) | (r/_params.kj_max < sqrt(rcutsq)) )
              {

              // Call gjk
              vec3<Scalar> pos1;  // First particle is at 0
              vec3<Scalar> pos2 = Scalar(-1.0)*dr;  // Second particle is at -dr
              vec3<Scalar> v, v1, v2;
              vec3<Scalar> a;
              vec3<Scalar> b;
              a.z = 0.0;
              b.z = 0.0;
              v.z = 0.0;
              v1.z = 0.0;
              v2.z = 0.0;
              bool success, overlap;
              gjk_inline<2>(pos1, pos2, vertsi, _params.Ni, vertsj, _params.Nj, v1, v2, a, b, success, overlap);
              

              // Get kernel
              Scalar sigma12 = (_params.sigma_i + _params.sigma_j)*Scalar(0.5);     
              Scalar epsilon = _params.epsilon;
    	      v.z = 0.0;
    	      v1.z = 0.0;
    	      v2.z = 0.0;
    	      a.z = 0.0;
    	      b.z = 0.0;

              rho = sigma12 /r;
              invr_rsq = rho*rho;
              invr_6 = invr_rsq*invr_rsq*invr_rsq;
              Scalar denom = (invr_6*invr_6 - invr_6);
              Scalar k1 = sqrt(dot(a,a));
              Scalar k2 = sqrt(dot(b,b));
              rho = sigma12 / (r - Scalar(0.5)*(k1/_params.sigma_i - 1.0) - Scalar(0.5)*(k2/_params.sigma_j - 1.0));
              invr_rsq = rho*rho;
              invr_6 = invr_rsq*invr_rsq*invr_rsq;
              Scalar numer = (invr_6*invr_6 - invr_6);
              epsilon = epsilon*(numer/denom);

              // Define relevant vectors
              if (overlap)
                  {
                  rvect = -1.0*v1;
                  }
              else
                  {
                  rvect = b - a;
                  }              
              Scalar rcheck = dot(v1,v1);
              Scalar rcheck_isq = fast::rsqrt(rcheck);
              // rvect = b - a;
              rvect = rvect*rcheck_isq;
              Scalar f_scalar;
              Scalar f_scalar_contact;
              Scalar r1check, r2check;
              Scalar f_scalar_contact1, f_scalar_contact2;

              // Check repulsion vs attraction for center particle
              if (_params.alpha < 1.0)
                  {
                  if (r < 1.12246204831*sigma12)
                      {
                      // Center force and energy
                      rho = sigma12 /r;
                      invr_rsq = rho*rho;
                      invr_6 = invr_rsq*invr_rsq*invr_rsq;
                      pair_eng = Scalar(4.0) * epsilon * (invr_6*invr_6 - invr_6); 
                      f_scalar = Scalar(4.0) * epsilon * ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / (r);

                      // Shift energy
                      rho = 1.0 / 1.12246204831;
                      invr_rsq = rho*rho;
                      invr_6 = invr_rsq*invr_rsq*invr_rsq;
                      pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - invr_6);
                      }
                  else
                      {
                      pair_eng = 0.0;
                      f_scalar = 0.0;
                      }
                  }
              else
                  {
                  // Center force and energy
                  rho = sigma12 /r;
                  invr_rsq = rho*rho;
                  invr_6 = invr_rsq*invr_rsq*invr_rsq;
                  pair_eng = Scalar(4.0) * epsilon * (invr_6*invr_6 - invr_6); 
                  f_scalar = Scalar(4.0) * epsilon * ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / (r);
                  }

              // Check repulsion attraction for contact point
              epsilon = _params.epsilon;
              overlap = false;
              if (!overlap)
                  {
                  // No overlap
                  if (_params.alpha*0.0 < 1.0)
                      {
                      if (sqrt(rcheck)  < 1.12246204831*(sub_sphere*sigma12))
                          {
                          // Contact force and energy
                          rho = sub_sphere * sigma12 * rcheck_isq;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6) ;
                          f_scalar_contact = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) * rcheck_isq;

                          // Shift energy
                          rho = 1.0 / 1.12246204831;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                          }
                      else
                          {
                          pair_eng += 0.0;
                          f_scalar_contact = 0.0;
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
                  f.z = 0.0;	
                  force = vec_to_scalar3(f);

                  // Torque
                  r1_store = a;
                  Scalar ftotal = 0.5*sqrt(dot(f,f));
                  ftotal = 1.0*f_scalar_contact;
                  torque_i = vec_to_scalar3(cross(r1_store-0.5*sub_sphere*sigma12*rvect+sqrt(rcheck)*rvect,Scalar(1.0)*f));
                  torque_j = vec_to_scalar3(cross(dr+r1_store+0.5*sub_sphere*sigma12*rvect,Scalar(-1.0)*f));  

                  }
              else
                  {
                  // j^th particle inside i^th particle 
                  // Scalar sigma_test = sqrt(dot(b-a,b-a));
                  r1check = sqrt(dot(v1,v1));
                  rvect1 = -v1/r1check;
                  Scalar sigma12_1 = (_params.sigma_i + (r1check - _params.sigma_i)*2.0)*Scalar(0.5);

                  if (_params.alpha < 1.0)
                      {
                      if ((r1check)  < 1.12246204831*(sigma12_1))
                          {
                          // Contact force and energy
                          rho = sigma12_1 / r1check;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                          f_scalar_contact1 = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / r1check;

                          // Shift energy
                          rho = 1.0 / 1.12246204831;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                          }
                      else
                          {
                          pair_eng += 0.0;
                          f_scalar_contact1 = 0.0;
                          }
                      }
                  else
                      {
                      // Contact force and energy
                      rho = sigma12_1 / r1check;
                      invr_rsq = rho*rho;
                      invr_6 = invr_rsq*invr_rsq*invr_rsq;
                      pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                      f_scalar_contact1 = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / r1check;                
                      }


                  // i^th particle inside j^th particle 
                  r2check = sqrt(dot(v2,v2));
                  rvect2 = v2/r2check;
                  Scalar sigma12_2 = (_params.sigma_j + (r2check - _params.sigma_j)*2.0)*Scalar(0.5);              

                  if (_params.alpha < 1.0)
                      {
                      if ((r2check)  < 1.12246204831*(sigma12_2))
                          {
                          // Contact force and energy
                          rho = sigma12_2 / r2check;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                          f_scalar_contact2 = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / r2check;

                          // Shift energy
                          rho = 1.0 / 1.12246204831;
                          invr_rsq = rho*rho;
                          invr_6 = invr_rsq*invr_rsq*invr_rsq;
                          pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                          }
                      else
                          {
                          pair_eng += 0.0;
                          f_scalar_contact2 = 0.0;
                          }
                      }
                  else
                      {
                      // Contact force and energy
                      rho = sigma12_2 / r2check;
                      invr_rsq = rho*rho;
                      invr_6 = invr_rsq*invr_rsq*invr_rsq;
                      pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                      f_scalar_contact2 = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) / r2check;                
                      }

                  // Net force
                  Scalar test_factor = 0.5;
                  f = f_scalar * unitr - ( test_factor*f_scalar_contact1 * rvect1 - test_factor*f_scalar_contact2 * rvect2 );              
                  force = vec_to_scalar3(f);

                  // Torque
                  torque_i = vec_to_scalar3(cross(rvect1*r1check-(b-a),Scalar(-1.0)*( test_factor*f_scalar_contact1 * rvect1 - test_factor*f_scalar_contact2 * rvect2 )));
                  torque_j = vec_to_scalar3(cross(rvect1*r1check-Scalar(-1.0)*dr,Scalar(1.0)*( test_factor*f_scalar_contact1 * rvect1 - test_factor*f_scalar_contact2 * rvect2 ))); 

                  }
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
            return "alj2D";
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


#undef HOOMD_2DALJ_MIN
#undef HOOMD_2DALJ_MAX
#endif // __EVALUATOR_PAIR_2DALJ_H__
