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
// Maintainer: grva

#ifndef __EvaluatorPairGayBerne__H
#define __EvaluatorPairGayBerne__H

#ifndef NVCC
#include <string>
#endif

#include "QuaternionMath.h"
#include "MatrixMath.h"

/*! \file EvaluatorPairGayBerne.h
    \brief Defines a mostly abstract base class for anisotropic pair potentials
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

#ifndef INDEX3
#define INDEX3(i,j) i*3+j
#endif

//! Class for evaluating anisotropic pair interations
/*! <b>General Overview</b>

    Provides a base class for detailed anisotropic pairwise interactions
*/
enum GayBerneType
    { 
    GAYBERNE_ELLIPSE_ELLIPSE,
    GAYBERNE_SPHERE_ELLIPSE,
    GAYBERNE_ELLIPSE_SPHERE,
    GAYBERNE_SPHERE_SPHERE
    };

struct EvaluatorPairGayBerneParams
    {
    Scalar sigma_0;
    Scalar nu,mu;
    Scalar gamma;
    Scalar3 S_i; //particle i's radii
    Scalar3 S_j; //particle j's radii
    Scalar s_i,s_j; //geometric factors
    Scalar3 epsilon_i,epsilon_j; //interaction energies along the various radii
    Scalar epsilon;
    Scalar Bi[9],Bj[9];
    Scalar Gi[9],Gj[9];
    GayBerneType gb_type;
    };

class EvaluatorPairGayBerne
    {
    public:
        typedef EvaluatorPairGayBerneParams param_type;
        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
	    \param _q_i Quaterion of i^th particle
	    \param _q_j Quaterion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairGayBerne(Scalar3& _dr, Scalar4& _quat_i, Scalar4& _quat_j, Scalar _rcutsq, param_type& params)
            :dr(_dr),rcutsq(_rcutsq),quat_i(_quat_i),quat_j(_quat_j),epsilon(params.epsilon),epsilon_i(params.epsilon_i),
             epsilon_j(params.epsilon_j),sigma_0(params.sigma_0),mu(params.mu), nu(params.nu),gamma(params.gamma),
             S_i(params.S_i),S_j(params.S_j),s_i(params.s_i),s_j(params.s_j)
            {
             matCopy3(Bi,params.Bi);
             matCopy3(Bj,params.Bj);
             matCopy3(Gi,params.Gi);
             matCopy3(Gj,params.Gj);
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
  
            //Implement the ellipse-ellipse calculation
            //preallocate some quantities
        
            Scalar tmpMat[9];
            
            quatToR(quat_i,Ai);
            
            matDiagMult3(S_i,Ai,tmpMat);
            matTransMatMult3(tmpMat,tmpMat,Gi);
            
            matDiagMult3(epsilon_i,Ai,tmpMat);
            matTransMatMult3(tmpMat,tmpMat,Bi);
            
            quatToR(quat_j,Aj);
            
            matDiagMult3(S_j,Aj,tmpMat);
            matTransMatMult3(tmpMat,tmpMat,Gj);
            
            matDiagMult3(epsilon_j,Aj,tmpMat);
            matTransMatMult3(tmpMat,tmpMat,Bj);
            
            Scalar r_sq;
            norm3(dr,r_sq);
            r_sq*=r_sq;

            //evaluate the energy = Ur*chi*eta
            Scalar Ur,eta,chi;
            Scalar rho,rho12,rho6;
            Scalar kappa[3],iota[3];
            Scalar r_inv = RSQRT(r_sq);
            Scalar r_hat[3];
            Scalar sigma_ij;
            Scalar det_G_ij;
            Scalar tmpVec[3],tmp;

            r_hat[0] = dr.x*r_inv;
            r_hat[1] = dr.y*r_inv;
            r_hat[2] = dr.z*r_inv;

            matMatAdd3(Gi,Gj,G_ij);
            matInverse3(G_ij,tmpMat);
            matVecMult3(tmpMat,dr,kappa);

            matMatAdd3(Bi,Bj,B_ij);
            matInverse3(B_ij,tmpMat);
            matVecMult3(tmpMat,dr,iota);
            
            det3(G_ij,det_G_ij);
            
            eta = _POW(Scalar(2.0)*s_i*s_j/det_G_ij,nu);
            
            dot3(r_hat,iota,chi);
            chi*= Scalar(2.0)*r_inv;
            chi = _POW(chi,mu);

            dot3(kappa,r_hat,sigma_ij);
            sigma_ij*=Scalar(0.5)*r_inv;
            sigma_ij=RSQRT(sigma_ij);

            rho = sigma_0/(r_sq*r_inv-gamma*sigma_ij+sigma_0);
            rho6 = rho*rho*rho;
            rho6*= rho6;
            rho12 = rho6*rho6;

            Ur = Scalar(4.0)*epsilon*(rho12-rho6);       
            
            pair_eng=Ur*chi*eta;

            //compute the force
            Scalar dU[3],dChi[3],dEta[3];
            Scalar tmp1,tmp2;
            Scalar dU_j[3],dEta_j[3],dChi_j[3];

            dot3(kappa,r_hat,tmp1);

            tmp = Scalar(24.0)*epsilon*rho*(rho12-rho6)/sigma_0;
            tmp2 =  tmp*r_inv*r_inv*sigma_ij*sigma_ij*sigma_ij*Scalar(0.5);

            dU[0]=tmp*r_hat[0] +tmp2*(kappa[0]-tmp1*r_hat[0]);
            dU[1]=tmp*r_hat[1] +tmp2*(kappa[1]-tmp1*r_hat[1]);
            dU[2]=tmp*r_hat[2] +tmp2*(kappa[2]-tmp1*r_hat[2]);
    
            dot3(iota,r_hat,tmp1);
            tmp=Scalar(-4.0)*_POW(chi,(mu-Scalar(1.0))/mu)*mu*r_inv*r_inv;
            dChi[0]=tmp*(iota[0]-tmp1*r_hat[0]);
            dChi[1]=tmp*(iota[1]-tmp1*r_hat[1]);
            dChi[2]=tmp*(iota[2]-tmp1*r_hat[2]);

            force.x = -eta*(dU[0]*chi+Ur*dChi[0]);
            force.y = -eta*(dU[1]*chi+Ur*dChi[1]);
            force.z = -eta*(dU[2]*chi+Ur*dChi[2]);

            //calculate the torque
            rowVecMatMult3(kappa,Gi,tmpVec);
            cross3(tmpVec,kappa,dU);
            tmp2*=sigma_ij*sigma_ij*sigma_ij;
            dU[0]*=tmp2;
            dU[1]*=tmp2;
            dU[2]*=tmp2;
            
            rowVecMatMult3(kappa,Gj,tmpVec);
            cross3(tmpVec,kappa,dU_j);
            dU_j[0]*=tmp2;
            dU_j[1]*=tmp2;
            dU_j[2]*=tmp2;

            rowVecMatMult3(iota,Bi,tmpVec);
            cross3(tmpVec,iota,dChi);
            tmp2=Scalar(-4.0)*r_inv*r_inv;
            dChi[0]*=tmp2;
            dChi[1]*=tmp2;
            dChi[2]*=tmp2;

            rowVecMatMult3(iota,Bj,tmpVec);
            cross3(tmpVec,iota,dChi_j);
            tmp2=Scalar(-4.0)*r_inv*r_inv;
            dChi_j[0]*=tmp2;
            dChi_j[1]*=tmp2;
            dChi_j[2]*=tmp2;

            dEtaTrace(Ai,G_ij,S_i,tmpMat);
            for(int ii=0;ii<3;++ii)
                {
                cross3(Ai+3*ii,tmpMat+3*ii,tmpVec);
                dEta[0]+=tmpVec[0];
                dEta[1]+=tmpVec[1];
                dEta[2]+=tmpVec[2];
                }
            tmp2=Scalar(-0.5)*eta*nu;
            dEta[0]*=tmp2;
            dEta[1]*=tmp2;
            dEta[2]*=tmp2;

            dEtaTrace(Aj,G_ij,S_j,tmpMat);
            for(int ii=0;ii<3;++ii)
                {
                cross3(Aj+3*ii,tmpMat+3*ii,tmpVec);
                dEta_j[0]+=tmpVec[0];
                dEta_j[1]+=tmpVec[1];
                dEta_j[2]+=tmpVec[2];
                } 
            tmp2=Scalar(-0.5)*eta*nu;
            dEta_j[0]*=tmp2;
            dEta_j[1]*=tmp2;
            dEta_j[2]*=tmp2;

            tmp = Ur*chi;
            tmp1= Ur*eta;
            tmp2= eta*chi;
            torque_i.x=Scalar(-1.0)*(tmp2*dU[0]+tmp1*dChi[0]+tmp*dEta[0]);
            torque_i.y=Scalar(-1.0)*(tmp2*dU[1]+tmp1*dChi[1]+tmp*dEta[1]);
            torque_i.z=Scalar(-1.0)*(tmp2*dU[2]+tmp1*dChi[2]+tmp*dEta[2]);
        
            return true;
            }
            
       #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "gay_berne";
            }
        #endif

    protected:
        Scalar3 dr;     //!< Stored vector pointing between particle centres of mass
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar4 quat_i,quat_j;     //!< Stored quaternion of ith and jth particle from constuctor
        Scalar Ai[9],Aj[9];
        Scalar Bi[9],Bj[9],B_ij[9];
        Scalar Gi[9],Gj[9],G_ij[9];
        Scalar epsilon;
        Scalar3 epsilon_i,epsilon_j;
        Scalar sigma_0;
        Scalar mu,nu,gamma;
        Scalar3 S_i,S_j;
        Scalar s_i,s_j;

        DEVICE void dEtaTrace(Scalar* A, Scalar* B,Scalar3& s, Scalar* C )
            {
            Scalar denom = A[INDEX3(1,0)]*A[INDEX3(0,2)]*A[INDEX3(2,1)]-A[INDEX3(0,0)]*A[INDEX3(1,2)]*A[INDEX3(2,1)]-
                           A[INDEX3(0,2)]*A[INDEX3(2,0)]*A[INDEX3(1,1)]+A[INDEX3(0,1)]*A[INDEX3(2,0)]*A[INDEX3(1,2)]-
                           A[INDEX3(1,0)]*A[INDEX3(0,1)]*A[INDEX3(2,2)]+A[INDEX3(0,0)]*A[INDEX3(1,1)]*A[INDEX3(2,2)];
              
            C[INDEX3(0,0)] = s.x*(A[INDEX3(1,2)]*A[INDEX3(0,1)]*B[INDEX3(0,2)]+2.0*A[INDEX3(1,1)]*A[INDEX3(2,2)]*B[INDEX3(0,0)]-
                      A[INDEX3(1,1)]*B[INDEX3(0,2)]*A[INDEX3(0,2)]-2.0*A[INDEX3(1,2)]*B[INDEX3(0,0)]*A[INDEX3(2,1)]+
                      B[INDEX3(0,1)]*A[INDEX3(0,2)]*A[INDEX3(2,1)]-B[INDEX3(0,1)]*A[INDEX3(0,1)]*A[INDEX3(2,2)]-
                      A[INDEX3(1,0)]*A[INDEX3(2,2)]*B[INDEX3(0,1)]+A[INDEX3(2,0)]*A[INDEX3(1,2)]*B[INDEX3(0,1)]+
                      A[INDEX3(1,0)]*B[INDEX3(0,2)]*A[INDEX3(2,1)]-B[INDEX3(0,2)]*A[INDEX3(2,0)]*A[INDEX3(1,1)])/denom;
              
            C[INDEX3(0,1)] = s.x*(A[INDEX3(0,2)]*B[INDEX3(0,0)]*A[INDEX3(2,1)]-A[INDEX3(2,2)]*B[INDEX3(0,0)]*A[INDEX3(0,1)]+
                      2.0*A[INDEX3(0,0)]*A[INDEX3(2,2)]*B[INDEX3(0,1)]-A[INDEX3(0,0)]*B[INDEX3(0,2)]*A[INDEX3(1,2)]-
                      2.0*A[INDEX3(2,0)]*A[INDEX3(0,2)]*B[INDEX3(0,1)]+B[INDEX3(0,2)]*A[INDEX3(1,0)]*A[INDEX3(0,2)]-
                      A[INDEX3(2,2)]*A[INDEX3(1,0)]*B[INDEX3(0,0)]+A[INDEX3(2,0)]*B[INDEX3(0,0)]*A[INDEX3(1,2)]+
                      A[INDEX3(2,0)]*B[INDEX3(0,2)]*A[INDEX3(0,1)]-B[INDEX3(0,2)]*A[INDEX3(0,0)]*A[INDEX3(2,1)])/denom;
             
            C[INDEX3(0,2)] = s.x*(A[INDEX3(0,1)]*A[INDEX3(1,2)]*B[INDEX3(0,0)]-A[INDEX3(0,2)]*B[INDEX3(0,0)]*A[INDEX3(1,1)]-
                      A[INDEX3(0,0)]*A[INDEX3(1,2)]*B[INDEX3(0,1)]+A[INDEX3(1,0)]*A[INDEX3(0,2)]*B[INDEX3(0,1)]-
                      B[INDEX3(0,1)]*A[INDEX3(0,0)]*A[INDEX3(2,1)]-A[INDEX3(2,0)]*A[INDEX3(1,1)]*B[INDEX3(0,0)]+
                      2.0*A[INDEX3(1,1)]*A[INDEX3(0,0)]*B[INDEX3(0,2)]-2.0*A[INDEX3(1,0)]*B[INDEX3(0,2)]*A[INDEX3(0,1)]+
                      A[INDEX3(1,0)]*A[INDEX3(2,1)]*B[INDEX3(0,0)]+A[INDEX3(2,0)]*B[INDEX3(0,1)]*A[INDEX3(0,1)])/denom;
            
            C[INDEX3(1,0)] = s.y*(-A[INDEX3(1,1)]*B[INDEX3(1,2)]*A[INDEX3(0,2)]+2.0*A[INDEX3(1,1)]*A[INDEX3(2,2)]*B[INDEX3(1,0)]+
                      A[INDEX3(1,2)]*A[INDEX3(0,1)]*B[INDEX3(1,2)]-2.0*A[INDEX3(1,2)]*B[INDEX3(1,0)]*A[INDEX3(2,1)]+
                      B[INDEX3(1,1)]*A[INDEX3(0,2)]*A[INDEX3(2,1)]-B[INDEX3(1,1)]*A[INDEX3(0,1)]*A[INDEX3(2,2)]-
                      A[INDEX3(1,0)]*A[INDEX3(2,2)]*B[INDEX3(1,1)]+A[INDEX3(2,0)]*A[INDEX3(1,2)]*B[INDEX3(1,1)]-
                      B[INDEX3(1,2)]*A[INDEX3(2,0)]*A[INDEX3(1,1)]+A[INDEX3(1,0)]*B[INDEX3(1,2)]*A[INDEX3(2,1)])/denom;
            
            C[INDEX3(1,1)] = s.y*(A[INDEX3(0,2)]*B[INDEX3(1,0)]*A[INDEX3(2,1)]-A[INDEX3(0,1)]*A[INDEX3(2,2)]*B[INDEX3(1,0)]+
                      2.0*A[INDEX3(2,2)]*A[INDEX3(0,0)]*B[INDEX3(1,1)]-B[INDEX3(1,2)]*A[INDEX3(0,0)]*A[INDEX3(1,2)]-
                      2.0*A[INDEX3(2,0)]*B[INDEX3(1,1)]*A[INDEX3(0,2)]-A[INDEX3(1,0)]*A[INDEX3(2,2)]*B[INDEX3(1,0)]+
                      A[INDEX3(2,0)]*A[INDEX3(1,2)]*B[INDEX3(1,0)]+A[INDEX3(1,0)]*B[INDEX3(1,2)]*A[INDEX3(0,2)]-
                      A[INDEX3(0,0)]*B[INDEX3(1,2)]*A[INDEX3(2,1)]+B[INDEX3(1,2)]*A[INDEX3(0,1)]*A[INDEX3(2,0)])/denom;
            
            C[INDEX3(1,2)] = s.y*(A[INDEX3(0,1)]*A[INDEX3(1,2)]*B[INDEX3(1,0)]-A[INDEX3(0,2)]*B[INDEX3(1,0)]*A[INDEX3(1,1)]-
                      A[INDEX3(0,0)]*A[INDEX3(1,2)]*B[INDEX3(1,1)]+A[INDEX3(1,0)]*A[INDEX3(0,2)]*B[INDEX3(1,1)]+
                      2.0*A[INDEX3(1,1)]*A[INDEX3(0,0)]*B[INDEX3(1,2)]-A[INDEX3(0,0)]*B[INDEX3(1,1)]*A[INDEX3(2,1)]+
                      A[INDEX3(0,1)]*A[INDEX3(2,0)]*B[INDEX3(1,1)]-B[INDEX3(1,0)]*A[INDEX3(2,0)]*A[INDEX3(1,1)]-
                      2.0*A[INDEX3(1,0)]*A[INDEX3(0,1)]*B[INDEX3(1,2)]+A[INDEX3(1,0)]*B[INDEX3(1,0)]*A[INDEX3(2,1)])/denom;
            
            C[INDEX3(2,0)] = s.z*(-A[INDEX3(1,1)]*A[INDEX3(0,2)]*B[INDEX3(2,2)]+A[INDEX3(0,1)]*A[INDEX3(1,2)]*B[INDEX3(2,2)]+
                      2.0*A[INDEX3(1,1)]*B[INDEX3(2,0)]*A[INDEX3(2,2)]-A[INDEX3(0,1)]*B[INDEX3(2,1)]*A[INDEX3(2,2)]+
                      A[INDEX3(0,2)]*A[INDEX3(2,1)]*B[INDEX3(2,1)]-2.0*B[INDEX3(2,0)]*A[INDEX3(2,1)]*A[INDEX3(1,2)]-
                      A[INDEX3(1,0)]*B[INDEX3(2,1)]*A[INDEX3(2,2)]+A[INDEX3(1,2)]*A[INDEX3(2,0)]*B[INDEX3(2,1)]-
                      A[INDEX3(1,1)]*A[INDEX3(2,0)]*B[INDEX3(2,2)]+A[INDEX3(2,1)]*A[INDEX3(1,0)]*B[INDEX3(2,2)])/denom;
            
            C[INDEX3(2,1)] = s.z*-(A[INDEX3(0,1)]*A[INDEX3(2,2)]*B[INDEX3(2,0)]-A[INDEX3(0,2)]*B[INDEX3(2,0)]*A[INDEX3(2,1)]-
                       2.0*B[INDEX3(2,1)]*A[INDEX3(0,0)]*A[INDEX3(2,2)]+A[INDEX3(1,2)]*B[INDEX3(2,2)]*A[INDEX3(0,0)]+
                       2.0*B[INDEX3(2,1)]*A[INDEX3(0,2)]*A[INDEX3(2,0)]+A[INDEX3(1,0)]*B[INDEX3(2,0)]*A[INDEX3(2,2)]-
                       A[INDEX3(1,0)]*A[INDEX3(0,2)]*B[INDEX3(2,2)]-A[INDEX3(1,2)]*A[INDEX3(2,0)]*B[INDEX3(2,0)]+
                       A[INDEX3(0,0)]*B[INDEX3(2,2)]*A[INDEX3(2,1)]-B[INDEX3(2,2)]*A[INDEX3(0,1)]*A[INDEX3(2,0)])/denom;
             
            C[INDEX3(2,2)] = s.z*(A[INDEX3(0,1)]*A[INDEX3(1,2)]*B[INDEX3(2,0)]-A[INDEX3(0,2)]*B[INDEX3(2,0)]*A[INDEX3(1,1)]-
                      A[INDEX3(0,0)]*A[INDEX3(1,2)]*B[INDEX3(2,1)]+A[INDEX3(1,0)]*A[INDEX3(0,2)]*B[INDEX3(2,1)]-
                      A[INDEX3(1,1)]*A[INDEX3(2,0)]*B[INDEX3(2,0)]-A[INDEX3(2,1)]*B[INDEX3(2,1)]*A[INDEX3(0,0)]+
                      2.0*A[INDEX3(1,1)]*B[INDEX3(2,2)]*A[INDEX3(0,0)]+A[INDEX3(2,1)]*A[INDEX3(1,0)]*B[INDEX3(2,0)]+
                      A[INDEX3(2,0)]*A[INDEX3(0,1)]*B[INDEX3(2,1)]-2.0*B[INDEX3(2,2)]*A[INDEX3(1,0)]*A[INDEX3(0,1)])/denom;
            }
    };


#endif // __EvaluatorPairGayBerne_H_

