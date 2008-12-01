/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

/*! \file ElectrostaticLongRangePPPM.h
	\brief Declares a class for computing the PPPM long range electrostatic force
*/


//! Long-range electrostatic force by the PPPM method
/*! In order for this class to be useful it needs to be complemented with a class that computes the short-range electrostatic part
    The total pair force is summed for each particle when compute() is called.
	This class requires the use of Fast Fourier Transforms, which are defined somewhere else
    
	Usage: Construct a ElectrostaticLongRangePPM class, providing it an already constructed ParticleData.
	The parameter alpha splits the short range and the long range electrostatic part.
    
	Details on how the parameter alpha is defined are found in
	"How to mesh up Ewald sums.I. A theoretical and numerical comparison of various particle mesh routines"
	M. Deserno and C. Holm
	J. Chem. Phys. 109, 7678 (1998).

	\note This class does not compute the parameter alpha, it uses the alpha as specified. 
	
	Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but 
	a more typical usage will be to add the force compute to NVEUpdator or NVTUpdator. 
	
	This base class defines the interface for performing the long-range part of the electrostatic force 
	computations using the PPPM method. It does provide a functional, single threaded method for computing the forces. 
	
	\ingroup computes
*/

#include <boost/shared_ptr.hpp>
#include <vector>
using namespace std;
#include "ForceCompute.h"

#ifndef __ELECTROSTATICLONGRANGEPPPM_H__
#define __ELECTROSTATICLONGRANGEPPPM_H__

using namespace std;

#ifdef USE_FFT
#include "FFTClass.h"
#include "IndexTransform.h"

//! Long range electrostatic forces via PPPM
/*! \ingroup computes
	\todo document me!
*/
class ElectrostaticLongRangePPPM : public ForceCompute
	{
	public:
		//!< Constructs the compute
		ElectrostaticLongRangePPPM(boost::shared_ptr<ParticleData> pdata,unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z, unsigned int P_order_a, Scalar alpha,boost::shared_ptr<FFTClass> FFTP,bool third_law_m);
		
		//!< Destructor
		virtual ~ElectrostaticLongRangePPPM();

		//!< function making the charge assignment
		virtual void make_rho(void);   

		//!< Delivers the charge density on the grid, convenient for unit testing
		const CScalar & Show_rho_real(unsigned int ix,unsigned int iy,unsigned int iz) const;

		//!< Show the value of the polynomial coefficients for charge distribution, convenient for unit testing
		const Scalar & Poly_coeff_Grid(unsigned int i,unsigned int j) const;

		//!< Show the value of the polynomial coefficients in the denominator of the Influence function, convenient for unit testing
		const Scalar & Poly_coeff_Denom_Influence_Function(unsigned int i) const;

		//!< Delivers the Influence function, convenient for unit testing
		const Scalar & Influence_function(unsigned int ix,unsigned int iy,unsigned int iz) const;
		
		//!< Delivers number of mesh points in the x axis, convenient for unit testing
		unsigned int N_mesh_x_axis(void) const;

		//!< Delivers number of mesh points in the y axis, convenient for unit testing
		unsigned int N_mesh_y_axis(void) const;

		//!< Delivers number of mesh points in the z axis, convenient for unit testing
		unsigned int N_mesh_z_axis(void) const;

	protected:
		Scalar m_alpha;              //!< split parameter of the short-range vs long-range forces.
		unsigned int N_mesh_x;       //!< number of points in the mesh along the x axis
		unsigned int N_mesh_y;       //!< number of points in the mesh along the y axis
		unsigned int N_mesh_z;       //!< number of points in the mesh along the z axis
		int Nu_mesh_x;				 //!< number of points in the mesh along the x axis
		int Nu_mesh_y;				 //!< number of points in the mesh along the y axis
		int Nu_mesh_z;				 //!< number of points in the mesh along the z axis
		IndexTransform T;            //!< convert 3d coordinates to 1D
		unsigned int P_order;        //!< The charge assignment on the mesh is of order P
		BoxDim box;                  //!< Copy of the simulation box
		boost::shared_ptr<FFTClass> FFT; //!< Fast Fourier Transform pointer
		bool third_law;              //!< Whether to use third law or not
		Scalar S_mesh_x;             //!< number of points in the mesh along the x axis as a scalar
		Scalar S_mesh_y;             //!< number of points in the mesh along the y axis as a scalar
		Scalar S_mesh_z;             //!< number of points in the mesh along the z axis as a scalar
		Scalar Lx;                   //!< Length size of the system along the x direction
		Scalar Ly;                   //!< Length size of the system along the y direction
		Scalar Lz;                   //!< Length size of the system along the z direction
		Scalar h_x;                  //!< Spacing along the x axis
		Scalar h_y;                  //!< Spacing along the y axis
		Scalar h_z;                  //!< Spacing along the z axis
		CScalar *rho_real;          //!< density of charge on the mesh
		CScalar *rho_kspace;        //!< density of charge in Fourier space
		Scalar  *G_Inf;              //!< Precomputed proximity function
		CScalar *fx_kspace;         //!< force in k-space, x component
		CScalar *fy_kspace;         //!< force in k-space, y component
		CScalar *fz_kspace;         //!< force in k-space, z component
		CScalar *e_kspace;          //!< Energy in k-space
		CScalar *v_kspace;          //!< virial in k-space
		CScalar *fx_real;           //!< force on mesh points, x component
		CScalar *fy_real;           //!< force on mesh points, y component
		CScalar *fz_real;           //!< force on mesh points, z component
		CScalar *e_real;            //!< Energy on mesh points 
		CScalar *v_real;            //!< virial on mesh points 
		Scalar **P_coeff;           //!< Coefficients of Polynomial to distribute charge
		double **a_P;				//!< Pointer necessary to determine P_coeff
		double **b_P;               //!< Pointer necessary to determine P_coeff
		void (ElectrostaticLongRangePPPM::*make_rho_helper)(void); //!<function pointer used to hide implementation details
		void (ElectrostaticLongRangePPPM::*back_interpolate_helper)(CScalar *,Scalar *); //!<function to pointer to ide implementation details
		Scalar *Denom_Coeff;         //!< Coefficients of the polynomial to compute the denominator of the influence function
		
		
		virtual Scalar Poly(int n,Scalar x); //!< Polynomial that computes the fraction of charge at point x
		virtual void back_interpolate(CScalar *Grid,Scalar *Continuum); //!< Backinterpolate mesh grid points into continuum for P even
		
		virtual void make_rho_even(void);    //!<Distribute charges on the mesh when P_order is even
		virtual void make_rho_odd(void);     //!<Distribute charges on the mesh when P_order is odd
		virtual void back_interpolate_even(CScalar *Grid,Scalar *Continuum);//!< Backinterpolate mesh grid points into continuum for P even
		virtual void back_interpolate_odd(CScalar *Grid,Scalar *Continuum);//!< Backinterpolate mesh grid points into continuum for P odd

		virtual void ComputePolyCoeff(void); //!<Compute the coefficients of the Polynomial (encoded in P_coeff)
		
		virtual void Compute_G(void);        //!<Compute the influence function
		virtual Scalar Denominator_G(Scalar x,Scalar y,Scalar z); //!< Denominator of the influence function (without the derivative square)
		virtual vector<Scalar> Numerator_G(Scalar x,Scalar y,Scalar z); //!< Numerator of the influence function (without the derivative);
		void Denominator_Poly_G(void);       //!<Compute the coefficients of the Polynomial Denom_Coeff

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
		};
	

#endif
#endif


