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

// conditionally compile only if fast fourier transforms are defined

#ifdef USE_FFT

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"

/*! \file ElectrostaticLongRangePPPM.h
	\brief Declares a class for computing the short range part of the electrostatic force and energy
*/

#ifndef __ELECTROSTATICLONGRANGEPPPM_H__
#define __ELECTROSTATICLONGRANGEPPPM_H__

//! Computes the Long-range electrostatic part of the force by using the PPPM approximation
/*! In order for this class to be useful it needs to be complemented with a class that computes the short-range electrostatic part
    The total pair force is summed for each particle when compute() is called.
	This class requires the use of Fast Fourier Transforms, which are defined somewhere else
    
	Usage: Construct a ElectrostaticLongRangePPM class, providing it an already constructed ParticleData.
	The parameter alpha splits the short range and the long range electrostatic part.
    
	Details on how the parameter alpha is defined are found in
	"How to mesh up Ewald sums.I. A theoretical and numerical comparison of various particle mesh routines"
	M. Deserno and C. Holm
	J. Chem. Phys. 109, 7678 (1998).

	NOTE: This class does not compute the parameter alpha, it uses the alpha as specified. 
	
	Forces can be computed directly by calling compute() and then retrieved with a call to acquire(), but 
	a more typical usage will be to add the force compute to NVEUpdator or NVTUpdator. 
	
	This base class defines the interface for performing the long-range part of the electrostatic force 
	computations using the PPPM method. It does provide a functional, single threaded method for computing the forces. 

*/
class ElectrostaticLongRangePPPM : public ForceCompute
	{
	public:
		//! Constructs the compute
		ElectrostaticLongRangePPPM(boost::shared_ptr<ParticleData> pdata,unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z, unsigned int P_order_a, Scalar alpha,bool third_law_m);
		
		//! Destructor
		virtual ~ElectrostaticLongRangePPPM();

	protected:
	    Scalar m_alpha;              //!< split parameter of the short-range vs long-range forces.
	    unsigned int N_mesh_x;       //!< number of points in the mesh along the x axis
	    unsigned int N_mesh_y;       //!< number of points in the mesh along the y axis
	    unsigned int N_mesh_z;       //!< number of points in the mesh along the z axis
	    unsigned int P_order;        //!< The charge assignment on the mesh is of order P
	    BoxDim box;                  //!< Copy of the simulation box
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
	    Scalar ***rho_real;          //!< density of charge on the mesh
	    Scalar ***rho_kspace;        //!< density of charge in fourier space
	    Scalar ***G_Inf;             //!< Precomputed proximity function
		void (ElectrostaticLongRangePPPM::*make_rho_helper)(void); //!<function pointer used to hide implementation details
		
		virtual void make_rho_even(void);    //!<Distribute charges on the mesh when P_order is even
		virtual void make_rho_odd(void);     //!<Distribute charges on the mesh when P_order is odd
		virtual void make_rho(void);         //!<Actual function making the charge assignment
		virtual Scalar Poly(int n,Scalar x); //<!Polynomial that computes the fraction of charge at point x

       	virtual void computeForces(unsigned int timestep); //! Actually compute the forces
	};
	

#endif
#endif


