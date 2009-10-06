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


#ifndef __INDEX__TRANSFORM__
#define __INDEX__TRANSFORM__

/*! \file IndexTransform.h
	\brief Maps n dimensional indices to m dimensional ones

*/

//! The purpose of this class is to avoid errors by defining a unique function
/*! In many instances it is necessary to transform n dimensional indices to m dimensional ones
    A typical example is a 3D matrix that needs to be stored as 1D one, and thus
	(i,j,k) indices need to be transformed to a 1D ind. 
	Other transformations may be required
	\note There is some freedom in how to map 3D indices to 1D indices, here we follow the 
	formula that elements are stored in row major order, that is (i,j,k) maps into
	   k+N_3*j+N_3*N_2*i
    where (N_1,N_2,N_3) are the number of grid points along 1,2,3. Caution must be taken as 
	a convention using minor order is also common practice.
*/

class IndexTransform
    {
	public:
			IndexTransform(void);   //!< void constructor
			~IndexTransform(void); //!< destructor
			void SetD3to1D(unsigned int N_1,unsigned int N_2,unsigned int N_3); //Set the grid
			unsigned int D3To1D(unsigned int i,unsigned int j,unsigned int k) const; //!< converts three dimensional index i,j,k to a single integer index
    // Only 3d to 1D is defined so far

	private:
			unsigned int N_1;	//!< Number of grid points along the 1 axis
			unsigned int N_2;	//!< Number of grid points along the 2 axis
			unsigned int N_3;	//!< Number of grid points along the 3 axis
};
#endif
