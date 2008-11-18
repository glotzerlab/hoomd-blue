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
*/

class IndexTransform
    {
	public:
			IndexTransform(void);   //!< void constructor
			~IndexTransform(void); //!< destructor
			void SetD3to1D(unsigned int N_x,unsigned int N_y,unsigned int N_z); //Set the grid
			unsigned int D3To1D(unsigned int i,unsigned int j,unsigned int k) const; //!< converts three dimensional index i,j,k to a single integer index
    // Only 3d to 1D is defined so far

	private:
			unsigned int N_x;	//!< Number of grid points along the x axis
			unsigned int N_y;	//!< Number of grid points along the y axis
			unsigned int N_z;	//!< Number of grid points along the z axis
};
#endif
