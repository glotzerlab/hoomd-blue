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

#include "IndexTransform.h"

/*! \file IndexTransform.cc
	\brief Implements the code for the  CoordinateTransform class
*/

//! basically defines the following parameters
	/*! \param N_1 number of grid points in the 1 axis
		\param N_2 number of grid points in the 2 axis
		\param N_3 number of grid points in the 3 axis
    */
IndexTransform::IndexTransform(void)
{
//constructor initializes to zero
	N_1=0;
	N_2=0;
	N_3=0;
}

IndexTransform::~IndexTransform()
{
// destructor does nothing
}
/*!				\param N1 number of grid points in the 1 axis
			    \param N2 number of grid points in the 2 axis
			    \param N3 number of grid points in the 3 axis
*/
void IndexTransform::SetD3to1D(unsigned int N1,unsigned int N2,unsigned int N3)
{
	N_1=N1;
	N_2=N2;
	N_3=N3;
}

unsigned int IndexTransform::D3To1D(unsigned int i,unsigned int j,unsigned int k) const
{
	return k+N_3*j+N_2*N_3*i;
}
