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

#ifndef __FFT_CLASS_H__
#define __FFT_CLASS_H__

#ifdef USE_FFT

#include "ParticleData.h"

/*! \file fft_class.h
	\brief Defines an abstract interface for performing a fast fourier transform
*/

//! provides a general interface for fft in HOOMD
/*! The three member functions are respectively, the 3D fft of a complex matrix, the fft of
    a real matrix, and the fft of a complex matrix whose fft is a real matrix. The data 
	types are three dimensional matrices.
	\note This class is abstract and therefore cannot be instantiated.	     
*/


class fft_class{
	virtual void cmplx_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar ***in,CScalar ***out,int sig)=0;	//!< 3D FFT of complex matrix in, result stored in matrix out, sign=-1 (forward) or +1 (backward)
	virtual void real_to_compl_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,Scalar ***in,CScalar ***out)=0;    //!< 3D FFT of real matrix in, result stored in matrix out, forward is implictly assumed
	virtual void compl_to_real_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar ***in,Scalar ***out)=0;    //!< 3D FFT of complex matrix in, result is real and stored in matrix out, backward is implictly assumed
};

#endif

#endif
