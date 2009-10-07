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


#ifndef __FFTW__WRAPPER__
#define __FFTW__WRAPPER__

/*! \file FftwWrapper.h
    \brief Adapts fftw for use in HOOMD
*/

#ifdef ENABLE_FFTW

#include "FFTClass.h"
#include "fftw3.h"
#include "IndexTransform.h"

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif

//! Implements fftw for HOOMD
/*! fftw stands for fastest fourier transform in the west, see http://www.fftw.org/ for
    proper documentation and details.
    The fftw used in this class is done in double precision.
    \note Due to licensing conflicts fftw cannot distributed with HOOMD.
    \todo After putting in placeholder documentation in FFTClass, I'm not doing so again
    it is the responsibility of the author to properly document the code so I'm not fixing it.
*/
class FftwWrapper:public FFTClass
    {
    public:
        FftwWrapper();   //!< void constructor
        FftwWrapper(unsigned int N_x,unsigned int N_y,unsigned int N_z); //!< constructor
        ~FftwWrapper(void); //!< destructor
        
        //! Redefines the class if constructed by void constructor
        void fftw_define(unsigned int N_x,unsigned int N_y,unsigned int N_z);
        
        //! fft transform complex to complex
        void cmplx_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar *Dat_in,CScalar *Dat_out,int sig);
        
        //! fft transform real to complex
        void real_to_compl_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,Scalar *Dat_in,CScalar *Dat_out);
        
        //! fft transform complex to real
        void compl_to_real_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar *Dat_in,Scalar *Dat_out);
        
    private:
        unsigned int N_x;   //!< Number of grid points along the x axis
        unsigned int N_y;   //!< Number of grid points along the y axis
        unsigned int N_z;   //!< Number of grid points along the z axis
        IndexTransform T;   //!< defines how 3d coordinates are mapped into 1D
        fftw_complex *in_f;     //!< data structure required as input for fftw to perform forward fft
        fftw_complex *out_f;    //!< data structure required as output for fftw to perform forward fft
        fftw_complex *in_b;     //!< data structure required as input for fftw to perfrom backward fft
        fftw_complex *out_b;    //!< data structure required as output for fftw to perfrom backaward fft
        fftw_plan p_forward;    //!< Defines the structure by which fftw will do the fftw forward (see fftw documentation)
        fftw_plan p_backward;   //!< Defines the structure by which fftw will do the fftw backward (see fftw documentation)
        bool plan_is_defined;   //!< logical variable to prevent redefining a plan once one is already defined
        double Initial_Conf_real(double x,double y); //!< Defines the real part of an initial configuration
        double Initial_Conf_Imag(double x,double y); //!< Defines the imag part of an initial configuration
        
    };
#endif
#endif

