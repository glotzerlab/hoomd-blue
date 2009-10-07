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

// $Id$
// $URL$
// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "FENEBondForceCompute.h"
#include "FENEBondForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file FENEBondForceComputeGPU.h
    \brief Declares the FENEBondForceGPU class
*/

#ifndef __FENEBONDFORCECOMPUTEGPU_H__
#define __FENEBONDFORCECOMPUTEGPU_H__

//! Implements the fene bond force calculation on the GPU
/*! FENEBondForceComputeGPU implements the same calculations as FENEBondForceCompute,
    but executing on the GPU.

    The calculation on the GPU is structured after that used in HarmonicBondForceCompute.
    See its documentation for more implementation details.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as float4's with the \a x component set to  K, the
    \a y component set to r_0, the z component set to epsilon and the \a w component
    set to sigma.

    The GPU computation is implemented in fenebondforce_kernel.cu

    \ingroup computes
*/
class FENEBondForceComputeGPU : public FENEBondForceCompute
    {
    public:
        //! Constructs the compute
        FENEBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef);
        //! Destructor
        ~FENEBondForceComputeGPU();
        
        //! Sets the block size to run on the device
        /*! \param block_size Block size to set
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
            
        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon);
        
    protected:
        int m_block_size;                   //!< Block size to run calculation on
        std::vector<float4 *> m_gpu_params; //!< Parameters stored on the GPU
        std::vector<int *> m_checkr;        //!< Error flag stored on the GPU
        float4 *m_host_params;              //!< Host parameters
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_FENEBondForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

