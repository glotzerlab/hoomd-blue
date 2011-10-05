/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "HarmonicBondForceCompute.h"
#include "HarmonicBondForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file HarmonicBondForceComputeGPU.h
    \brief Declares the HarmonicBondForceGPU class
*/

#ifndef __HARMONICBONDFORCECOMPUTEGPU_H__
#define __HARMONICBONDFORCECOMPUTEGPU_H__

//! Implements the harmonic bond force calculation on the GPU
/*! HarmonicBondForceComputeGPU implements the same calculations as HarmonicBondForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as float2's with the \a x component being K and the
    \a y component being r_0.

    The GPU kernel can be found in bondforce_kernel.cu.

    \ingroup computes
*/
class HarmonicBondForceComputeGPU : public HarmonicBondForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, const std::string& log_suffix="");
        //! Destructor
        ~HarmonicBondForceComputeGPU();
        
        //! Sets the block size to run on the device
        /*! \param block_size Block size to set
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
            
        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar r_0);
        
    protected:
        int m_block_size;            //!< Block size to run calculation on
        GPUArray<float2>  m_params;  //!< Parameters stored on the GPU
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_HarmonicBondForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

