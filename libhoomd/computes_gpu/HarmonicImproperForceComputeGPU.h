/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "HarmonicImproperForceCompute.h"
#include "HarmonicImproperForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file HarmonicImproperForceComputeGPU.h
    \brief Declares the HarmonicImproperForceGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __HARMONICIMPROPERFORCECOMPUTEGPU_H__
#define __HARMONICIMPROPERFORCECOMPUTEGPU_H__

//! Implements the harmonic improper force calculation on the GPU
/*! HarmonicImproperForceComputeGPU implements the same calculations as HarmonicImproperForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as float2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in improperforce_kernel.cu.

    \ingroup computes
*/
class HarmonicImproperForceComputeGPU : public HarmonicImproperForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicImproperForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef);
        //! Destructor
        ~HarmonicImproperForceComputeGPU();
        
        //! Sets the block size to run on the device
        /*! \param block_size Block size to set
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
            
        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar chi);
        
    protected:
        int m_block_size;               //!< Block size to run calculation on
        GPUArray<float2>  m_params;     //!< Parameters stored on the GPU (k,chi)
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the ImproperForceComputeGPU class to python
void export_HarmonicImproperForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

