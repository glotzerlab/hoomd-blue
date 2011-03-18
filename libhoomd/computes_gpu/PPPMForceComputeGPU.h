/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
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

// $Id$
// $URL$
// Maintainer: sbarr

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <cufft.h>
#include "PPPMForceCompute.h"
#include "PPPMForceGPU.cuh"
#include "NeighborList.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file PPPMForceComputeGPU.h
    \brief Declares the PPPMForceGPU class
*/

#ifndef __PPPMFORCECOMPUTEGPU_H__
#define __PPPMFORCECOMPUTEGPU_H__

//! Implements the harmonic bond force calculation on the GPU
/*! PPPMForceComputeGPU implements the same calculations as PPPMForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as float2's with the \a x component being K and the
    \a y component being r_0.

    The GPU kernel can be found in bondforce_kernel.cu.

    \ingroup computes
*/
class PPPMForceComputeGPU : public PPPMForceCompute
    {
    public:
        //! Constructs the compute
        PPPMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                            boost::shared_ptr<NeighborList> nlist,
                            boost::shared_ptr<ParticleGroup> group);
        //! Destructor
        ~PPPMForceComputeGPU();
        
        //! Sets the block size to run on the device
        /*! \param block_size Block size to set
         */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
            
        //! Set the parameters
        virtual void setParams(int Nx, int Ny, int Nz, int order, Scalar kappa, Scalar rcut);


    protected:
        int m_block_size;                    //!< Block size to run calculation on
        cufftHandle plan;                    //!< Used for the Fast Fourier Transformations performed on the GPU                   

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_PPPMForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

