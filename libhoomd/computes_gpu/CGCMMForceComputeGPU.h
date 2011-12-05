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

// Maintainer: akohlmey

#include "CGCMMForceCompute.h"
#include "NeighborList.h"
#include "CGCMMForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file CGCMMForceComputeGPU.h
    \brief Declares the class CGCMMForceComputeGPU
*/

#ifndef __CGCMMFORCECOMPUTEGPU_H__
#define __CGCMMFORCECOMPUTEGPU_H__

//! Computes CGCMM forces on each particle using the GPU
/*! Calculates the same forces as CGCMMForceCompute, but on the GPU.

    The GPU kernel for calculating the forces is in cgcmmforcesum_kernel.cu.
    \ingroup computes
*/
class CGCMMForceComputeGPU : public CGCMMForceCompute
    {
    public:
        //! Constructs the compute
        CGCMMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<NeighborList> nlist, Scalar r_cut);
        
        //! Destructor
        virtual ~CGCMMForceComputeGPU();
        
        //! Set the parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4);
        
        //! Sets the block size to run at
        void setBlockSize(int block_size);
        
    protected:
        GPUArray<float4>  m_coeffs;     //!< Coefficients for the force
        int m_block_size;               //!< The block size to run on the GPU
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the CGCMMForceComputeGPU class to python
void export_CGCMMForceComputeGPU();

#endif

