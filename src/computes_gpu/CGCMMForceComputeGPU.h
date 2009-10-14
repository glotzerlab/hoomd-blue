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
        vector<float4 *> d_coeffs;      //!< Pointer to the coefficients on the GPU
        float4 * h_coeffs;              //!< Pointer to the coefficients on the host
        int m_block_size;               //!< The block size to run on the GPU
        bool m_ulf_workaround;          //!< Stores decision made by the constructor whether to enable the ULF workaround
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the CGCMMForceComputeGPU class to python
void export_CGCMMForceComputeGPU();

#endif

