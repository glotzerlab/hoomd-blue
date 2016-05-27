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

// Maintainer: mspells

#include "hoomd/GPUArray.h"
#include "hoomd/md/NeighborList.h"

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "DEM3DForceCompute.h"

/*! \file DEM3DForceComputeGPU.h
  \brief Declares the class DEM3DForceComputeGPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM3DFORCECOMPUTEGPU_H__
#define __DEM3DFORCECOMPUTEGPU_H__

#ifdef ENABLE_CUDA

//! Computes DEM3D forces on each particle using the GPU
/*! Calculates the same forces as DEM3DForceCompute, but on the GPU.

  The GPU kernel for calculating the forces is in DEM3DForceGPU.cu.
  \ingroup computes
*/
template<typename Real, typename Real4, typename Potential>
class DEM3DForceComputeGPU: public DEM3DForceCompute<Real, Real4, Potential>
    {
    public:
        //! Constructs the compute
        DEM3DForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
            boost::shared_ptr<NeighborList> nlist,
            Real r_cut, Potential potential);

        //! Destructor
        virtual ~DEM3DForceComputeGPU();

        //! Set parameters for the builtin autotuner
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Find the maximum number of GPU threads (2*vertices + edges) among all shapes
        size_t maxGPUThreads() const;

    protected:
        boost::scoped_ptr<Autotuner> m_tuner;     //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

#include "DEM3DForceComputeGPU.cc"

#endif

#endif
