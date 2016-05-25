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

#include "hoomd/Autotuner.h"
#include "hoomd/GPUArray.h"
#include "hoomd/md/NeighborList.h"

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "DEM2DForceCompute.h"
#include "DEM2DForceGPU.cuh"

/*! \file DEM2DForceComputeGPU.h
  \brief Declares the class DEM2DForceComputeGPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM2DFORCECOMPUTEGPU_H__
#define __DEM2DFORCECOMPUTEGPU_H__

#ifdef ENABLE_CUDA

//! Computes DEM2D forces on each particle using the GPU
/*! Calculates the same forces as DEM2DForceCompute, but on the GPU.

  The GPU kernel for calculating the forces is in DEM2DForceGPU.cu.
  \ingroup computes
*/
template<typename Real, typename Real2, typename Real4, typename Potential>
class DEM2DForceComputeGPU : public DEM2DForceCompute<Real, Real4, Potential>
{
public:
    //! Constructs the compute
    DEM2DForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         Scalar r_cut, Potential potential);

    //! Destructor
    virtual ~DEM2DForceComputeGPU();

    //! Set the vertices for a particle type
    virtual void setParams(unsigned int type,
                           const boost::python::list &vertices);

    //! Set parameters for the builtin autotuner
    virtual void setAutotunerParams(bool enable, unsigned int period)
    {
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
    }

protected:
    GPUArray<Real2> m_vertices;     //!< Vertices for all shapes
    GPUArray<unsigned int> m_num_shape_vertices;    //!< Number of vertices for each shape
    boost::scoped_ptr<Autotuner> m_tuner;     //!< Autotuner for block size

    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

    //! Re-send the list of vertices and links to the GPU
    void createGeometry();

    //! Find the total number of vertices in the current set of shapes
    size_t numVertices() const;

    //! Find the maximum number of vertices in the current set of shapes
    size_t maxVertices() const;
};

#include "DEM2DForceComputeGPU.cc"

#endif

#endif
