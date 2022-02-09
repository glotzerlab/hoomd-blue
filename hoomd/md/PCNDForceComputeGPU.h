// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "PCNDForceCompute.h"
#include "hoomd/md/NeighborList.h"
#include "PCNDForceGPU.cuh"

#include <memory>

/*! \file PCNDForceComputeGPU.h
    \brief Declares the class PCNDForceComputeGPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __PCNDFORCECOMPUTEGPU_H__
#define __PCNDFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes PCND forces on each particle using the GPU
/*! Calculates the same forces as PCNDForceCompute, but on the GPU.

    The GPU kernel for calculating the forces is in pcndforcesum_kernel.cu.
    \ingroup computes
*/
class PCNDForceComputeGPU : public PCNDForceCompute
    {
    public:
        //! Constructs the compute
        PCNDForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, std::shared_ptr<NeighborList> nlist, Scalar r_cut);

        //! Destructor
        virtual ~PCNDForceComputeGPU();

        //! Set the parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4);

        //! Sets the block size to run at
        void setBlockSize(int block_size);

    protected:
        GPUArray<Scalar4>  m_coeffs;     //!< Coefficients for the force
        int m_block_size;               //!< The block size to run on the GPU

        //! Actually compute the forces
        virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the PCNDForceComputeGPU class to python
void export_PCNDForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
#endif
