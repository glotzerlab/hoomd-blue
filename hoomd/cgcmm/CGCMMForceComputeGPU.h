// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "CGCMMForceCompute.h"
#include "hoomd/md/NeighborList.h"
#include "CGCMMForceGPU.cuh"

#include <memory>

/*! \file CGCMMForceComputeGPU.h
    \brief Declares the class CGCMMForceComputeGPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CGCMMFORCECOMPUTEGPU_H__
#define __CGCMMFORCECOMPUTEGPU_H__

//! Computes CGCMM forces on each particle using the GPU
/*! Calculates the same forces as CGCMMForceCompute, but on the GPU.

    The GPU kernel for calculating the forces is in cgcmmforcesum_kernel.cu.
    \ingroup computes
*/
class PYBIND11_EXPORT CGCMMForceComputeGPU : public CGCMMForceCompute
    {
    public:
        //! Constructs the compute
        CGCMMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist, Scalar r_cut);

        //! Destructor
        virtual ~CGCMMForceComputeGPU();

        //! Set the parameters for a single type pair
        virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4);

        //! Sets the block size to run at
        void setBlockSize(int block_size);

    protected:
        GPUArray<Scalar4>  m_coeffs;     //!< Coefficients for the force
        int m_block_size;               //!< The block size to run on the GPU

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();
    };

//! Exports the CGCMMForceComputeGPU class to python
void export_CGCMMForceComputeGPU(pybind11::module& m);

#endif
