// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveForceCompute.h"

/*! \file ActiveForceComputeGPU.h
    \brief Declares a class for computing active forces on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ACTIVEFORCECOMPUTE_GPU_H__
#define __ACTIVEFORCECOMPUTE_GPU_H__

//! Adds an active force to a number of particles on the GPU
/*! \ingroup computes
*/
class PYBIND11_EXPORT ActiveForceComputeGPU : public ActiveForceCompute
    {
    public:
        //! Constructs the compute
        ActiveForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             int seed, pybind11::list f_lst, pybind11::list t_lst,
                             bool orientation_link, bool orientation_reverse_link, Scalar rotation_diff,
                             Scalar3 P,
                             Scalar rx,
                             Scalar ry,
                             Scalar rz);

    protected:
        unsigned int m_block_size;  //!< block size to execute on the GPU

        //! Set forces for particles
        virtual void setForces();

        //! Orientational diffusion for spherical particles
        virtual void rotationalDiffusion(unsigned int timestep);

        //! Set constraints if particles confined to a surface
        virtual void setConstraint();

        GPUArray<unsigned int>  m_groupTags; //! Stores list converting group index to global tag
    };

//! Exports the ActiveForceComputeGPU Class to python
void export_ActiveForceComputeGPU(pybind11::module& m);
#endif
