// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ConstraintSphere.h"

/*! \file ConstraintSphereGPU.h
    \brief Declares a class for computing sphere constraint forces on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CONSTRAINT_SPHERE_GPU_H__
#define __CONSTRAINT_SPHERE_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere on the GPU
/*! \ingroup computes
*/
class PYBIND11_EXPORT ConstraintSphereGPU : public ConstraintSphere
    {
    public:
        //! Constructs the compute
        ConstraintSphereGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            Scalar3 P,
                            Scalar r);

    protected:
        unsigned int m_block_size;  //!< block size to execute on the GPU

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the ConstraintSphereGPU class to python
void export_ConstraintSphereGPU(pybind11::module& m);

#endif
