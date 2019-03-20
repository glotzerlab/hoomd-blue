// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ConstraintEllipsoid.h"

/*! \file ConstraintEllipsoidGPU.h
    \brief Declares a class for computing ellipsoid constraint forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CONSTRAINT_ELLIPSOID_GPU_H__
#define __CONSTRAINT_ELLIPSOID_GPU_H__

//! Applys a constraint force to keep a group of particles on a Ellipsoid
/*! \ingroup computes
*/
class PYBIND11_EXPORT ConstraintEllipsoidGPU : public ConstraintEllipsoid
    {
    public:
        //! Constructs the compute
        ConstraintEllipsoidGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         Scalar3 P,
                         Scalar rx,
                         Scalar ry,
                         Scalar rz);

    protected:
        unsigned int m_block_size;  //!< block size to execute on the GPU

        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Exports the ConstraintEllipsoidGPU class to python
void export_ConstraintEllipsoidGPU(pybind11::module& m);

#endif
