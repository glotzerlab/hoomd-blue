// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: grva

#include "hoomd/ForceCompute.h"

#include <memory>

/*! \file ConstExternalFieldDipoleForceCompute.h
    \brief Declares a class for computing external forces on anisotropic particles
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CONSTEXTERNALFIELDDIPOLEFORCECOMPUTE_H__
#define __CONSTEXTERNALFIELDDIPOLEFORCECOMPUTE_H__

//! Adds the force of a constant external field on a dipole for each particle
/*! \ingroup computes
*/
class PYBIND11_EXPORT ConstExternalFieldDipoleForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        ConstExternalFieldDipoleForceCompute(std::shared_ptr<SystemDefinition> sysdef, Scalar field_x,Scalar field_y, Scalar field_z,Scalar p);

        //! Set the force to a new value
        void setParams(Scalar field_x,Scalar field_y, Scalar field_z,Scalar p);

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    private:
    Scalar4 field;  //!< Electric field
    };

//! Exports the ConstExternalFieldDipoleForceComputeClass to python
void export_ConstExternalFieldDipoleForceCompute(pybind11::module& m);

#endif
