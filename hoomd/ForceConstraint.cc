// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ForceConstraint.h"

namespace py = pybind11;

using namespace std;

/*! \file ForceConstraint.cc
    \brief Contains code for the ForceConstraint class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceConstraint::ForceConstraint(std::shared_ptr<SystemDefinition> sysdef)
        : ForceCompute(sysdef)
    {
    }

/*! Does nothing in the base class
    \param timestep Current timestep
*/
void ForceConstraint::computeForces(unsigned int timestep)
    {
    }


void export_ForceConstraint(py::module& m)
    {
    py::class_< ForceConstraint, std::shared_ptr<ForceConstraint> >(m,"ForceConstraint",py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
