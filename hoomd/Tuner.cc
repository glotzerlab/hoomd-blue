// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Tuner.h"

/*! \file Updater.cc
    \brief Defines a base class for all tuners
*/

namespace hoomd
    {
/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle.
*/
Tuner::Tuner(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger)
    : Updater(sysdef, trigger)
    {
    }

namespace detail
    {
void export_Tuner(pybind11::module& m)
    {
    pybind11::class_<Tuner, Updater, std::shared_ptr<Tuner>>(m, "Tuner");
    }

    } // end namespace detail

    } // end namespace hoomd
