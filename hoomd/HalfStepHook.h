// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __HALFSTEPHOOK_H__
#define __HALFSTEPHOOK_H__

/*! \file HalfStepHook.h
    \brief Declares the HalfStepHook class
    This is an abstract base class to enable external libraries to read
    and optionally manipulate snapshot data during the half-step of the
    integration.
*/

#include "SystemDefinition.h"

namespace hoomd
    {
class PYBIND11_EXPORT HalfStepHook
    {
    public:
    // Set SystemDefinition.
    virtual void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) = 0;

    // Synchronize snapshot with external library after computing forces
    virtual void update(uint64_t timestep) = 0;

    virtual ~HalfStepHook() {};
    };

    } // namespace hoomd

#endif
