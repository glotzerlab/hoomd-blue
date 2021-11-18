// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: bdice, csadorf
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
class HalfStepHook
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
