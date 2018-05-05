// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: bdice
#ifndef __SSAGESHOOK_H__
#define __SSAGESHOOK_H__

/*! \file SSAGESHook.h
    \brief Declares the SSAGESHook class
    The class SSAGESHOOMDHook in the SSAGES code base inherits from this
    abstract class.
*/

#include "SystemDefinition.h"

class SSAGESHook
{
    public:
        // Set SystemDefinition.
        virtual void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) = 0;

        // Setup for presimulation call.
        virtual void setup() = 0;

        // Post force where the synchronization occurs.
        virtual void post_force() = 0;

        // Post-run for post-simulation call.
        virtual void post_run() = 0;

        // Post-step for post-step call.
        virtual void end_of_step() = 0;

        virtual ~SSAGESHook() {};
};

#endif
