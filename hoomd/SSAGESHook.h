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

        // Synchronize snapshot with SSAGES after computing forces
        virtual void updateSSAGES() = 0;

        virtual ~SSAGESHook() {};
};

#endif
