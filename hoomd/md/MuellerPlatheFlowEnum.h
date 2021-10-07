// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

namespace hoomd
    {
namespace md
    {
/// Struct to embed the enum out of the global scope
struct flow_enum
    {
    /// Direction enum for MuellerPlatheFlow
    enum Direction
        {
        X = 0, //!< X-direction
        Y,     //!< Y-direction
        Z      //!< Z-direction
        };
    };

    } // end namespace md
    } // end namespace hoomd
