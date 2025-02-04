// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
