// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __Manifold_Surface_H__
#define __Manifold_Surface_H__

using namespace std;


/*! \file ManifoldSurface.h
    \brief Defines the manifold surfaces enumerator 
*/



//! Dummy struct to use manifold with GPU
struct manifold_enum
    {
        //! Enum for dimensions
        enum surf
            {
            sphere=0,
            gyroid,
            diamond,
            primitive,
            xy,
            xz,
            yz,
            cylinder
            };
    };


#endif // __Manifold_Surface_H__
