/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file BinnerCompute.h
    \brief Declares a base class for all computes
*/

#ifndef __BINNER_COMPUTE_H__
#define __BINNER_COMPUTE_H__

#include "Compute.h"
#include "GPUArray.h"

//! Simple structure for holding the binned particles
struct BinData
    {
    //! Constructs an empty BinData
    BinData();
    //! Allocate and/or reallocate memory as needed to setup the bins given the parameters
    void resize(const BoxDim& box, Scalar width);
    
    GPUArray< unsigned int > bin_idxlist;   //!< \a Nmax x (\a bin_dim.x x \a bin_dim.y x \a  bin_dim.z) 4D array holding the indices of the particles in each cell
    GPUArray< uint4 > bin_coord;            //!< \a bin_dim.x x \a bin_dim.y x \a  bin_dim.z 1D array containing the coordinates of each bin
    GPUArray< unsigned int > bin_adjlist;   //!< (\a bin_dim.x x \a bin_dim.y x \a  bin_dim.z) x 27 2D array listing the bins adjacect to bin x,y,z
    GPUArray< unsigned int > bin_size;      //!< \a bin_dim.x x \a bin_dim.y x \a  bin_dim.z 1D array containing the size of each bin
    
    Scalar3 bin_shape;  //!< Length of each side of the bins in .x, .y, and .z
    Scalar3 bin_scale;  //!< Scale factor to take particle coordinates into the bin coordinates in .x, .y, and .z
    uint3 bin_dim;      //!< Dimensions of the 3D grid of bins
    unsigned int Nmax;  //!< Maximum number of particles that can be stored in each bin
    
private:
    bool allocated;     //!< Set to true if this BinData has been allocated
    };


#endif

