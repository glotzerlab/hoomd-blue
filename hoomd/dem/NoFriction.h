/*
  Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
  (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
  Iowa State University and The Regents of the University of Michigan All rights
  reserved.

  HOOMD-blue may contain modifications ("Contributions") provided, and to which
  copyright is held, by various Contributors who have granted The Regents of the
  University of Michigan the right to modify and/or distribute such Contributions.

  You may redistribute, use, and create derivate works of HOOMD-blue, in source
  and binary forms, provided you abide by the following conditions:

  * Redistributions of source code must retain the above copyright notice, this
  list of conditions, and the following disclaimer both in the code and
  prominently in any materials provided with the distribution.

  * Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions, and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  * All publications and presentations based on HOOMD-blue, including any reports
  or published results obtained, in whole or in part, with HOOMD-blue, will
  acknowledge its use according to the terms posted at the time of submission on:
  http://codeblue.umich.edu/hoomd-blue/citations.html

  * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
  http://codeblue.umich.edu/hoomd-blue/

  * Apart from the above required attributions, neither the name of the copyright
  holder nor the names of HOOMD-blue's contributors may be used to endorse or
  promote products derived from this software without specific prior written
  permission.

  Disclaimer

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
  WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: mspells

/*! \file NoFriction.h
  \brief Declares the pure virtual NoFriction class
*/

#ifndef __NOFRICTION_H__
#define __NOFRICTION_H__

#include "VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

template<typename Real>
class NoFriction
    {
    public:
        NoFriction() {}

        DEVICE static bool needsVelocity() {return false;}

        DEVICE inline void setVelocity(const vec3<Real> &v) {}

        DEVICE inline void swapij() {}

        template<typename Vec>
        DEVICE inline Vec modifiedForce(const Vec &r0Prime, const Vec &force) const {return force;}
    };

#endif
