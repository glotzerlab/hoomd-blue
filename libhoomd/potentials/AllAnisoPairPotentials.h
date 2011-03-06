/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: AllAnisoPairPotentials.h 3437 2010-08-30 13:11:57Z baschult $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/aniso/libhoomd/potentials/AllAnisoPairPotentials.h $
// Maintainer: baschult / Anyone is free to add their own pair potentials here

#ifndef __ANISO_PAIR_POTENTIALS__H__
#define __ANISO_PAIR_POTENTIALS__H__

#include "AnisoPotentialPair.h"
#include "EvaluatorPairGayBerne.h"
#include "EvaluatorPairAnisoModulated.h"
#include "AllDirectionalEvaluators.h"

#ifdef ENABLE_CUDA
//include cuda kernels
#endif

/*! \file AllAnisoPairPotentials.h
    \brief Handy list of typedefs for all of the templated pair potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for Gay-Berne forces and torques
typedef AnisoPotentialPair<EvaluatorPairGayBerne> PotentialPairGayBerne;
typedef AnisoPotentialPair< EvaluatorPairAnisoModulated< EvaluatorPairLJ , JanusDirectionalEvaluator> > PotentialPairJanusLJ;
typedef AnisoPotentialPair< EvaluatorPairAnisoModulated<EvaluatorPairLJ,JanusComplementDirectionalEvaluator> > PotentialPairJanusComplimentLJ;
typedef AnisoPotentialPair<EvaluatorPairAnisoModulated<EvaluatorPairLJ,TriblockJanusDirectionalEvaluator> >PotentialPairTriblockJanusLJ;
typedef AnisoPotentialPair<EvaluatorPairAnisoModulated<EvaluatorPairLJ,TriblockJanusComplementDirectionalEvaluator> > PotentialPairTriblockJanusComplimentLJ;

#ifdef ENABLE_CUDA
/*
//! Pair potential force compute for Gay-Berne forces and torques on the GPU
typedef PotentialPairGPU< EvaluatorPairGayBerne, gpu_compute_GB...somethign > PotentialPairGayBerneGPU;
*/
#endif

#endif // __ANISO_PAIR_POTENTIALS_H__

