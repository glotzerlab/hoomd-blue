/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#ifndef __BOND_EXT_POTENTIALS__H__
#define __BOND_EXT_POTENTIALS__H__

// need to include hoomd_config and PotentialBond here
#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialBond.h"

// include all of the evaluators that the plugin contains
#include "EvaluatorBondHarmonicDPD.h"

#ifdef ENABLE_CUDA
// PotentialBondGPU is the class that performs the bond computations on the GPU
#include "hoomd/PotentialBondGPU.h"
// AllDriverPotentialBondExtGPU.cuh is a header file containing the kernel driver functions for computing the bond
// potentials defined in this plugin. See it for more details
#include "AllDriverPotentialBondExtGPU.cuh"
#endif

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Bond potential force compute for Harmonic+DPD forces
typedef PotentialBond<EvaluatorBondHarmonicDPD> PotentialBondHarmonicDPD;

#ifdef ENABLE_CUDA
//! Bond potential force compute for Harmonic+DPD forces on the GPU
typedef PotentialBondGPU< EvaluatorBondHarmonicDPD, gpu_compute_harmonic_dpd_forces > PotentialBondHarmonicDPDGPU;
#endif

#endif // __BOND_EXT_POTENTIALS_H__
