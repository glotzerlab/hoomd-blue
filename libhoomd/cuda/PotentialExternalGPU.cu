/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

#include "WallData.h"
#include "PotentialExternalGPU.cuh"
#include "EvaluatorWalls.h"
#include "EvaluatorExternalPeriodic.h"
#include "EvaluatorExternalElectricField.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairMie.h"


template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field)
    {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_external_forces_kernel<evaluator>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(external_potential_args.block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( external_potential_args.N / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);
        unsigned int bytes = (sizeof(typename evaluator::field_type)/sizeof(int)+1)*sizeof(int);

        // run the kernel
        gpu_compute_external_forces_kernel<evaluator><<<grid, threads, bytes>>>(external_potential_args.d_force,
                                                                                external_potential_args.d_virial,
                                                                                external_potential_args.virial_pitch,
                                                                                external_potential_args.N,
                                                                                external_potential_args.d_pos,
                                                                                external_potential_args.d_diameter,
                                                                                external_potential_args.d_charge,
                                                                                external_potential_args.box,
                                                                                d_params,
                                                                                d_field);

        return cudaSuccess;
    };

//Instantiate external evaluator templates

//! Evaluator for External Periodic potentials.
template cudaError_t gpu_cpef<EvaluatorExternalPeriodic>(const external_potential_args_t& external_potential_args, const typename EvaluatorExternalPeriodic::param_type *d_params, const typename EvaluatorExternalPeriodic::field_type *d_field);
//! Evaluator for electric fields
template cudaError_t gpu_cpef<EvaluatorExternalElectricField>(const external_potential_args_t& external_potential_args, const typename EvaluatorExternalElectricField::param_type *d_params, const typename EvaluatorExternalElectricField::field_type *d_field);
//! Evaluator for Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairLJ>::field_type *d_field);
//! Evaluator for Gaussian pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairGauss> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairGauss>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairGauss>::field_type *d_field);
//! Evaluator for Yukawa pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairYukawa> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairYukawa>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairYukawa>::field_type *d_field);
//! Evaluator for Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairSLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairSLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairSLJ>::field_type *d_field);
//! Evaluator for Morse pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairMorse> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairMorse>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairMorse>::field_type *d_field);
//! Evaluator for Force Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairForceShiftedLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::field_type *d_field);
//! Evaluator for Mie pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairMie> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairMie>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairMie>::field_type *d_field);
