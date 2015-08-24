/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

#include "PotentialExternalGPU.cuh"

#include "EvaluatorExternalPeriodic.h"
/*
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairEwald.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairDPDThermo.h"
#include "EvaluatorPairMoliere.h"
#include "EvaluatorPairZBL.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairMie.h"
*/
template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args, const typename evaluator::field_type field, const typename evaluator::param_type *d_params)
    {
        return cudaSuccess;
    }
//Instantiate external evaluator templates

//! Evaluator for External Periodic potentials.    
template cudaError_t gpu_cpef<EvaluatorExternalPeriodic>(const external_potential_args_t& external_potential_args, const typename EvaluatorExternalPeriodic::field_type field, const typename EvaluatorExternalPeriodic::param_type *d_params);
/*//! Evaluator for Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorPairLJ>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairLJ::field_type field, const typename EvaluatorPairLJ::param_type *d_params);
//! Evaluator for Gaussian pair potential.
template cudaError_t gpu_cpef<EvaluatorPairGauss>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairGauss::field_type field, const typename EvaluatorPairGauss::param_type *d_params);
//! Evaluator for Yukawa pair potential.
template cudaError_t gpu_cpef<EvaluatorPairYukawa>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairYukawa::field_type field, const typename EvaluatorPairYukawa::param_type *d_params);
//! Evaluator for Ewald pair potential.
template cudaError_t gpu_cpef<EvaluatorPairEwald>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairEwald::field_type field, const typename EvaluatorPairEwald::param_type *d_params);
//! Evaluator for Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorPairSLJ>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairSLJ::field_type field, const typename EvaluatorPairSLJ::param_type *d_params);
//! Evaluator for Morse pair potential.
template cudaError_t gpu_cpef<EvaluatorPairMorse>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairMorse::field_type field, const typename EvaluatorPairMorse::param_type *d_params);
//! Evaluator for Dissipative Particle Dynamics pair potential.
template cudaError_t gpu_cpef<EvaluatorPairDPDThermo>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairDPDThermo::field_type field, const typename EvaluatorPairDPDThermo::param_type *d_params);
//! Evaluator for Moliere pair potential.
template cudaError_t gpu_cpef<EvaluatorPairMoliere>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairMoliere::field_type field, const typename EvaluatorPairMoliere::param_type *d_params);
//! Evaluator for Ziegler-Biersack-Littmark pair potential.
template cudaError_t gpu_cpef<EvaluatorPairZBL>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairZBL::field_type field, const typename EvaluatorPairZBL::param_type *d_params);
//! Evaluator for Dissipative Particle Dynamics with Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorPairDPDLJThermo>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairDPDLJThermo::field_type field, const typename EvaluatorPairDPDLJThermo::param_type *d_params);
//! Evaluator for Force Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorPairForceShiftedLJ>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairForceShiftedLJ::field_type field, const typename EvaluatorPairForceShiftedLJ::param_type *d_params);
//! Evaluator for Mie pair potential.
template cudaError_t gpu_cpef<EvaluatorPairMie>(const external_potential_args_t& external_potential_args, const typename EvaluatorPairMie::field_type field, const typename EvaluatorPairMie::param_type *d_params);
*/


