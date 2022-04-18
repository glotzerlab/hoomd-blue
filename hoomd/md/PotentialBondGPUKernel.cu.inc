// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialBondGPU.cuh"
#include "hoomd/md/EvaluatorBond@_bond@.h"


namespace hoomd { namespace md { namespace kernel {
template
__attribute__((visibility("default")))
hipError_t
gpu_compute_bond_forces<EvaluatorBond@_bond@>(const kernel::bond_args_t& bond_args,
                                   const typename EvaluatorBond@_bond@::param_type* d_params,
                                   unsigned int* d_flags);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
