// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_MPCD_TEST_EXTERNAL_FIELD_TEST_CUH_
#define HOOMD_MPCD_TEST_EXTERNAL_FIELD_TEST_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/mpcd/ExternalField.h"

namespace gpu
    {
cudaError_t test_external_field(Scalar3* out,
                                const mpcd::ExternalField* field,
                                const Scalar3* pos,
                                const unsigned int N);
    } // end namespace gpu

#endif // HOOMD_MPCD_TEST_EXTERNAL_FIELD_TEST_CUH_
