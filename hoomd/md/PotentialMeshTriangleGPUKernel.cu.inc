// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off

// clang-format off
#include "hoomd/md/PotentialMeshTriangleGPU.cuh"
#include "hoomd/md/EvaluatorPair@_mesh_triangle_pair@.h"

#define EVALUATOR_CLASS EvaluatorPair@_mesh_triangle_pair@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_triangles_force<EVALUATOR_CLASS>(const kernel::meshtriangle_args_t& meshtriangle_args,
                                            const typename EVALUATOR_CLASS::param_type* d_params,
                                            const group_storage<3>* tlist,
					    const unsigned int* tpos_list,
					    const Index2D tlist_idx,
					    const unsigned int* n_triangles_list);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
