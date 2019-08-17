// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHTraverser.cuh"
#include "OutputOps.h"
#include "QueryOps.h"
#include "TransformOps.h"

namespace neighbor
{
namespace gpu
{

// template declaration for compressing without transforming primitives
template void lbvh_compress_ropes(LBVHCompressedData ctree,
                                  const NullTransformOp& transform,
                                  const LBVHData tree,
                                  unsigned int N_internal,
                                  unsigned int N_nodes,
                                  unsigned int block_size,
                                  cudaStream_t stream);

// template declaration for compressing with map transformation of primitives
template void lbvh_compress_ropes(LBVHCompressedData ctree,
                                  const MapTransformOp& transform,
                                  const LBVHData tree,
                                  unsigned int N_internal,
                                  unsigned int N_nodes,
                                  unsigned int block_size,
                                  cudaStream_t stream);

// template declaration to count neighbors
template void lbvh_traverse_ropes(CountNeighborsOp& out,
                                  const LBVHCompressedData& lbvh,
                                  const SphereQueryOp& query,
                                  const Scalar3 *d_images,
                                  unsigned int Nimages,
                                  unsigned int block_size,
                                  cudaStream_t stream);

// template declaration to generate neighbor list
template void lbvh_traverse_ropes(NeighborListOp& out,
                                  const LBVHCompressedData& lbvh,
                                  const SphereQueryOp& query,
                                  const Scalar3 *d_images,
                                  unsigned int Nimages,
                                  unsigned int block_size,
                                  cudaStream_t stream);

} // end namespace gpu
} // end namespace neighbor
