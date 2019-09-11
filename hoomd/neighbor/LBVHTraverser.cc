// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHTraverser.h"
#include "OutputOps.h"
#include "QueryOps.h"

namespace neighbor
{

template void LBVHTraverser::traverse(CountNeighborsOp&,
                                      const SphereQueryOp&,
                                      const LBVH&,
                                      const GlobalArray<Scalar3>&,
                                      cudaStream_t);

template void LBVHTraverser::traverse(NeighborListOp&,
                                      const SphereQueryOp&,
                                      const LBVH&,
                                      const GlobalArray<Scalar3>&,
                                      cudaStream_t);

} // end namespace neighbor
