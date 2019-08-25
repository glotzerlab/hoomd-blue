# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

set(NEIGHBOR_CXX_SRCS LBVH.cc LBVHTraverser.cc PARENT_SCOPE)
set(NEIGHBOR_CU_SRCS LBVH.cu LBVHTraverser.cu PARENT_SCOPE)
set(NEIGHBOR_HEADERS
    BoundingVolumes.h
    InsertOps.h
    LBVH.cuh
    LBVH.h
    LBVHTraverser.cuh
    LBVHTraverser.h
    OutputOps.h
    QueryOps.h
    TransformOps.h
    PARENT_SCOPE
    )
