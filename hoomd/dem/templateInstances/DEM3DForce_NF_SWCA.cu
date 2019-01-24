// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "../DEM3DForceGPU.cu"

typedef DEMEvaluator<Scalar, Scalar4, SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > > SWCADEM;

template cudaError_t gpu_compute_dem3d_forces<Scalar, Scalar4, SWCADEM>(
        Scalar4* d_force, Scalar4* d_torque, Scalar* d_virial,
        const unsigned int virial_pitch, const unsigned int N, const unsigned int n_ghosts,
        const Scalar4 *d_pos,
        const Scalar4 *d_quat,     const unsigned int *d_nextFaces,
        const unsigned int *d_firstFaceVertices,
        const unsigned int *d_nextVertices,
        const unsigned int *d_realVertices,
        const Scalar4 *d_vertices,
        const Scalar *d_diam,
        const Scalar4 *d_velocity,
        const unsigned int maxFeatures, const unsigned int maxVertices,
        const unsigned int numFaces,
        const unsigned int numDegenerateVerts,
        const unsigned int numVerts,
        const unsigned int numEdges,
        const unsigned int numTypes,
        const BoxDim& box,
        const unsigned int *d_n_neigh, const unsigned int *d_nlist,
        const unsigned int *d_head_list, const SWCADEM evaluator, const Scalar r_cutsq,
        const unsigned int particlesPerBlock, const unsigned int *d_firstTypeVert,
        const unsigned int *d_numTypeVerts, const unsigned int *d_firstTypeEdge,
        const unsigned int *d_numTypeEdges, const unsigned int *d_numTypeFaces,
        const unsigned int *d_vertexConnectivity, const unsigned int *d_edges);
