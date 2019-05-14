// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

#include "VectorMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/TextureTools.h"
#include "DEM3DForceGPU.cuh"
#include "hoomd/HOOMDMath.h"
#include "atomics.cuh"
#include "WCAPotential.h"
#include "SWCAPotential.h"

#include "NoFriction.h"

/*! \file DEM3DForceGPU.cu
  \brief Defines GPU kernel code for calculating conservative DEM pair forces. Used by DEM3DForceComputeGPU.
*/

//! Kernel for calculating 3D DEM forces
/*! This kernel is called to calculate the DEM forces for all N particles.

  \param d_force Device memory to write computed forces
  \param d_torque Device memory to write computed torques
  \param d_virial Device memory to write computed virials
  \param virial_pitch pitch of 3D virial array
  \param N number of particles
  \param d_vertices Vertex indices on the GPU
  \param d_vertex_indices Vertex linkage indices on the GPU
  \param vertexCount Total number of vertices in all shapes
  \param box Box dimensions (in GPU format) to use for periodic boundary conditions
  \param d_n_neigh Device memory array listing the number of neighbors for each particle
  \param d_nlist Device memory array containing the neighbor list contents
  \param nli Indexer for indexing \a d_nlist
  \param potential Parameters for the given potential (such as sigma for WCA)
  \param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
  force is set to 0

  Enough shared memory to hold vertexCount Scalar2's and unsigned int's
  as well as blockDim.x (1*Scalar4 and 6*Scalars) should be allocated
  for the kernel call.

  Developer information:
  Each block will calculate the forces for blockDim.x particles.
  Each thread will calculate the force contribution for one vertex being treated as a vertex or an edge.
  The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template<typename Real, typename Real4, typename Evaluator>
__global__ void gpu_compute_dem3d_forces_kernel(
    const Scalar4* d_pos, const Scalar4* d_quat,
    Scalar4* d_force, Scalar4* d_torque, Scalar* d_virial,
    const unsigned int virial_pitch, const unsigned int N,
    const unsigned int *d_nextFaces, const unsigned int *d_firstFaceVertices,
    const unsigned int *d_nextVertices, const unsigned int *d_realVertices,
    const Real4 *d_vertices, const Scalar *d_diam, const Scalar4 *d_velocity,
    const unsigned int maxFeatures, const unsigned int maxVertices,
    const unsigned int numFaces, const unsigned int numDegenerateVerts,
    const unsigned int numVerts, const unsigned int numEdges,
    const unsigned int numTypes, const BoxDim box, const unsigned int *d_n_neigh,
    const unsigned int *d_nlist, const unsigned int *d_head_list, Evaluator evaluator,
    const Real r_cutsq, const unsigned int *d_firstTypeVert,
    const unsigned int *d_numTypeVerts, const unsigned int *d_firstTypeEdge,
    const unsigned int *d_numTypeEdges, const unsigned int *d_numTypeFaces,
    const unsigned int *d_vertexConnectivity, const unsigned int *d_edges)
    {
    extern __shared__ int sh[];

    // part{ForceTorques, Virials} are the forces and torques
    // (force.x, force.y, torque.z, potentialEnergy), and virials for
    // each particle that this block will calculate for (blockDim.x
    // particles)
    Scalar4 *partForces((Scalar4*)sh);
    size_t shOffset(sizeof(Scalar4)/sizeof(int)*blockDim.x);
    Scalar4 *partTorques((Scalar4*)&sh[shOffset]);
    shOffset += sizeof(Scalar4)/sizeof(int)*blockDim.x;

    // real vertex index->vertex (position)
    Real4 *vertices((Real4*)&sh[shOffset]);
    shOffset += sizeof(Real4)/sizeof(int)*numVerts;

    // particle virials
    Real *partVirials((Real*)&sh[shOffset]);
    shOffset += sizeof(Real)/sizeof(int)*6*blockDim.x;

    // face->next face, face->first vertex in face
    unsigned int *nextFaces((unsigned int*)&sh[shOffset]);
    shOffset += numFaces;
    unsigned int *firstFaceVertex((unsigned int*)&sh[shOffset]);
    shOffset += numFaces;

    // deg. vertex->next deg. vertex, deg.vertex->real vertex
    unsigned int *nextVertex((unsigned int*)&sh[shOffset]);
    shOffset += numDegenerateVerts;
    unsigned int *realVertex((unsigned int*)&sh[shOffset]);
    shOffset += numDegenerateVerts;

    // 2*edge->first real vert, 2*edge+1->second real vert in edge
    unsigned int *edges((unsigned int*)&sh[shOffset]);
    shOffset += 2*numEdges;

    // type->first real vertex index, type->number of vertices,
    // type->first edge in pair, type->number of edges,
    // type -> number of faces
    unsigned int *firstTypeVert((unsigned int*)&sh[shOffset]);
    shOffset += numTypes;
    unsigned int *numTypeVerts((unsigned int*)&sh[shOffset]);
    shOffset += numTypes;
    unsigned int *firstEdgeInType((unsigned int*)&sh[shOffset]);
    shOffset += numTypes;
    unsigned int *numTypeEdges((unsigned int*)&sh[shOffset]);
    shOffset += numTypes;
    unsigned int *numTypeFaces((unsigned int*)&sh[shOffset]);
    shOffset += numTypes;

    // real vertex index -> vertex connectivity
    unsigned int *vertexConnectivity((unsigned int*)&sh[shOffset]);
    shOffset += numVerts;

    // partIdx is the absolute index of the particle this thread is
    // calculating for
    size_t partIdx(blockIdx.x*blockDim.x + threadIdx.x);

    // localThreadIdx is just this thread's index in the block; use it
    // to load vertices
    const size_t localThreadIdx(threadIdx.y*blockDim.x + threadIdx.x);
    unsigned int offset(0);
    do
        {
        if(localThreadIdx + offset < numFaces)
            {
            nextFaces[localThreadIdx + offset] = d_nextFaces[localThreadIdx + offset];
            firstFaceVertex[localThreadIdx + offset] = d_firstFaceVertices[localThreadIdx + offset];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < numFaces);

    offset = 0;
    do
        {
        if(localThreadIdx + offset < numDegenerateVerts)
            {
            nextVertex[localThreadIdx + offset] = d_nextVertices[localThreadIdx + offset];
            realVertex[localThreadIdx + offset] = d_realVertices[localThreadIdx + offset];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < numDegenerateVerts);

    offset = 0;
    do
        {
        if(localThreadIdx + offset < numVerts)
            {
            vertices[localThreadIdx + offset] = d_vertices[localThreadIdx + offset];
            vertexConnectivity[localThreadIdx + offset] = d_vertexConnectivity[localThreadIdx + offset];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < numVerts);

    offset = 0;
    do
        {
        if(localThreadIdx + offset < numEdges)
            {
            edges[2*(localThreadIdx + offset)] = d_edges[2*(localThreadIdx + offset)];
            edges[2*(localThreadIdx + offset) + 1] = d_edges[2*(localThreadIdx + offset) + 1];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < numEdges);

    offset = 0;
    do
        {
        if(localThreadIdx + offset < numTypes)
            {
            firstTypeVert[localThreadIdx + offset] = d_firstTypeVert[localThreadIdx + offset];
            numTypeVerts[localThreadIdx + offset] = d_numTypeVerts[localThreadIdx + offset];
            firstEdgeInType[localThreadIdx + offset] = d_firstTypeEdge[localThreadIdx + offset];
            numTypeEdges[localThreadIdx + offset] = d_numTypeEdges[localThreadIdx + offset];
            numTypeFaces[localThreadIdx + offset] = d_numTypeFaces[localThreadIdx + offset];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < numTypes);

    // zero the accumulator values
    if(threadIdx.y == 0)
        {
        partForces[threadIdx.x] = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
        partTorques[threadIdx.x] = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
        for(size_t i(0); i < 6; ++i)
            partVirials[6*threadIdx.x + i] = 0.0f;
        }

    // Don't calculate results for nonsensical features
    if(threadIdx.y >= maxFeatures)
        partIdx = N;

    // zero the calculated force, torque, and virial for this particle
    // in this thread. Note that localForceTorque is (force.x,
    // force.y, torque.z, potentialEnergy).
    Real4 localForce(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
    Real4 localTorque(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
    Real localVirial[6];
    for(size_t i(0); i < 6; ++i)
        localVirial[i] = 0.0f;

    // make sure the shared memory initializations above are visible
    // for the whole block
    __syncthreads();

    if(partIdx < N)
        {
        const unsigned int n_neigh(d_n_neigh[partIdx]);
        const unsigned int myHead(d_head_list[partIdx]);

        // fetch position and orientation of this particle
        const Scalar4 postype(__ldg(d_pos + partIdx));
        const vec3<Scalar> pos_i(postype.x, postype.y, postype.z);
        const unsigned int type_i(__scalar_as_int(postype.w));
        const Scalar4 quati(__ldg(d_quat + partIdx));
        const quat<Real> quat_i(quati.x, vec3<Real>(quati.y, quati.z, quati.w));

        Scalar di = 0.0f;
        if (Evaluator::needsDiameter())
            di = __ldg(d_diam + partIdx);
        else
            di += 1.0f; //shut up compiler warning. Vestigial from HOOMD

        vec3<Scalar> vi;
        if (Evaluator::needsVelocity())
            vi = vec3<Scalar>(
                __ldg(d_velocity + partIdx));

        for(unsigned int featureEpoch(0);
            featureEpoch < (maxFeatures + blockDim.y - 1)/blockDim.y; ++featureEpoch)
            {

            const unsigned int localFeatureIdx(featureEpoch*blockDim.y + threadIdx.y);
            if(localFeatureIdx >= maxFeatures)
                continue;
            const unsigned int featureType(min(localFeatureIdx/maxVertices, 2));
            const unsigned int featureIdx(localFeatureIdx - maxVertices*featureType);

            // reference points for vertex/face or edge/edge interactions
            vec3<Real> r0, r1;
            if(featureType == 0 && featureIdx < numTypeVerts[type_i])
                r0 = rotate(quat_i, vec3<Real>(vertices[firstTypeVert[type_i] + featureIdx]));
            else if(featureType == 2 && featureIdx < numTypeEdges[type_i])
                {
                r0 = rotate(quat_i, vec3<Real>(vertices[edges[2*(firstEdgeInType[type_i] + featureIdx)]]));
                r1 = rotate(quat_i, vec3<Real>(vertices[edges[2*(firstEdgeInType[type_i] + featureIdx) + 1]]));
                }

            // prefetch neighbor index
            unsigned int cur_neigh(0);
            unsigned int next_neigh(d_nlist[myHead]);

            for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
                {
                // read the current neighbor index (MEM TRANSFER: 4 bytes)
                // prefetch the next value and set the current one
                cur_neigh = next_neigh;
                next_neigh = d_nlist[myHead + neigh_idx + 1];

                // grab the position and type of the neighbor
                const Scalar4 neigh_postype(__ldg(d_pos + cur_neigh));
                const unsigned int type_j(__scalar_as_int(neigh_postype.w));
                const vec3<Scalar> neigh_pos(neigh_postype.x, neigh_postype.y, neigh_postype.z);

                // rij is the distance from the center of particle
                // i to particle j
                vec3<Scalar> rijScalar(neigh_pos - pos_i);
                rijScalar = vec3<Scalar>(box.minImage(vec_to_scalar3(rijScalar)));
                const vec3<Real> rij(rijScalar);
                const Real rsq(dot(rij, rij));

                // read in the diameter of the particle j if necessary
                // also, set the diameter in the evaluator
                Scalar dj(0);
                if (Evaluator::needsDiameter())
                    {
                    dj = __ldg(d_diam + cur_neigh);
                    evaluator.setDiameter(di, dj);
                    }
                else
                    dj += 1.0f; //shut up compiler warning. Vestigial from HOOMD

                if(evaluator.withinCutoff(rsq, r_cutsq))
                    {
                    // fetch neighbor's orientation
                    const Scalar4 neighQuatF(__ldg(d_quat + cur_neigh));
                    const quat<Real> neighQuat(
                        neighQuatF.x, vec3<Real>(neighQuatF.y, neighQuatF.z, neighQuatF.w));

                    if (Evaluator::needsVelocity())
                        {
                        Scalar4 vj(__ldg(d_velocity + cur_neigh));
                        evaluator.setVelocity(vi - vec3<Scalar>(vj));
                        }

                    Real potentialE(0.0f);
                    vec3<Real> forceij;
                    vec3<Real> forceji;
                    vec3<Real> torqueij;
                    vec3<Real> torqueji;

                    if(featureType == 0 && featureIdx < numTypeVerts[type_i]) // vertices of i and faces/edges of j
                        {
                        // faces of j
                        unsigned int faceIndexj(type_j);
                        // shape j is a polyhedron
                        if(numTypeFaces[type_j] > 0)
                            {
                            do
                                {
                                evaluator.vertexFace(rij, r0, neighQuat,
                                    vertices, realVertex,
                                    nextVertex,
                                    firstFaceVertex[faceIndexj],
                                    potentialE, forceij,
                                    torqueij, forceji,
                                    torqueji);
                                faceIndexj = nextFaces[faceIndexj];
                                }
                            while(faceIndexj != type_j);
                            }
                        // shape j wasn't a polyhedron, is it a spherocylinder?
                        else if(numTypeEdges[type_j] > 0)
                            {
                            vec3<Real> p10(vertices[edges[2*(firstEdgeInType[type_j])]]);
                            vec3<Real> p11(vertices[edges[2*(firstEdgeInType[type_j]) + 1]]);
                            p10 = rotate(neighQuat, p10);
                            p11 = rotate(neighQuat, p11);

                            evaluator.vertexEdge(rij, r0, p10, p11, potentialE, forceij, torqueij, forceji, torqueji);
                            }
                        // shape j wasn't a spherocylinder either, must be a sphere
                        else
                            {
                            vec3<Real> vertex1(vertices[firstTypeVert[type_j]]);
                            vertex1 = rotate(neighQuat, vertex1);

                            evaluator.vertexVertex(rij, r0, rij + vertex1,
                                potentialE, forceij, torqueij,
                                forceji, torqueji);

                            }
                        }
                    else if(featureType == 1 && featureIdx < numTypeVerts[type_j]) // vertices of j and faces/edges of i
                        {
                        evaluator.swapij();
                        r0 = rotate(neighQuat, vec3<Real>(vertices[firstTypeVert[type_j] + featureIdx]));

                        // faces of i
                        unsigned int faceIndexi(type_i);
                        if(numTypeFaces[type_i] > 0)
                            {
                            do
                                {
                                evaluator.vertexFace(-rij, r0, quat_i,
                                    vertices, realVertex, nextVertex,
                                    firstFaceVertex[faceIndexi], potentialE,
                                    forceji, torqueji, forceij, torqueij);
                                faceIndexi = nextFaces[faceIndexi];
                                }
                            while(faceIndexi != type_i);
                            }
                        // shape j wasn't a polyhedron, is it a spherocylinder?
                        else if(numTypeEdges[type_i] > 0)
                            {
                            vec3<Real> p00(vertices[edges[2*(firstEdgeInType[type_i])]]);
                            vec3<Real> p01(vertices[edges[2*(firstEdgeInType[type_i]) + 1]]);
                            p00 = rotate(quat_i, p00);
                            p01 = rotate(quat_i, p01);

                            evaluator.vertexEdge(-rij, r0, p00, p01, potentialE, forceji, torqueji, forceij, torqueij);
                            }
                        // shape j wasn't a spherocylinder either, must be
                        // a sphere; this is accounted for above, though.
                        }
                    else if(featureType == 2 && featureIdx < numTypeEdges[type_i]) // edge/edge
                        {
                        for(unsigned int edgeIdx(firstEdgeInType[type_j]);
                            edgeIdx < firstEdgeInType[type_j] + numTypeEdges[type_j]; ++edgeIdx)
                            {
                            const vec3<Real> r2(rotate(neighQuat, vec3<Real>(vertices[edges[2*edgeIdx]])));
                            const vec3<Real> r3(rotate(neighQuat, vec3<Real>(vertices[edges[2*edgeIdx + 1]])));
                            evaluator.edgeEdge(rij, r0, r1, rij + r2, rij + r3,
                                potentialE, forceij, torqueij, forceji, torqueji);
                            }
                        }

                    localForce.x += forceij.x;
                    localForce.y += forceij.y;
                    localForce.z += forceij.z;
                    localForce.w += potentialE;
                    localTorque.x += torqueij.x;
                    localTorque.y += torqueij.y;
                    localTorque.z += torqueij.z;

                    localVirial[0] -= .5f*rij.x*forceij.x;
                    localVirial[1] -= .5f*rij.y*forceij.x;
                    localVirial[2] -= .5f*rij.z*forceij.x;
                    localVirial[3] -= .5f*rij.y*forceij.y;
                    localVirial[4] -= .5f*rij.z*forceij.y;
                    localVirial[5] -= .5f*rij.z*forceij.z;

                    }
                }
            }
        }

    // sum all the intermediate force and torque values for each
    // particle we calculate for in the block.
    if(partIdx < N)
        {
        genAtomicAdd(&partForces[threadIdx.x].x, localForce.x);
        genAtomicAdd(&partForces[threadIdx.x].y, localForce.y);
        genAtomicAdd(&partForces[threadIdx.x].z, localForce.z);
        genAtomicAdd(&partForces[threadIdx.x].w, localForce.w);

        genAtomicAdd(&partTorques[threadIdx.x].x, localTorque.x);
        genAtomicAdd(&partTorques[threadIdx.x].y, localTorque.y);
        genAtomicAdd(&partTorques[threadIdx.x].z, localTorque.z);

        for(size_t i(0); i < 6; ++i)
            genAtomicAdd(partVirials + 6*threadIdx.x + i, localVirial[i]);
        }

    __syncthreads();

    // finally, write the result.
    if(partIdx < N && threadIdx.y == 0)
        {
        partForces[threadIdx.x].w *= .5f;
        d_force[partIdx] = partForces[threadIdx.x];
        d_torque[partIdx] = partTorques[threadIdx.x];

        for(size_t i(0); i < 6; ++i)
            d_virial[i*virial_pitch + partIdx] = partVirials[6*threadIdx.x + i];
        }
    }

/*! \param d_force Device memory to write computed forces
  \param d_torque Device memory to write computed torques
  \param d_virial Device memory to write computed virials
  \param virial_pitch pitch of 3D virial array
  \param N number of particles
  \param d_pos particle positions on the GPU
  \param d_quat particle orientations on the GPU
  \param d_vertices Vertex indices on the GPU
  \param d_vertex_indices Vertex linkage indices on the GPU
  \param vertexCount Total number of vertices in all shapes
  \param box Box dimensions (in GPU format) to use for periodic boundary conditions
  \param d_n_neigh Device memory array listing the number of neighbors for each particle
  \param d_nlist Device memory array containing the neighbor list contents
  \param nli Indexer for indexing \a d_nlist
  \param potential Parameters for the given potential (such as sigma for WCA)
  \param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
  force is set to 0
  \param particlesPerBlock Block size to execute
  \param maxVerts Maximum number of vertices in any shape

  \returns Any error code resulting from the kernel launch

  This is just a driver for gpu_compute_dem3d_forces_kernel, see the documentation for it for more information.
*/
template<typename Real, typename Real4, typename Evaluator>
cudaError_t gpu_compute_dem3d_forces(
    Scalar4* d_force, Scalar4* d_torque, Scalar* d_virial,
    const unsigned int virial_pitch, const unsigned int N, const unsigned int n_ghosts,
    const Scalar4 *d_pos, const Scalar4 *d_quat, const unsigned int *d_nextFaces,
    const unsigned int *d_firstFaceVertices, const unsigned int *d_nextVertices,
    const unsigned int *d_realVertices, const Real4 *d_vertices,
    const Scalar *d_diam, const Scalar4 *d_velocity,
    const unsigned int maxFeatures, const unsigned int maxVertices,
    const unsigned int numFaces, const unsigned int numDegenerateVerts,
    const unsigned int numVerts, const unsigned int numEdges,
    const unsigned int numTypes, const BoxDim& box, const unsigned int *d_n_neigh,
    const unsigned int *d_nlist, const unsigned int *d_head_list, const Evaluator evaluator,
    const Real r_cutsq, const unsigned int particlesPerBlock,
    const unsigned int *d_firstTypeVert, const unsigned int *d_numTypeVerts,
    const unsigned int *d_firstTypeEdge, const unsigned int *d_numTypeEdges,
    const unsigned int *d_numTypeFaces, const unsigned int *d_vertexConnectivity,
    const unsigned int *d_edges)
    {

    // setup the grid to run the kernel
    dim3 grid((int)ceil((double)N / (double)particlesPerBlock), 1, 1);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, gpu_compute_dem3d_forces_kernel<Real, Real4, Evaluator>);
    const unsigned int numFeatures(min(maxFeatures, attr.maxThreadsPerBlock/particlesPerBlock));
    assert(numFeatures);

    dim3 threads(particlesPerBlock, numFeatures, 1);

    // Calculate the amount of shared memory required
    const size_t shmSize(2*particlesPerBlock*sizeof(Real4) + 6*particlesPerBlock*sizeof(Real) + // forces, torques, virials per-particle
        2*numFaces*sizeof(unsigned int) + // face->next face, face->first vertex in face
        2*numDegenerateVerts*sizeof(unsigned int) + // vertex->next vertex, vertex->real vertex
        numVerts*sizeof(Real4) + 2*numEdges*sizeof(unsigned int) + // real vertex->point, edge->real index
        5*numTypes*sizeof(unsigned int) + numVerts*sizeof(unsigned int)); // per-type counts and per-vertex connectivity

    // run the kernel
    gpu_compute_dem3d_forces_kernel<Real, Real4, Evaluator> <<< grid, threads,shmSize>>>
        (d_pos, d_quat, d_force, d_torque, d_virial, virial_pitch, N,
        d_nextFaces, d_firstFaceVertices, d_nextVertices,
        d_realVertices, d_vertices, d_diam, d_velocity, maxFeatures,
        maxVertices, numFaces, numDegenerateVerts, numVerts,
        numEdges, numTypes, box, d_n_neigh, d_nlist, d_head_list, evaluator,
        r_cutsq, d_firstTypeVert, d_numTypeVerts, d_firstTypeEdge,
        d_numTypeEdges, d_numTypeFaces, d_vertexConnectivity,
        d_edges);

    return cudaSuccess;
    }

// vim:syntax=cpp
