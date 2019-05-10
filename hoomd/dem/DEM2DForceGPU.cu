// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include "VectorMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "DEM2DForceGPU.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "DEMEvaluator.h"
#include "atomics.cuh"
#include "WCAPotential.h"
#include "SWCAPotential.h"

#include "NoFriction.h"

/*! \file DEM2DForceGPU.cu
  \brief Defines GPU kernel code for calculating conservative DEM pair forces. Used by DEM2DForceComputeGPU.
*/

//! Kernel for calculating 2D DEM forces
/*! This kernel is called to calculate the DEM forces for all N particles.

  \param d_force Device memory to write computed forces
  \param d_torque Device memory to write computed torques
  \param d_virial Device memory to write computed virials
  \param virial_pitch pitch of 2D virial array
  \param d_diameter Device memory to read particle diameters
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
  as well as blockDim.y (1*Scalar4 and 6*Scalars) should be allocated
  for the kernel call.

  Developer information:
  Each block will calculate the forces for blockDim.y particles.
  Each thread will calculate the force contribution for one vertex being treated as a vertex or an edge.
  The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/

template<typename Real, typename Real2, typename Real4, typename Evaluator>
__global__ void gpu_compute_dem2d_forces_kernel(const Scalar4 *d_pos,
    const Scalar4 *d_quat,
    Scalar4* d_force,
    Scalar4* d_torque,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Real2 *d_vertices,
    const unsigned int *d_num_shape_verts,
    const Scalar *d_diam,
    const Scalar4 *d_velocity,
    const unsigned int vertexCount,
    const BoxDim box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    Evaluator evaluator,
    const Real r_cutsq,
    const unsigned int n_shapes,
    const unsigned int maxVerts)
    {
    extern __shared__ int sh[];
    // part{ForceTorques, Virials} are the forces and torques
    // (force.x, force.y, torque.z, potentialEnergy), and virials for
    // each particle that this block will calculate for (blockDim.y
    // particles)
    Real4 *partForceTorques((Real4*)sh);
    size_t shOffset(sizeof(Real4)/sizeof(int)*blockDim.y);

    // vertices and vertexIndices are the 2D vertices and an index
    // into these lists of the next vertex for the corresponding edge,
    // respectively
    Real2 *vertices((Real2*)&sh[shOffset]);
    shOffset += sizeof(Real2)/sizeof(int)*vertexCount;

    // per-particle virials
    Real *partVirials((Real*)&sh[shOffset]);
    shOffset += sizeof(Real)/sizeof(int)*6*blockDim.y;

    // numShapeVerts is an array of shape index -> number of
    // vertices in that shape
    unsigned int *numShapeVerts((unsigned int*)&sh[shOffset]);
    shOffset += n_shapes;
    // firstShapeVert is an array of shape index -> first vertex in
    // vertices
    unsigned int *firstShapeVert((unsigned int*)&sh[shOffset]);
    shOffset += n_shapes;

    // partIdx is the absolute index of the particle this thread is
    // calculating for
    size_t partIdx(blockIdx.x*blockDim.y + threadIdx.y);

    // localThreadIdx is just this thread's index in the block; use it
    // to load vertices
    const size_t localThreadIdx(threadIdx.z*blockDim.x*blockDim.y +
        threadIdx.y*blockDim.x + threadIdx.x);
    int offset(0);
    do
        {
        if(localThreadIdx + offset < vertexCount)
            {
            vertices[localThreadIdx + offset] = d_vertices[localThreadIdx + offset];
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < vertexCount);

    offset = 0;
    do
        {
        if(localThreadIdx + offset < n_shapes)
            {
            numShapeVerts[localThreadIdx + offset] = d_num_shape_verts[localThreadIdx + offset];
            unsigned int firstVert(0);
            for(int i(localThreadIdx + offset - 1); i >= 0; --i)
                firstVert += d_num_shape_verts[i];
            firstShapeVert[localThreadIdx + offset] = firstVert;
            }
        offset += blockDim.x*blockDim.y*blockDim.z;
        }
    while(offset < n_shapes);

    // zero the accumulator values
    if(threadIdx.z == 0 && threadIdx.x == 0)
        {
        partForceTorques[threadIdx.y] = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
        for(size_t i(0); i < 6; ++i)
            partVirials[6*threadIdx.y + i] = 0.0f;
        }

    // zero the calculated force, torque, and virial for this particle
    // in this thread. Note that localForceTorque is (force.x,
    // force.y, torque.z, potentialEnergy).
    Real4 localForceTorque(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
    Real localVirial[6];
    for(size_t i(0); i < 6; ++i)
        localVirial[i] = 0.0f;

    // make sure the shared memory initializations above are visible
    // for the whole block
    __syncthreads();

    if(partIdx < N)
        {
        const size_t n_neigh(d_n_neigh[partIdx]);
        const unsigned int myHead(d_head_list[partIdx]);

        // fetch position and orientation of this particle
        const Scalar4 postype(__ldg(d_pos + partIdx));
        const vec3<Scalar> pos_i(postype.x, postype.y, 0);
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
            featureEpoch < (numShapeVerts[type_i] + blockDim.x - 1)/blockDim.x; ++featureEpoch)
            {

            const unsigned int localFeatureIdx(featureEpoch*blockDim.x + threadIdx.x);
            if(localFeatureIdx >= numShapeVerts[type_i])
                continue;

            // go ahead and fetch/rotate the vertex this thread will deal with
            vec2<Real> r0(vec_from_scalar2<Real>(vertices[firstShapeVert[type_i] + localFeatureIdx]));
            r0 = rotate(quat_i, r0);

            // prefetch neighbor index
            size_t cur_neigh(0);
            size_t next_neigh(d_nlist[myHead]);

            // loop over neighbors
            for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
                {
                    {
                    // read the current neighbor index (MEM TRANSFER: 4 bytes)
                    // prefetch the next value and set the current one
                    cur_neigh = next_neigh;
                    next_neigh = d_nlist[myHead + neigh_idx + 1];

                    // grab the position and type of the neighbor
                    const Scalar4 neigh_postype(__ldg(d_pos + cur_neigh));
                    const unsigned int neigh_type(__scalar_as_int(neigh_postype.w));
                    const vec3<Scalar> neigh_pos(neigh_postype.x, neigh_postype.y, 0);

                    // rij is the distance from the center of particle
                    // i to particle j
                    vec3<Scalar> rij3(neigh_pos - pos_i);
                    rij3 = vec3<Scalar>(box.minImage(make_scalar3(rij3.x, rij3.y, 0)));
                    vec2<Real> rij(rij3.x, rij3.y);
                    const Real rsq(dot(rij, rij));

                    // read in the diameter of the particle j if necessary
                    // also, set the diameter in the evaluator
                    if (Evaluator::needsDiameter())
                        {
                        Scalar dj(0);
                        dj = __ldg(d_diam + cur_neigh);
                        evaluator.setDiameter(di, dj);
                        }

                    if(evaluator.withinCutoff(rsq,r_cutsq))
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

                        for(unsigned int neighVertex(0);
                            neighVertex < numShapeVerts[neigh_type]; ++neighVertex)
                            {
                            // Intermediate force/torque storage values
                            Real potentialE(0.0f);
                            vec2<Real> forceij;
                            vec2<Real> forceji;
                            Real torqueij(0.0f);
                            Real torqueji(0.0f);

                            // threadIdx.z == 1: treat this vertex as
                            // the first vertex of an edge
                            if(threadIdx.z)
                                {
                                const unsigned int nextVert((localFeatureIdx + 1)%numShapeVerts[type_i]);
                                // Only evaluate the edge if we aren't
                                // already going to evaluate it
                                // elsewhere (in the case of a
                                // spherocylinder) and we have edges
                                // to evaluate (i.e., the "edge"
                                // doesn't belong to a point particle)
                                if(numShapeVerts[type_i] > 2 || nextVert > localFeatureIdx)
                                    {
                                    evaluator.swapij();
                                    vec2<Real> r1(vec_from_scalar2<Real>(vertices[firstShapeVert[neigh_type] + neighVertex]));
                                    r1 = rotate(neighQuat, r1);
                                    vec2<Real> r2(vec_from_scalar2<Real>(vertices[firstShapeVert[type_i] + nextVert]));
                                    r2 = rotate(quat_i, r2);

                                    evaluator.vertexEdge(-rij, r1, r0, r2, potentialE,
                                        forceji, torqueji, forceij,
                                        torqueij);
                                    }
                                }
                            else // threadIdx.z == 0: treat this
                                // vertex as a vertex
                                {
                                const unsigned int nextVert((neighVertex + 1)%numShapeVerts[neigh_type]);
                                vec2<Real> r1(vec_from_scalar2<Real>(vertices[firstShapeVert[neigh_type] + neighVertex]));
                                r1 = rotate(neighQuat, r1);
                                // if neighVertex==nextVert then
                                // particle j has no edges and no need
                                // to check i verts against j edges;
                                // if the neighbor shape is a
                                // spherocylinder, only count its edge
                                // once.
                                if(numShapeVerts[neigh_type] > 2 || nextVert > neighVertex)
                                    {
                                    vec2<Real> r2(vec_from_scalar2<Real>(vertices[firstShapeVert[neigh_type] + nextVert]));
                                    r2 = rotate(neighQuat, r2);

                                    evaluator.vertexEdge(rij, r0, r1, r2, potentialE,
                                        forceij, torqueij, forceji,
                                        torqueji);
                                    }

                                // if i and j are both disks, evaluate the vertex-vertex potential here
                                if(numShapeVerts[type_i] == 1 && numShapeVerts[neigh_type] == 1)
                                    {
                                    evaluator.vertexVertex(rij, r0, rij + r1,
                                        potentialE, forceij,
                                        torqueij, forceji,
                                        torqueji);
                                    }
                                }

                            localForceTorque.x += forceij.x;
                            localForceTorque.y += forceij.y;
                            localForceTorque.z += torqueij;
                            localForceTorque.w += potentialE;

                            localVirial[0] -= .5f*rij.x*forceij.x;
                            localVirial[1] -= .5f*rij.y*forceij.x;
                            localVirial[3] -= .5f*rij.y*forceij.y;
                            }
                        }
                    }
                }
            }
        }

    // sum all the intermediate force and torque values for each
    // particle we calculate for in the block
    if(partIdx < N)
        {
        genAtomicAdd(&partForceTorques[threadIdx.y].x, localForceTorque.x);
        genAtomicAdd(&partForceTorques[threadIdx.y].y, localForceTorque.y);
        genAtomicAdd(&partForceTorques[threadIdx.y].z, localForceTorque.z);
        genAtomicAdd(&partForceTorques[threadIdx.y].w, localForceTorque.w);
        genAtomicAdd(partVirials + 6*threadIdx.y + 0, localVirial[0]);
        genAtomicAdd(partVirials + 6*threadIdx.y + 1, localVirial[1]);
        genAtomicAdd(partVirials + 6*threadIdx.y + 3, localVirial[3]);
        }

    __syncthreads();

    // finally, write the result
    if(partIdx < N && threadIdx.z == 0 && threadIdx.x == 0)
        {
        partForceTorques[threadIdx.y].w *= .5f;
        d_force[partIdx].x = partForceTorques[threadIdx.y].x;
        d_force[partIdx].y = partForceTorques[threadIdx.y].y;
        d_force[partIdx].w = partForceTorques[threadIdx.y].w;
        d_torque[partIdx].z = partForceTorques[threadIdx.y].z;
        d_virial[0*virial_pitch + partIdx] = partVirials[6*threadIdx.y + 0];
        d_virial[1*virial_pitch + partIdx] = partVirials[6*threadIdx.y + 1];
        d_virial[3*virial_pitch + partIdx] = partVirials[6*threadIdx.y + 3];
        }
    }

/*! \param d_force Device memory to write computed forces
  \param d_torque Device memory to write computed torques
  \param d_virial Device memory to write computed virials
  \param virial_pitch pitch of 2D virial array
  \param N number of particles
  \param d_pos particle positions on the GPU
  \param d_quat particle orientations on the GPU
  \param d_vertices Vertex indices on the GPU
  \param d_vertex_indices Vertex linkage indices on the GPU
  \param d_diameter Device memory to read particle diameters
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

  This is just a driver for gpu_compute_dem2d_forces_kernel, see the documentation for it for more information.
*/
template<typename Real, typename Real2, typename Real4, typename Evaluator>
cudaError_t gpu_compute_dem2d_forces(Scalar4* d_force,
    Scalar4* d_torque,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const unsigned int n_ghosts,
    const Scalar4 *d_pos,
    const Scalar4 *d_quat,
    const Real2 *d_vertices,
    const unsigned int *d_num_shape_verts,
    const Scalar *d_diam,
    const Scalar4 *d_velocity,
    const unsigned int vertexCount,
    const BoxDim& box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    Evaluator potential,
    const Real r_cutsq,
    const unsigned int n_shapes,
    const unsigned int particlesPerBlock,
    const unsigned int maxVerts)
    {

    // setup the grid to run the kernel
    dim3 grid((int)ceil((double)N / (double)particlesPerBlock), 1, 1);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, gpu_compute_dem2d_forces_kernel<Real, Real2, Real4, Evaluator>);
    const unsigned int numFeatures(min(maxVerts, attr.maxThreadsPerBlock/particlesPerBlock/2));
    assert(numFeatures);

    dim3 threads(numFeatures, particlesPerBlock, 2);

    // Calculate the amount of shared memory required
    size_t shmSize(vertexCount*sizeof(Real2) + n_shapes*2*sizeof(unsigned int) +
        particlesPerBlock*(sizeof(Real4) + 6*sizeof(Real)));

    // run the kernel
    gpu_compute_dem2d_forces_kernel<Real, Real2, Real4, Evaluator> <<< grid, threads, shmSize>>>
        (d_pos, d_quat, d_force, d_torque, d_virial, virial_pitch, N, d_vertices,
        d_num_shape_verts, d_diam, d_velocity, vertexCount, box, d_n_neigh,
        d_nlist, d_head_list, potential, r_cutsq, n_shapes, maxVerts);

    return cudaSuccess;
    }

// vim:syntax=cpp
