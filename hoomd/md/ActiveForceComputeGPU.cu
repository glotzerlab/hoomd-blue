// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ActiveForceComputeGPU.cuh"
#include "EvaluatorConstraintEllipsoid.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;

#include <assert.h>

/*! \file ActiveForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveForceComputeGPU.
*/

//! Kernel for setting active force vectors on the GPU
/*! \param group_size number of particles
    \param d_rtag convert global tag to global index
    \param d_groupTags stores list to convert group index to global tag
    \param d_force particle force on device
    \param d_torque particle torque on device
    \param d_orientation particle orientation on device
    \param d_f_actVec particle active force unit vector
    \param d_f_actMag particle active force vector magnitude
    \param d_t_actVec particle active torque unit vector
    \param d_t_actMag particle active torque vector magnitude
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param orientationLink check if particle orientation is linked to active force vector
*/
__global__ void gpu_compute_active_force_set_forces_kernel(const unsigned int group_size,
                                                    unsigned int *d_rtag,
                                                    unsigned int *d_groupTags,
                                                    Scalar4 *d_force,
                                                    Scalar4 *d_torque,
                                                    Scalar4 *d_orientation,
                                                    Scalar3 *d_f_actVec,
                                                    Scalar *d_f_actMag,
                                                    Scalar3 *d_t_actVec,
                                                    Scalar *d_t_actMag,
                                                    const Scalar3& P,
                                                    Scalar rx,
                                                    Scalar ry,
                                                    Scalar rz,
                                                    bool orientationLink,
                                                    bool orientationReverseLink,
                                                    const unsigned int N)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int tag = d_groupTags[group_idx];
    unsigned int idx = d_rtag[tag];

    Scalar3 f;
    Scalar3 t;
    // rotate force according to particle orientation only if orientation is linked to active force vector
    if (orientationLink == true)
        {
        vec3<Scalar> fi;
        f = make_scalar3(d_f_actMag[tag] * d_f_actVec[tag].x,
                        d_f_actMag[tag] * d_f_actVec[tag].y, d_f_actMag[tag] * d_f_actVec[tag].z);
        quat<Scalar> quati(d_orientation[idx]);
        fi = rotate(quati, vec3<Scalar>(f));
        d_force[idx].x = fi.x;
        d_force[idx].y = fi.y;
        d_force[idx].z = fi.z;

        vec3<Scalar> ti;
        t = make_scalar3(d_t_actMag[tag] * d_t_actVec[tag].x,
                        d_t_actMag[tag] * d_t_actVec[tag].y, d_t_actMag[tag] * d_t_actVec[tag].z);
        ti = rotate(quati, vec3<Scalar>(t));
        d_torque[idx].x = ti.x;
        d_torque[idx].y = ti.y;
        d_torque[idx].z = ti.z;

        }
    else // no orientation link
        {
        f = make_scalar3(d_f_actMag[tag] * d_f_actVec[tag].x,
                        d_f_actMag[tag] * d_f_actVec[tag].y, d_f_actMag[tag] * d_f_actVec[tag].z);
        d_force[idx].x = f.x;
        d_force[idx].y = f.y;
        d_force[idx].z = f.z;

        t = make_scalar3(d_t_actMag[tag] * d_t_actVec[tag].x,
                        d_t_actMag[tag] * d_t_actVec[tag].y, d_t_actMag[tag] * d_t_actVec[tag].z);
        d_torque[idx].x = t.x;
        d_torque[idx].y = t.y;
        d_torque[idx].z = t.z;

        }
    // rotate particle orientation only if orientation is reverse linked to active force vector. Ignore torque here
    if (orientationReverseLink == true)
        {
        vec3<Scalar> f(d_f_actMag[tag] * d_f_actVec[tag].x,
                        d_f_actMag[tag] * d_f_actVec[tag].y, d_f_actMag[tag] * d_f_actVec[tag].z);
        vec3<Scalar> vecZ(0.0, 0.0, 1.0);
        vec3<Scalar> quatVec = cross(vecZ, f);
        Scalar quatScal = slow::sqrt(d_f_actMag[tag]*d_f_actMag[tag]) + dot(f, vecZ);
        quat<Scalar> quati(quatScal, quatVec);
        quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
        d_orientation[idx] = quat_to_scalar4(quati);
        }
    }

//! Kernel for adjusting active force vectors to align parallel to an ellipsoid surface constraint on the GPU
/*! \param group_size number of particles
    \param d_rtag convert global tag to global index
    \param d_groupTags stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_actVec particle active force unit vector
    \param d_t_actVec particle active force unit vector
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
*/
__global__ void gpu_compute_active_force_set_constraints_kernel(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   const Scalar4 *d_pos,
                                                   Scalar3 *d_f_actVec,
                                                   Scalar3 *d_t_actVec,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int tag = d_groupTags[group_idx];
    unsigned int idx = d_rtag[tag];

    EvaluatorConstraintEllipsoid Ellipsoid(P, rx, ry, rz);
    Scalar3 current_pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

    Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
    vec3<Scalar> norm;
    norm = vec3<Scalar>(norm_scalar3);
    Scalar f_dot_prod = d_f_actVec[tag].x * norm.x + d_f_actVec[tag].y * norm.y + d_f_actVec[tag].z * norm.z;
    Scalar t_dot_prod = d_t_actVec[tag].x * norm.x + d_t_actVec[tag].y * norm.y + d_t_actVec[tag].z * norm.z;

    d_f_actVec[tag].x -= norm.x * f_dot_prod;
    d_f_actVec[tag].y -= norm.y * f_dot_prod;
    d_f_actVec[tag].z -= norm.z * f_dot_prod;

    d_t_actVec[tag].x -= norm.x * t_dot_prod;
    d_t_actVec[tag].y -= norm.y * t_dot_prod;
    d_t_actVec[tag].z -= norm.z * t_dot_prod;


    Scalar new_f_norm = slow::sqrt(d_f_actVec[tag].x * d_f_actVec[tag].x
                                 + d_f_actVec[tag].y * d_f_actVec[tag].y
                                 + d_f_actVec[tag].z * d_f_actVec[tag].z);
    Scalar new_t_norm = slow::sqrt(d_t_actVec[tag].x * d_t_actVec[tag].x
                                 + d_t_actVec[tag].y * d_t_actVec[tag].y
                                 + d_t_actVec[tag].z * d_t_actVec[tag].z);


    d_f_actVec[tag].x /= new_f_norm;
    d_f_actVec[tag].y /= new_f_norm;
    d_f_actVec[tag].z /= new_f_norm;

    d_t_actVec[tag].x /= new_t_norm;
    d_t_actVec[tag].y /= new_t_norm;
    d_t_actVec[tag].z /= new_t_norm;

    }

//! Kernel for applying rotational diffusion to active force vectors on the GPU
/*! \param group_size number of particles
    \param d_rtag convert global tag to global index
    \param d_groupTags stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_actVec particle active force unit vector
    \param d_t_actVec particle active torque unit vector
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param is2D check if simulation is 2D or 3D
    \param rotationDiff particle rotational diffusion constant
    \param seed seed for random number generator
*/
__global__ void gpu_compute_active_force_rotational_diffusion_kernel(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   const Scalar4 *d_pos,
                                                   Scalar3 *d_f_actVec,
                                                   Scalar3 *d_t_actVec,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   bool is2D,
                                                   const Scalar rotationDiff,
                                                   const unsigned int timestep,
                                                   const int seed)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int tag = d_groupTags[group_idx];
    unsigned int idx = d_rtag[tag];
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::ActiveForceCompute, seed, tag, timestep);

    if (is2D) // 2D
        {
        Scalar delta_theta; // rotational diffusion angle
        delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);
        Scalar theta; // angle on plane defining orientation of active force vector
        theta = atan2(d_f_actVec[tag].y, d_f_actVec[tag].x);
        theta += delta_theta;
        d_f_actVec[tag].x = cos(theta);
        d_f_actVec[tag].y = sin(theta);
        // in 2D there is only one meaningful direction for torque
        }
    else // 3D: Following Stenhammar, Soft Matter, 2014
        {
        if (rx == 0) // if no constraint
            {
            hoomd::SpherePointGenerator<Scalar> unit_vec;
            vec3<Scalar> rand_vec;
            unit_vec(rng, rand_vec);

            vec3<Scalar> aux_vec;
            aux_vec.x = d_f_actVec[tag].y * rand_vec.z - d_f_actVec[tag].z * rand_vec.y;
            aux_vec.y = d_f_actVec[tag].z * rand_vec.x - d_f_actVec[tag].x * rand_vec.z;
            aux_vec.z = d_f_actVec[tag].x * rand_vec.y - d_f_actVec[tag].y * rand_vec.x;
            Scalar aux_vec_mag = sqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
            aux_vec.x /= aux_vec_mag;
            aux_vec.y /= aux_vec_mag;
            aux_vec.z /= aux_vec_mag;

            vec3<Scalar> current_vec;
            current_vec.x = d_f_actVec[tag].x;
            current_vec.y = d_f_actVec[tag].y;
            current_vec.z = d_f_actVec[tag].z;

            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);
            d_f_actVec[tag].x = cos(delta_theta)*current_vec.x + sin(delta_theta)*aux_vec.x;
            d_f_actVec[tag].y = cos(delta_theta)*current_vec.y + sin(delta_theta)*aux_vec.y;
            d_f_actVec[tag].z = cos(delta_theta)*current_vec.z + sin(delta_theta)*aux_vec.z;

            // torque vector rotates rigidly along with force vector
            d_t_actVec[tag].x = cos(delta_theta)*current_vec.x + sin(delta_theta)*aux_vec.x;
            d_t_actVec[tag].y = cos(delta_theta)*current_vec.y + sin(delta_theta)*aux_vec.y;
            d_t_actVec[tag].z = cos(delta_theta)*current_vec.z + sin(delta_theta)*aux_vec.z;

            }
        else // if constraint
            {
            EvaluatorConstraintEllipsoid Ellipsoid(P, rx, ry, rz);
            Scalar3 current_pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

            Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
            vec3<Scalar> norm;
            norm = vec3<Scalar> (norm_scalar3);

            vec3<Scalar> current_vec;
            current_vec.x = d_f_actVec[tag].x;
            current_vec.y = d_f_actVec[tag].y;
            current_vec.z = d_f_actVec[tag].z;
            vec3<Scalar> aux_vec = cross(current_vec, norm); // aux vec for defining direction that active force vector rotates towards.

            Scalar delta_theta; // rotational diffusion angle
            delta_theta = hoomd::NormalDistribution<Scalar>(rotationDiff)(rng);

            d_f_actVec[tag].x = cos(delta_theta) * current_vec.x + sin(delta_theta) * aux_vec.x;
            d_f_actVec[tag].y = cos(delta_theta) * current_vec.y + sin(delta_theta) * aux_vec.y;
            d_f_actVec[tag].z = cos(delta_theta) * current_vec.z + sin(delta_theta) * aux_vec.z;

            // torque vector rotates rigidly along with force vector
            d_t_actVec[tag].x = cos(delta_theta) * current_vec.x + sin(delta_theta) * aux_vec.x;
            d_t_actVec[tag].y = cos(delta_theta) * current_vec.y + sin(delta_theta) * aux_vec.y;
            d_t_actVec[tag].z = cos(delta_theta) * current_vec.z + sin(delta_theta) * aux_vec.z;

            }
        }
    }


cudaError_t gpu_compute_active_force_set_forces(const unsigned int group_size,
                                           unsigned int *d_rtag,
                                           unsigned int *d_groupTags,
                                           Scalar4 *d_force,
                                           Scalar4 *d_torque,
                                           Scalar4 *d_orientation,
                                           Scalar3 *d_f_actVec,
                                           Scalar *d_f_actMag,
                                           Scalar3 *d_t_actVec,
                                           Scalar *d_t_actMag,
                                           const Scalar3& P,
                                           Scalar rx,
                                           Scalar ry,
                                           Scalar rz,
                                           bool orientationLink,
                                           bool orientationReverseLink,
                                           const unsigned int N,
                                           unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_set_forces_kernel<<< grid, threads>>>( group_size,
                                                                    d_rtag,
                                                                    d_groupTags,
                                                                    d_force,
                                                                    d_torque,
                                                                    d_orientation,
                                                                    d_f_actVec,
                                                                    d_f_actMag,
                                                                    d_t_actVec,
                                                                    d_t_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz,
                                                                    orientationLink,
                                                                    orientationReverseLink,
                                                                    N);
    return cudaSuccess;
    }

cudaError_t gpu_compute_active_force_set_constraints(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_force,
                                                   Scalar4 *d_torque,
                                                   Scalar3 *d_f_actVec,
                                                   Scalar3 *d_t_actVec,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_compute_active_force_set_constraints_kernel<<< grid, threads>>>(group_size,
                                                                    d_rtag,
                                                                    d_groupTags,
                                                                    d_pos,
                                                                    d_f_actVec,
                                                                    d_t_actVec,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);
    return cudaSuccess;
    }

cudaError_t gpu_compute_active_force_rotational_diffusion(const unsigned int group_size,
                                                       unsigned int *d_rtag,
                                                       unsigned int *d_groupTags,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_force,
                                                       Scalar4 *d_torque,
                                                       Scalar3 *d_f_actVec,
                                                       Scalar3 *d_t_actVec,
                                                       const Scalar3& P,
                                                       Scalar rx,
                                                       Scalar ry,
                                                       Scalar rz,
                                                       bool is2D,
                                                       const Scalar rotationDiff,
                                                       const unsigned int timestep,
                                                       const int seed,
                                                       unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_compute_active_force_rotational_diffusion_kernel<<< grid, threads>>>(group_size,
                                                                    d_rtag,
                                                                    d_groupTags,
                                                                    d_pos,
                                                                    d_f_actVec,
                                                                    d_t_actVec,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz,
                                                                    is2D,
                                                                    rotationDiff,
                                                                    timestep,
                                                                    seed);
    return cudaSuccess;
    }



