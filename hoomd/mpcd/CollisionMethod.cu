// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cu
 * \brief Defines GPU functions and kernels used by mpcd::CollisionMethod
 */

#include "CollisionMethod.cuh"
#include "hoomd/ParticleData.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {

__global__ void draw_velocities_constituent_particles(Scalar3* d_linmom_accum,
                                                      Scalar3* d_angmom_accum,
                                                      Scalar4* d_alt_vel,
                                                      const Scalar4* d_postype,
                                                      const Scalar4* d_velocity,
                                                      const int3* d_image,
                                                      const unsigned int* d_tag,
                                                      const unsigned int* d_lookup_center,
                                                      const BoxDim global_box,
                                                      const uint64_t timestep,
                                                      const uint16_t seed,
                                                      const Scalar T,
                                                      const unsigned int num_total)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;

    // check if it is a constituent and get central index of rigid body
    const unsigned int central_idx = d_lookup_center[idx];
    if (central_idx == NO_BODY || idx == central_idx)
        {
        return;
        }

    const Scalar mass_const = d_velocity[idx].w;
    const unsigned int tag = d_tag[idx];
    // draw random velocities from normal distribution
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::CollisionMethod, timestep, seed),
                               hoomd::Counter(tag, 1));
    hoomd::NormalDistribution<Scalar> gen(fast::sqrt(T / mass_const), 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);
    d_alt_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, mass_const);

    // get displacement
    const Scalar4 postype_const = d_postype[idx];
    const vec3<Scalar> pos_const(postype_const);
    const int3 img_const = d_image[idx];

    const Scalar4 postype_central = d_postype[central_idx];
    const vec3<Scalar> pos_central(postype_central);
    const int3 img_central = d_image[central_idx];

    vec3<Scalar> displacement = pos_const - pos_central;
    const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                            img_const.y - img_central.y,
                                            img_const.z - img_central.z);
    displacement = global_box.shift(displacement, displacement_img);

    // add to net momentum
    const vec3<Scalar> rand_vel(vel);
    const vec3<Scalar> linmom_change = mass_const * rand_vel;
    const vec3<Scalar> angmom_change = mass_const * cross(displacement, rand_vel);

    // accumulate onto central particle
    atomicAdd(&d_linmom_accum[central_idx].x, linmom_change.x);
    atomicAdd(&d_linmom_accum[central_idx].y, linmom_change.y);
    atomicAdd(&d_linmom_accum[central_idx].z, linmom_change.z);
    atomicAdd(&d_angmom_accum[central_idx].x, angmom_change.x);
    atomicAdd(&d_angmom_accum[central_idx].y, angmom_change.y);
    atomicAdd(&d_angmom_accum[central_idx].z, angmom_change.z);
    }

__global__ void get_net_velocity_rigid_body(const Scalar3* d_linmom_accum,
                                            Scalar3* d_angmom_accum,
                                            Scalar4* d_alt_vel,
                                            const Scalar4* d_velocity,
                                            const Scalar4* d_orientation,
                                            const Scalar3* d_inertia,
                                            const unsigned int* d_rigid_center,
                                            const unsigned int num_centers)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_centers)
        return;

    const unsigned int central_idx = d_rigid_center[idx];

    // store the net linear velocity in AltVelocities
    const Scalar mass = d_velocity[central_idx].w;
    const Scalar3 net_velocity = d_linmom_accum[central_idx] / mass;
    d_alt_vel[central_idx] = make_scalar4(net_velocity.x, net_velocity.y, net_velocity.z, mass);

    // get net angular momentum
    const vec3<Scalar> net_angmom(d_angmom_accum[central_idx]);
    const quat<Scalar> orientation(d_orientation[central_idx]);
    const vec3<Scalar> inertia(d_inertia[central_idx]);

    // rotate to body frame
    vec3<Scalar> net_angvel_body = rotate(conj(orientation), net_angmom);
    if (inertia.x != Scalar(0))
        {
        net_angvel_body.x = net_angvel_body.x / inertia.x;
        }
    else
        {
        net_angvel_body.x = Scalar(0);
        }
    if (inertia.y != Scalar(0))
        {
        net_angvel_body.y = net_angvel_body.y / inertia.y;
        }
    else
        {
        net_angvel_body.y = Scalar(0);
        }
    if (inertia.z != Scalar(0))
        {
        net_angvel_body.z = net_angvel_body.z / inertia.z;
        }
    else
        {
        net_angvel_body.z = Scalar(0);
        }
    const vec3<Scalar> net_angvel_space = rotate(orientation, net_angvel_body);

    // set net angular velocity in space frame
    d_angmom_accum[central_idx] = vec_to_scalar3(net_angvel_space);
    }

__global__ void apply_thermalized_velocity_vectors(const Scalar3* d_angmom_accum,
                                                   const Scalar4* d_alt_vel,
                                                   const Scalar4* d_postype,
                                                   Scalar4* d_velocity,
                                                   const int3* d_image,
                                                   const unsigned int* d_lookup_center,
                                                   const BoxDim global_box,
                                                   const unsigned int num_total)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;

    // check if it is a constituent and get central index of rigid body
    const unsigned int central_idx = d_lookup_center[idx];
    if (central_idx == NO_BODY || idx == central_idx)
        {
        return;
        }

    // get velocities and masses
    Scalar4 vel_constituent = d_velocity[idx];
    const Scalar4 thermal_vel_mass = d_alt_vel[idx];
    vec3<Scalar> thermal_vel(thermal_vel_mass);

    // get net angular and linear velocity
    const Scalar4 net_vel_mass = d_alt_vel[central_idx];
    const vec3<Scalar> net_vel(net_vel_mass);
    const vec3<Scalar> net_angvel_space(d_angmom_accum[central_idx]);

    // get displacement
    const Scalar4 postype_const = d_postype[idx];
    const vec3<Scalar> pos_const(postype_const);
    const int3 img_const = d_image[idx];

    const Scalar4 postype_central = d_postype[central_idx];
    const vec3<Scalar> pos_central(postype_central);
    const int3 img_central = d_image[central_idx];

    vec3<Scalar> displacement = pos_const - pos_central;
    const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                            img_const.y - img_central.y,
                                            img_const.z - img_central.z);
    displacement = global_box.shift(displacement, displacement_img);

    // subtract net velocities
    thermal_vel -= net_vel + cross(net_angvel_space, displacement);

    // add to constituent
    vel_constituent.x += thermal_vel.x;
    vel_constituent.y += thermal_vel.y;
    vel_constituent.z += thermal_vel.z;
    d_velocity[idx] = vel_constituent;
    }

__global__ void store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                        const Scalar4* d_velocity,
                                                        const unsigned int* d_embed_group,
                                                        const unsigned int num_group)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;

    // store the initial velocity
    const unsigned int particle_idx = d_embed_group[idx];
    d_initial_vel[idx] = d_velocity[particle_idx];
    }

__global__ void accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                              Scalar3* d_angmom_accum,
                                              const Scalar4* d_initial_vel,
                                              const unsigned int* d_embed_group,
                                              const Scalar4* d_postype,
                                              const Scalar4* d_velocity,
                                              const int3* d_image,
                                              const unsigned int* d_lookup_center,
                                              const BoxDim global_box,
                                              const unsigned int num_group)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;

    // get the index from the embedded group and check if in a rigid body
    unsigned int particle_idx = d_embed_group[idx];
    const unsigned int central_idx = d_lookup_center[particle_idx];
    if (central_idx == NO_BODY || particle_idx == central_idx)
        {
        return;
        }

    // get velocities and masses
    const Scalar4 vel_mass_const = d_velocity[particle_idx];
    const vec3<Scalar> vel_const(vel_mass_const);
    const Scalar mass_const = vel_mass_const.w;

    // get displacement
    const Scalar4 postype_const = d_postype[particle_idx];
    const vec3<Scalar> pos_const(postype_const);
    const int3 img_const = d_image[particle_idx];

    const Scalar4 postype_central = d_postype[central_idx];
    const vec3<Scalar> pos_central(postype_central);
    const int3 img_central = d_image[central_idx];

    vec3<Scalar> displacement = pos_const - pos_central;
    const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                            img_const.y - img_central.y,
                                            img_const.z - img_central.z);
    displacement = global_box.shift(displacement, displacement_img);

    // change in linear and angular momentum
    const vec3<Scalar> initial_vel_const(d_initial_vel[idx]);
    const vec3<Scalar> linmom_change = mass_const * (vel_const - initial_vel_const);
    const vec3<Scalar> angmom_change = cross(displacement, linmom_change);

    // accumulate onto central particle
    atomicAdd(&d_linmom_accum[central_idx].x, linmom_change.x);
    atomicAdd(&d_linmom_accum[central_idx].y, linmom_change.y);
    atomicAdd(&d_linmom_accum[central_idx].z, linmom_change.z);
    atomicAdd(&d_angmom_accum[central_idx].x, angmom_change.x);
    atomicAdd(&d_angmom_accum[central_idx].y, angmom_change.y);
    atomicAdd(&d_angmom_accum[central_idx].z, angmom_change.z);
    }

__global__ void transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                            Scalar3* d_angmom_accum,
                                            Scalar4* d_velocity,
                                            const Scalar4* d_orientation,
                                            Scalar4* d_angmom,
                                            const Scalar3* d_inertia,
                                            const unsigned int* d_rigid_center,
                                            const unsigned int num_centers)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_centers)
        {
        return;
        }

    const unsigned int central_idx = d_rigid_center[idx];

    // get accumulated momentum for particle
    const Scalar3 linmom_accum(d_linmom_accum[central_idx]);
    const vec3<Scalar> angmom_accum(d_angmom_accum[central_idx]);

    // compute and store new velocity
    Scalar4 vel_mass = d_velocity[central_idx];
    const Scalar mass = vel_mass.w;
    if (mass > 0)
        {
        vel_mass.x += linmom_accum.x / mass;
        vel_mass.y += linmom_accum.y / mass;
        vel_mass.z += linmom_accum.z / mass;
        d_velocity[central_idx] = vel_mass;
        }

    // compute new angular momentum
    quat<Scalar> angmom(d_angmom[central_idx]);
    const quat<Scalar> orientation(d_orientation[central_idx]);

    // convert angular momentum to quaternion and update
    const vec3<Scalar> inertia(d_inertia[central_idx]);
    vec3<Scalar> angmom_change_body = rotate(conj(orientation), angmom_accum);
    if (inertia.x == Scalar(0))
        {
        angmom_change_body.x = Scalar(0);
        }

    if (inertia.y == Scalar(0))
        {
        angmom_change_body.y = Scalar(0);
        }

    if (inertia.z == Scalar(0))
        {
        angmom_change_body.z = Scalar(0);
        }
    angmom += Scalar(2.0) * orientation * quat(0.0, angmom_change_body);

    // save update
    d_angmom[central_idx] = quat_to_scalar4(angmom);
    }
    } // end namespace kernel

cudaError_t draw_velocities_constituent_particles(Scalar3* d_linmom_accum,
                                                  Scalar3* d_angmom_accum,
                                                  Scalar4* d_alt_vel,
                                                  const Scalar4* d_postype,
                                                  const Scalar4* d_velocity,
                                                  const int3* d_image,
                                                  const unsigned int* d_tag,
                                                  const unsigned int* d_lookup_center,
                                                  const BoxDim& global_box,
                                                  const uint64_t timestep,
                                                  const uint16_t seed,
                                                  const Scalar T,
                                                  const unsigned int num_total,
                                                  const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::draw_velocities_constituent_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::draw_velocities_constituent_particles<<<grid, run_block_size>>>(
        d_linmom_accum,
        d_angmom_accum,
        d_alt_vel,
        d_postype,
        d_velocity,
        d_image,
        d_tag,
        d_lookup_center,
        global_box,
        timestep,
        seed,
        T,
        num_total);
    return cudaSuccess;
    }

cudaError_t get_net_velocity_rigid_body(const Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_alt_vel,
                                        const Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_rigid_center,
                                        const unsigned int num_centers,
                                        const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::get_net_velocity_rigid_body);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_centers / run_block_size + 1);
    mpcd::gpu::kernel::get_net_velocity_rigid_body<<<grid, run_block_size>>>(d_linmom_accum,
                                                                             d_angmom_accum,
                                                                             d_alt_vel,
                                                                             d_velocity,
                                                                             d_orientation,
                                                                             d_inertia,
                                                                             d_rigid_center,
                                                                             num_centers);
    return cudaSuccess;
    }

cudaError_t apply_thermalized_velocity_vectors(const Scalar3* d_angmom_accum,
                                               const Scalar4* d_alt_vel,
                                               const Scalar4* d_postype,
                                               Scalar4* d_velocity,
                                               const int3* d_image,
                                               const unsigned int* d_lookup_center,
                                               const BoxDim& global_box,
                                               const unsigned int num_total,
                                               const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::apply_thermalized_velocity_vectors);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::apply_thermalized_velocity_vectors<<<grid, run_block_size>>>(d_angmom_accum,
                                                                                    d_alt_vel,
                                                                                    d_postype,
                                                                                    d_velocity,
                                                                                    d_image,
                                                                                    d_lookup_center,
                                                                                    global_box,
                                                                                    num_total);
    return cudaSuccess;
    }

cudaError_t store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                    const Scalar4* d_velocity,
                                                    const unsigned int* d_embed_group,
                                                    const unsigned int num_group,
                                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::store_initial_embedded_group_velocities);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_group / run_block_size + 1);
    mpcd::gpu::kernel::store_initial_embedded_group_velocities<<<grid, run_block_size>>>(
        d_initial_vel,
        d_velocity,
        d_embed_group,
        num_group);

    return cudaSuccess;
    }

cudaError_t accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                          Scalar3* d_angmom_accum,
                                          const Scalar4* d_initial_vel,
                                          const unsigned int* d_embed_group,
                                          const Scalar4* d_postype,
                                          const Scalar4* d_velocity,
                                          const int3* d_image,
                                          const unsigned int* d_lookup_center,
                                          const BoxDim& global_box,
                                          const unsigned int num_group,
                                          const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::accumulate_rigid_body_momenta);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_group / run_block_size + 1);
    mpcd::gpu::kernel::accumulate_rigid_body_momenta<<<grid, run_block_size>>>(d_linmom_accum,
                                                                               d_angmom_accum,
                                                                               d_initial_vel,
                                                                               d_embed_group,
                                                                               d_postype,
                                                                               d_velocity,
                                                                               d_image,
                                                                               d_lookup_center,
                                                                               global_box,
                                                                               num_group);

    return cudaSuccess;
    }

cudaError_t transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        Scalar4* d_angmom,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_rigid_center,
                                        const unsigned int num_centers,
                                        const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::transfer_rigid_body_momenta);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_centers / run_block_size + 1);
    mpcd::gpu::kernel::transfer_rigid_body_momenta<<<grid, run_block_size>>>(d_linmom_accum,
                                                                             d_angmom_accum,
                                                                             d_velocity,
                                                                             d_orientation,
                                                                             d_angmom,
                                                                             d_inertia,
                                                                             d_rigid_center,
                                                                             num_centers);

    return cudaSuccess;
    }
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
