// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MuellerPlatheFlowGPU.cuh"
#include "MuellerPlatheFlowGPU.h"
#include "hoomd/HOOMDMath.h"
#include <assert.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#pragma GCC diagnostic pop

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
struct vel_search_un_opt : public thrust::unary_function<const unsigned int, Scalar3>
    {
    vel_search_un_opt(const Scalar4* const d_vel,
                      const unsigned int* const d_tag,
                      flow_enum::Direction flow_direction)
        : m_vel(d_vel), m_tag(d_tag), m_flow_direction(flow_direction)
        {
        }
    const Scalar4* const m_vel;
    const unsigned int* const m_tag;
    const flow_enum::Direction m_flow_direction;
    __host__ __device__ Scalar3 operator()(const unsigned int idx) const
        {
        const unsigned int tag = m_tag[idx];
        Scalar vel;
        switch (m_flow_direction)
            {
        case flow_enum::X:
            vel = m_vel[idx].x;
            break;
        case flow_enum::Y:
            vel = m_vel[idx].y;
            break;
        case flow_enum::Z:
            vel = m_vel[idx].z;
            break;
            }
        const Scalar mass = m_vel[idx].w;
        vel *= mass;
        Scalar3 result;
        result.x = vel;
        result.y = mass;
        result.z = __int_as_scalar(tag);
        return result;
        }
    };
template<typename CMP>
struct vel_search_binary_opt : public thrust::binary_function<Scalar3, Scalar3, Scalar3>
    {
    vel_search_binary_opt(const unsigned int* const d_rtag,
                          const Scalar4* const d_pos,
                          const BoxDim gl_box,
                          const unsigned int Nslabs,
                          const unsigned int slab_index,
                          const Scalar3 invalid,
                          const flow_enum::Direction slab_direction)
        : m_rtag(d_rtag), m_pos(d_pos), m_gl_box(gl_box), m_Nslabs(Nslabs),
          m_slab_index(slab_index), m_invalid(invalid), m_slab_direction(slab_direction)
        {
        }
    const unsigned int* const m_rtag;
    const Scalar4* const m_pos;
    const BoxDim m_gl_box;
    const unsigned int m_Nslabs;
    const unsigned int m_slab_index;
    const Scalar3 m_invalid;
    const flow_enum::Direction m_slab_direction;

    __host__ __device__ Scalar3 operator()(const Scalar3& a, const Scalar3& b) const
        {
        Scalar3 result = m_invalid;
        // Early exit, if invalid args involved.
        if (a.z == m_invalid.z)
            return b;
        if (b.z == m_invalid.z)
            return a;

        const unsigned int idx_a = m_rtag[__scalar_as_int(a.z)];
        const unsigned int idx_b = m_rtag[__scalar_as_int(b.z)];

        unsigned int index_a, index_b;
        switch (m_slab_direction)
            {
        case flow_enum::X:
            index_a = (m_pos[idx_a].x / m_gl_box.getL().x + .5) * m_Nslabs;
            index_b = (m_pos[idx_b].x / m_gl_box.getL().x + .5) * m_Nslabs;
            break;
        case flow_enum::Y:
            index_a = (m_pos[idx_a].y / m_gl_box.getL().y + .5) * m_Nslabs;
            index_b = (m_pos[idx_b].y / m_gl_box.getL().y + .5) * m_Nslabs;
            break;
        case flow_enum::Z:
            index_a = (m_pos[idx_a].z / m_gl_box.getL().z + .5) * m_Nslabs;
            index_b = (m_pos[idx_b].z / m_gl_box.getL().z + .5) * m_Nslabs;
            break;
            }
        index_a %= m_Nslabs;
        index_b %= m_Nslabs;

        if (index_a == index_b)
            {
            if (index_a == m_slab_index)
                {
                CMP cmp;
                if (cmp(a.x, b.x))
                    result = a;
                else
                    result = b;
                }
            }
        else
            {
            if (index_a == m_slab_index)
                result = a;
            if (index_b == m_slab_index)
                result = b;
            }
        return result;
        }
    };

hipError_t gpu_search_min_max_velocity(const unsigned int group_size,
                                       const Scalar4* const d_vel,
                                       const Scalar4* const d_pos,
                                       const unsigned int* const d_tag,
                                       const unsigned int* const d_rtag,
                                       const unsigned int* const d_group_members,
                                       const BoxDim gl_box,
                                       const unsigned int Nslabs,
                                       const unsigned int max_slab,
                                       const unsigned int min_slab,
                                       Scalar3* const last_max_vel,
                                       Scalar3* const last_min_vel,
                                       const bool has_max_slab,
                                       const bool has_min_slab,
                                       const unsigned int blocksize,
                                       const flow_enum::Direction flow_direction,
                                       const flow_enum::Direction slab_direction)
    {
    thrust::device_ptr<const unsigned int> member_ptr(d_group_members);

    vel_search_un_opt un_opt(d_vel, d_tag, flow_direction);

    if (has_max_slab)
        {
        vel_search_binary_opt<thrust::greater<const Scalar>>
            max_bin_opt(d_rtag, d_pos, gl_box, Nslabs, max_slab, *last_max_vel, slab_direction);
        Scalar3 init = *last_max_vel;
        *last_max_vel = thrust::transform_reduce(member_ptr,
                                                 member_ptr + group_size,
                                                 un_opt,
                                                 init,
                                                 max_bin_opt);
        }

    if (has_min_slab)
        {
        vel_search_binary_opt<thrust::less<const Scalar>>
            min_bin_opt(d_rtag, d_pos, gl_box, Nslabs, min_slab, *last_min_vel, slab_direction);
        Scalar3 init = *last_min_vel;
        *last_min_vel = thrust::transform_reduce(member_ptr,
                                                 member_ptr + group_size,
                                                 un_opt,
                                                 init,
                                                 min_bin_opt);
        }

    return hipPeekAtLastError();
    }

void __global__ gpu_update_min_max_velocity_kernel(const unsigned int* const d_rtag,
                                                   Scalar4* const d_vel,
                                                   const unsigned int Ntotal,
                                                   const Scalar3 last_max_vel,
                                                   const Scalar3 last_min_vel,
                                                   const flow_enum::Direction flow_direction)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1)
        return;
    const unsigned int min_tag = __scalar_as_int(last_min_vel.z);
    const unsigned int min_idx = d_rtag[min_tag];
    const unsigned int max_tag = __scalar_as_int(last_max_vel.z);
    const unsigned int max_idx = d_rtag[max_tag];
    // Is the particle local on the processor?
    // Swap the particles the new velocities.
    if (min_idx < Ntotal)
        {
        const Scalar new_min_vel = last_max_vel.x / last_min_vel.y;
        switch (flow_direction)
            {
        case flow_enum::X:
            d_vel[min_idx].x = new_min_vel;
            break;
        case flow_enum::Y:
            d_vel[min_idx].y = new_min_vel;
            break;
        case flow_enum::Z:
            d_vel[min_idx].z = new_min_vel;
            break;
            }
        }

    if (max_idx < Ntotal)
        {
        const Scalar new_max_vel = last_min_vel.x / last_max_vel.y;
        switch (flow_direction)
            {
        case flow_enum::X:
            d_vel[max_idx].x = new_max_vel;
            break;
        case flow_enum::Y:
            d_vel[max_idx].y = new_max_vel;
            break;
        case flow_enum::Z:
            d_vel[max_idx].z = new_max_vel;
            break;
            }
        }
    }

hipError_t gpu_update_min_max_velocity(const unsigned int* const d_rtag,
                                       Scalar4* const d_vel,
                                       const unsigned int Ntotal,
                                       const Scalar3 last_max_vel,
                                       const Scalar3 last_min_vel,
                                       const flow_enum::Direction flow_direction)
    {
    dim3 grid(1, 1, 1);
    dim3 threads(1, 1, 1);

    hipLaunchKernelGGL((gpu_update_min_max_velocity_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_rtag,
                       d_vel,
                       Ntotal,
                       last_max_vel,
                       last_min_vel,
                       flow_direction);

    return hipPeekAtLastError();
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
