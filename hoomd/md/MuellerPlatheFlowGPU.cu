// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "MuellerPlatheFlow.h"
#include "MuellerPlatheFlowGPU.h"
#include "MuellerPlatheFlowGPU.cuh"
#include <assert.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

struct vel_search_un_opt : public thrust::unary_function< const unsigned int,Scalar_Int>
    {
        vel_search_un_opt(const Scalar4*const d_vel,const unsigned int *const d_tag,
                          const unsigned int flow_direction)
            :
            m_vel(d_vel),
            m_tag(d_tag),
            m_flow_direction(flow_direction)
            {}
        const Scalar4*const m_vel;
        const unsigned int*const m_tag;
        const unsigned int m_flow_direction;
        __host__ __device__ Scalar_Int operator()(const unsigned int idx)const
            {
            const unsigned int tag = m_tag[idx];
            Scalar vel;
            switch( m_flow_direction)
                {
                case X: vel = m_vel[idx].x;break;
                case Y: vel = m_vel[idx].y;break;
                case Z: vel = m_vel[idx].z;break;
                }
            Scalar_Int result;
            result.s = vel;
            result.i = tag;
            return result;
            }
    };

template <typename CMP>
struct vel_search_binary_opt : public thrust::binary_function< Scalar_Int, Scalar_Int, Scalar_Int >
    {
        vel_search_binary_opt(const unsigned int*const d_rtag,
                              const Scalar4*const d_pos,
                              const BoxDim gl_box,
                              const unsigned int Nslabs,
                              const unsigned int slab_index,
                              const unsigned int slab_direction,
                              const Scalar_Int invalid)
            : m_rtag(d_rtag),
              m_pos(d_pos),
              m_gl_box(gl_box),
              m_Nslabs(Nslabs),
              m_slab_index(slab_index),
              m_slab_direction(slab_direction),
              m_invalid(invalid)
            {}
        const unsigned int*const m_rtag;
        const Scalar4*const m_pos;
        const BoxDim m_gl_box;
        const unsigned int m_Nslabs;
        const unsigned int m_slab_index;
        const unsigned int m_slab_direction;
        const Scalar_Int m_invalid;

        __host__ __device__ Scalar_Int operator()(const Scalar_Int& a,const Scalar_Int& b)const
            {
            Scalar_Int result = m_invalid;
            //Early exit, if invalid args involved.
            if( a.i == m_invalid.i )
                return b;
            if( b.i == m_invalid.i )
                return a;

            const unsigned int idx_a = m_rtag[a.i];
            const unsigned int idx_b = m_rtag[b.i];

            unsigned int index_a,index_b;
            switch(m_slab_direction)
                {
                case X:
                    index_a = (m_pos[idx_a].x/m_gl_box.getL().x +.5) * m_Nslabs;
                    index_b = (m_pos[idx_b].x/m_gl_box.getL().x +.5) * m_Nslabs;
                    break;
                case Y:
                    index_a = (m_pos[idx_a].y/m_gl_box.getL().y +.5) * m_Nslabs;
                    index_b = (m_pos[idx_b].y/m_gl_box.getL().y +.5) * m_Nslabs;
                    break;
                case Z:
                    index_a = (m_pos[idx_a].z/m_gl_box.getL().z +.5) * m_Nslabs;
                    index_b = (m_pos[idx_b].z/m_gl_box.getL().z +.5) * m_Nslabs;
                    break;
                }
            index_a %= m_Nslabs;
            index_b %= m_Nslabs;

            if( index_a == index_b)
                {
                if( index_a == m_slab_index )
                    {
                    CMP cmp;
                    if( cmp(a.s,b.s) )
                        result = a;
                    else
                        result = b;
                    }
                }
            else
                {
                if( index_a == m_slab_index )
                    result = a;
                if( index_b == m_slab_index )
                    result = b;
                }
            return result;
            }
    };

cudaError_t gpu_search_min_max_velocity(const unsigned int group_size,
                                        const Scalar4*const d_vel,
                                        const Scalar4*const d_pos,
                                        const unsigned int *const d_tag,
                                        const unsigned int *const d_rtag,
                                        const unsigned int *const d_group_members,
                                        const BoxDim gl_box,
                                        const unsigned int Nslabs,
                                        const unsigned int slab_direction,
                                        const unsigned int flow_direction,
                                        const unsigned int max_slab,
                                        const unsigned int min_slab,
                                        Scalar_Int*const last_max_vel,
                                        Scalar_Int*const last_min_vel,
                                        const bool has_max_slab,
                                        const bool has_min_slab,
                                        const unsigned int blocksize)
    {
    thrust::device_ptr<const unsigned int> member_ptr(d_group_members);
    vel_search_un_opt un_opt(d_vel, d_tag, flow_direction);

    if( has_max_slab )
        {
        vel_search_binary_opt<thrust::greater<const Scalar> > max_bin_opt(d_rtag,d_pos,gl_box,Nslabs,
                                                                    max_slab,slab_direction,
                                                                    *last_max_vel);
        Scalar_Int init = *last_max_vel;
        *last_max_vel = thrust::transform_reduce(member_ptr,member_ptr+group_size,
                                                 un_opt,init,max_bin_opt);
        }

    if( has_min_slab )
        {
        vel_search_binary_opt<thrust::less<const Scalar> > min_bin_opt(d_rtag,d_pos,gl_box,Nslabs,
                                                                 min_slab,slab_direction,
                                                                 *last_min_vel);
        Scalar_Int init = *last_min_vel;
        *last_min_vel = thrust::transform_reduce(member_ptr,member_ptr+group_size,
                                                 un_opt,init,min_bin_opt);
        }


    return cudaPeekAtLastError();
    }


void __global__ gpu_update_min_max_velocity_kernel(const unsigned int *const d_rtag,
                                                   Scalar4*const d_vel,
                                                   const unsigned int Ntotal,
                                                   const Scalar_Int last_max_vel,
                                                   const Scalar_Int last_min_vel,
                                                   const unsigned int flow_direction)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1)
        return;

    const unsigned int min_idx = d_rtag[last_min_vel.i];
    const unsigned int max_idx = d_rtag[last_max_vel.i];
    //Is the particle local on the processor?
    //Swap the particles the new velocities.
    if( min_idx < Ntotal)
        {
        switch(flow_direction)
            {
            case X: d_vel[min_idx].x = last_max_vel.s; break;
            case Y: d_vel[min_idx].y = last_max_vel.s; break;
            case Z: d_vel[min_idx].z = last_max_vel.s; break;
            }
        }
    if( max_idx < Ntotal)
        switch(flow_direction)
            {
            case X: d_vel[max_idx].x = last_min_vel.s;
            case Y: d_vel[max_idx].y = last_min_vel.s;
            case Z: d_vel[max_idx].z = last_min_vel.s;
            }
    }

cudaError_t gpu_update_min_max_velocity(const unsigned int *const d_rtag,
                                        Scalar4*const d_vel,
                                        const unsigned int Ntotal,
                                        const Scalar_Int last_max_vel,
                                        const Scalar_Int last_min_vel,
                                        const unsigned int flow_direction)
    {
    dim3 grid( 1, 1, 1);
    dim3 threads(1, 1, 1);

    gpu_update_min_max_velocity_kernel<<<grid,threads>>>(d_rtag, d_vel, Ntotal,
                                                         last_max_vel, last_min_vel,
                                                         flow_direction);

    return cudaPeekAtLastError();
    }
