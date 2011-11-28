/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include "Integrator.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file Integrator.cu
    \brief Defines methods and data structures used by the Integrator class on the GPU
*/

//! helper to add a given force/virial pointer pair
__device__ void add_force_total(float4& net_force, float *net_virial, float4& net_torque, float4* d_f, float* d_v, const unsigned int virial_pitch, float4* d_t, int idx)
    {
    if (d_f != NULL && d_v != NULL && d_t != NULL)
        {
        float4 f = d_f[idx];
        float4 t = d_t[idx];
        
        net_force.x += f.x;
        net_force.y += f.y;
        net_force.z += f.z;
        net_force.w += f.w;

        for (int i=0; i < 6; i++)
            net_virial[i] += d_v[i*virial_pitch+idx];
        
        net_torque.x += t.x;
        net_torque.y += t.y;
        net_torque.z += t.z;
        net_torque.w += t.w;
        }
    }

//! Kernel for summing forces on the GPU
/*! The speicified forces and virials are summed for every particle into \a d_net_force and \a d_net_virial

    \param d_net_force Output device array to hold the computed net force
    \param d_net_virial Output device array to hold the computed net virial
    \param net_virial_pitch The pitch of the 2D net_virial array
    \param d_net_torque Output device array to hold the computed net torque
    \param force_list List of pointers to force data to sum
    \param nparticles Number of particles in the arrays
    \param clear When true, initializes the sums to 0 before adding. When false, reads in the current \a d_net_force
           and \a d_net_virial and adds to that
*/
__global__ void gpu_integrator_sum_net_force_kernel(float4 *d_net_force,
                                                    float *d_net_virial,
                                                    const unsigned int net_virial_pitch,
                                                    float4 *d_net_torque,
                                                    const gpu_force_list force_list,
                                                    unsigned int nparticles,
                                                    bool clear)
    {
    // calculate the index we will be handling
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < nparticles)
        {
        // set the initial net_force and net_virial to sum into
        float4 net_force;
        float net_virial[6];
        float4 net_torque;
        if (clear)
            {
            net_force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i=0; i<6; i++)
                net_virial[i] = 0.0f;
            net_torque = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        else
            {
            // if clear is false, intialize to the current d_net_force and d_net_virial
            net_force = d_net_force[idx];
            for (int i=0; i<6; i++)
                net_virial[i] = d_net_virial[i*net_virial_pitch+idx];
            net_torque = d_net_torque[idx];
            }
        
        // sum up the totals
        add_force_total(net_force, net_virial, net_torque, force_list.f0, force_list.v0, force_list.vpitch0, force_list.t0, idx);
        add_force_total(net_force, net_virial, net_torque, force_list.f1, force_list.v1, force_list.vpitch1, force_list.t1, idx);
        add_force_total(net_force, net_virial, net_torque, force_list.f2, force_list.v2, force_list.vpitch2, force_list.t2, idx);
        add_force_total(net_force, net_virial, net_torque, force_list.f3, force_list.v3, force_list.vpitch3, force_list.t3, idx);
        add_force_total(net_force, net_virial, net_torque, force_list.f4, force_list.v4, force_list.vpitch4, force_list.t4, idx);
        add_force_total(net_force, net_virial, net_torque, force_list.f5, force_list.v5, force_list.vpitch5, force_list.t5, idx);
        
        // write out the final result
        d_net_force[idx] = net_force;
        for (int i=0; i < 6; i++)
            d_net_virial[i*net_virial_pitch+idx] = net_virial[i];
        d_net_torque[idx] = net_torque;
        }
    }

cudaError_t gpu_integrator_sum_net_force(float4 *d_net_force,
                                         float *d_net_virial,
                                         const unsigned int net_virial_pitch,
                                         float4 *d_net_torque,
                                         const gpu_force_list& force_list,
                                         unsigned int nparticles,
                                         bool clear)
    {
    // sanity check
    assert(d_net_force);
    assert(d_net_virial);
    assert(d_net_torque);
    
    const int block_size = 256;
    
    gpu_integrator_sum_net_force_kernel<<< nparticles/block_size+1, block_size >>>
        (d_net_force, d_net_virial, net_virial_pitch, d_net_torque, force_list, nparticles, clear);
    
    return cudaSuccess;
    }

