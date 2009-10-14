/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: dnlebard

#include "gpu_settings.h"
#include "CGCMMAngleForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! small number. cutoff for igoring the angle as being ill defined.
#define SMALL 0.001f

/*! \file CGCMMAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the CGCMM angle forces. Used by CGCMMAngleForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading angle parameters
texture<float2, 1, cudaReadModeElementType> angle_params_tex;

//! Texture for reading angle CGCMM S-R parameters
texture<float2, 1, cudaReadModeElementType> angle_CGCMMsr_tex; // MISSING EPSILON!!! sigma=.x, rcut=.y

//! Texture for reading angle CGCMM Epsilon-pow/pref parameters
texture<float4, 1, cudaReadModeElementType> angle_CGCMMepow_tex; // now with EPSILON=.x, pow1=.y, pow2=.z, pref=.w

//! Kernel for caculating CGCMM angle forces on the GPU
/*! \param force_data Data to write the compute forces to
    \param pdata Particle data arrays to calculate forces on
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
*/
extern "C" __global__ void gpu_compute_CGCMM_angle_forces_kernel(gpu_force_data_arrays force_data,
                                                                 gpu_pdata_arrays pdata,
                                                                 gpu_boxsize box,
                                                                 gpu_angletable_array alist)
    {
    // start by identifying which particle we are to handle
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    
    
    if (idx_local >= pdata.local_num)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = alist.n_angles[idx_local];
    
    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    float4 idx_pos = tex1Dfetch(pdata_pos_tex, idx_global);  // we can be either a, b, or c in the a-b-c triplet
    float4 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet
    
    // initialize the force to 0
    float4 force_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    float fab[3], fcb[3];
    float fac, eac, vacX, vacY, vacZ;
    
    // initialize the virial to 0
    float virial_idx = 0.0f;
    
    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        // the volatile fails to compile in device emulation mode (MEM TRANSFER: 8 bytes)
#ifdef _DEVICEEMU
        uint4 cur_angle = alist.angles[alist.pitch*angle_idx + idx_local];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint4 cur_angle = alist.angles[alist.pitch*angle_idx + idx_local];
#endif
        
        int cur_angle_x_idx = cur_angle.x;
        int cur_angle_y_idx = cur_angle.y;
        
        // store the a and c positions to accumlate their forces
        int cur_angle_type = cur_angle.z;
        int cur_angle_abc = cur_angle.w;
        
        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        float4 x_pos = tex1Dfetch(pdata_pos_tex, cur_angle_x_idx);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 y_pos = tex1Dfetch(pdata_pos_tex, cur_angle_y_idx);
        
        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }
            
        // calculate dr for a-b,c-b,and a-c(FLOPS: 9)
        float dxab = a_pos.x - b_pos.x;
        float dyab = a_pos.y - b_pos.y;
        float dzab = a_pos.z - b_pos.z;
        
        float dxcb = c_pos.x - b_pos.x;
        float dycb = c_pos.y - b_pos.y;
        float dzcb = c_pos.z - b_pos.z;
        
        float dxac = a_pos.x - c_pos.x;
        float dyac = a_pos.y - c_pos.y;
        float dzac = a_pos.z - c_pos.z;
        
        // apply periodic boundary conditions (FLOPS: 36)
        dxab -= box.Lx * rintf(dxab * box.Lxinv);
        dxcb -= box.Lx * rintf(dxcb * box.Lxinv);
        dxac -= box.Lx * rintf(dxac * box.Lxinv);
        
        dyab -= box.Ly * rintf(dyab * box.Lyinv);
        dycb -= box.Ly * rintf(dycb * box.Lyinv);
        dyac -= box.Ly * rintf(dyac * box.Lyinv);
        
        dzab -= box.Lz * rintf(dzab * box.Lzinv);
        dzcb -= box.Lz * rintf(dzcb * box.Lzinv);
        dzac -= box.Lz * rintf(dzac * box.Lzinv);
        
        // get the angle parameters (MEM TRANSFER: 8 bytes)
        float2 params = tex1Dfetch(angle_params_tex, cur_angle_type);
        float K = params.x;
        float t_0 = params.y;
        
        // FLOPS: was 16, now... ?
        float rsqab = dxab*dxab+dyab*dyab+dzab*dzab;
        float rab = sqrtf(rsqab);
        float rsqcb = dxcb*dxcb+dycb*dycb+dzcb*dzcb;
        float rcb = sqrtf(rsqcb);
        float rsqac = dxac*dxac+dyac*dyac+dzac*dzac;
        float rac = sqrtf(rsqac);
        
        float c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
        c_abbc /= rab*rcb;
        
        
        if (c_abbc > 1.0f) c_abbc = 1.0f;
        if (c_abbc < -1.0f) c_abbc = -1.0f;
        
        float s_abbc = sqrtf(1.0f - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0f/s_abbc;
        
        //////////////////////////////////////////
        // THIS CODE DOES THE 1-3 LJ repulsions //
        //////////////////////////////////////////////////////////////////////////////
        fac = 0.0f;
        eac = 0.0f;
        vacX = vacY = vacZ = 0.0f;
        
        // get the angle E-S-R parameters (MEM TRANSFER: 12 bytes)
        const float2 cgSR = tex1Dfetch(angle_CGCMMsr_tex, cur_angle_type);
        
        float cgsigma = cgSR.x;
        float cgrcut = cgSR.y;
        
        if (rac < cgrcut)
            {
            const float4 cgEPOW = tex1Dfetch(angle_CGCMMepow_tex, cur_angle_type);
            
            // get the angle pow/pref parameters (MEM TRANSFER: 12 bytes)
            float cgeps = cgEPOW.x;
            float cgpow1 = cgEPOW.y;
            float cgpow2 = cgEPOW.z;
            float cgpref = cgEPOW.w;
            
            float cgratio = cgsigma/rac;
            // INTERESTING NOTE: __powf has weird behavior depending
            // on the inputted parameters.  Try sigma=2.05, versus sigma=0.05
            // in cgcmm_angle_force_test.cc 4 particle test
            fac = cgpref*cgeps / rsqac * (cgpow1*__powf(cgratio,cgpow1) - cgpow2*__powf(cgratio,cgpow2));
            eac = cgeps + cgpref*cgeps * (__powf(cgratio,cgpow1) - __powf(cgratio,cgpow2));
            
            vacX = fac * dxac*dxac;
            vacY = fac * dyac*dyac;
            vacZ = fac * dzac*dzac;
            }
        //////////////////////////////////////////////////////////////////////////////
        
        // actually calculate the force
        float dth = acosf(c_abbc) - t_0;
        float tk = K*dth;
        
        float a = -1.0f * tk * s_abbc;
        float a11 = a*c_abbc/rsqab;
        float a12 = -a / (rab*rcb);
        float a22 = a*c_abbc / rsqcb;
        
        fab[0] = a11*dxab + a12*dxcb;
        fab[1] = a11*dyab + a12*dycb;
        fab[2] = a11*dzab + a12*dzcb;
        
        fcb[0] = a22*dxcb + a12*dxab;
        fcb[1] = a22*dycb + a12*dyab;
        fcb[2] = a22*dzcb + a12*dzab;
        
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        float angle_eng = (0.5f*tk*dth + eac)*float(1.0f/3.0f);
        
        // do we really need a virial here for harmonic angles?
        // ... if not, this may be wrong...
        float vx = dxab*fab[0] + dxcb*fcb[0] + vacX;
        float vy = dyab*fab[1] + dycb*fcb[1] + vacY;
        float vz = dzab*fab[2] + dzcb*fcb[2] + vacZ;
        
        float angle_virial = float(1.0f/6.0f)*(vx + vy + vz);
        
        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0] + fac*dxac;
            force_idx.y += fab[1] + fac*dyac;
            force_idx.z += fab[2] + fac*dzac;
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0] - fac*dxac;
            force_idx.y += fcb[1] - fac*dyac;
            force_idx.z += fcb[2] - fac*dzac;
            }
            
        force_idx.w += angle_eng;
        virial_idx += angle_virial;
        
        }
        
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    force_data.force[idx_local] = force_idx;
    force_data.virial[idx_local] = virial_idx;
    
    
    
    }

/*! \param force_data Force data on GPU to write forces to
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param d_params K and t_0 params packed as float2 variables
    \param d_CGCMMsr sigma, and rcut packed as a float2
    \param d_CGCMMepow epsilon, pow1, pow2, and prefactor packed as a float4
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
cudaError_t gpu_compute_CGCMM_angle_forces(const gpu_force_data_arrays& force_data,
                                           const gpu_pdata_arrays &pdata,
                                           const gpu_boxsize &box,
                                           const gpu_angletable_array &atable,
                                           float2 *d_params,
                                           float2 *d_CGCMMsr,
                                           float4 *d_CGCMMepow,
                                           unsigned int n_angle_types,
                                           int block_size)
    {
    assert(d_params);
    assert(d_CGCMMsr);
    assert(d_CGCMMepow);
    
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.local_num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(float2) * n_angle_types);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, angle_CGCMMsr_tex, d_CGCMMsr, sizeof(float2) * n_angle_types);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, angle_CGCMMepow_tex, d_CGCMMepow, sizeof(float4) * n_angle_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_CGCMM_angle_forces_kernel<<< grid, threads>>>(force_data, pdata, box, atable);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

