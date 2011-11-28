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

// Maintainer: dnlebard

#include "HarmonicAngleForceGPU.cuh"
#include "AngleData.cuh" // SERIOUSLY, DO I NEED THIS HERE??

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic angle forces. Used by HarmonicAngleForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading angle parameters
texture<float2, 1, cudaReadModeElementType> angle_params_tex;

//! Kernel for caculating harmonic angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param pdata Particle data arrays to calculate forces on
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
*/
extern "C" __global__ void gpu_compute_harmonic_angle_forces_kernel(float4* d_force,
                                                                    float* d_virial,
                                                                    const unsigned int virial_pitch,
                                                                    gpu_pdata_arrays pdata,
                                                                    gpu_boxsize box,
                                                                    gpu_angletable_array alist)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    
    if (idx >= pdata.N)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = alist.n_angles[idx];
    
    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    float4 idx_pos = tex1Dfetch(pdata_pos_tex, idx);  // we can be either a, b, or c in the a-b-c triplet
    float4 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet
    
    // initialize the force to 0
    float4 force_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    float fab[3], fcb[3];
    
    // initialize the virial to 0
    float virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = 0.0f;
    
    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        // the volatile fails to compile in device emulation mode (MEM TRANSFER: 8 bytes)
#ifdef _DEVICEEMU
        uint4 cur_angle = alist.angles[alist.pitch*angle_idx + idx];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint4 cur_angle = alist.angles[alist.pitch*angle_idx + idx];
#endif
        
        int cur_angle_x_idx = cur_angle.x;
        int cur_angle_y_idx = cur_angle.y;
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
        
        float c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
        c_abbc /= rab*rcb;
        
        
        if (c_abbc > 1.0f) c_abbc = 1.0f;
        if (c_abbc < -1.0f) c_abbc = -1.0f;
        
        float s_abbc = sqrtf(1.0f - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0f/s_abbc;
        
        
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
        float angle_eng = tk*dth*float(1.0f/6.0f);

        // do we really need a virial here for harmonic angles?
        // ... if not, this may be wrong...
        // compute 1/3 of the virial, 1/3 for each atom in the angle
        float angle_virial[6];
        angle_virial[0] = float(1./3.) * ( dxab*fab[0] + dxcb*fcb[0] );
        angle_virial[1] = float(1./6.) * ( dxab*fab[1] + dxcb*fcb[1]
                                          + dyab*fab[0] + dycb*fcb[0] );
        angle_virial[2] = float(1./6.) * ( dxab*fab[2] + dxcb*fcb[2]
                                          + dzab*fab[0] + dzcb*fcb[0] );
        angle_virial[3] = float(1./3.) * ( dyab*fab[1] + dycb*fcb[1] );
        angle_virial[4] = float(1./6.) * ( dyab*fab[2] + dycb*fcb[2]
                                          + dzab*fab[1] + dzcb*fcb[1] );
        angle_virial[5] = float(1./3.) * ( dzab*fab[2] + dzcb*fcb[2] );


        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }
            
        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }
        
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        d_virial[i*virial_pitch+idx] = virial[idx];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial arary
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param d_params K and t_0 params packed as float2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
cudaError_t gpu_compute_harmonic_angle_forces(float4* d_force,
                                              float* d_virial,
                                              const unsigned int virial_pitch,
                                              const gpu_pdata_arrays &pdata,
                                              const gpu_boxsize &box,
                                              const gpu_angletable_array &atable,
                                              float2 *d_params,
                                              unsigned int n_angle_types,
                                              int block_size)
    {
    assert(d_params);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(float2) * n_angle_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_harmonic_angle_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, pdata, box, atable);
    
    return cudaSuccess;
    }

