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

// Maintainer: dnlebard

#include "HarmonicDihedralForceGPU.cuh"
#include "DihedralData.cuh" // SERIOUSLY, DO I NEED THIS HERE??

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic dihedral forces. Used by HarmonicDihedralForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading dihedral parameters
texture<float4, 1, cudaReadModeElementType> dihedral_params_tex;

//! Kernel for caculating harmonic dihedral forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data arrays to calculate forces on
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Dihedral data to use in calculating the forces
*/
extern "C" __global__ 
void gpu_compute_harmonic_dihedral_forces_kernel(float4* d_force,
                                                 float* d_virial,
                                                 gpu_pdata_arrays pdata,
                                                 gpu_boxsize box,
                                                 gpu_dihedraltable_array tlist)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    
    if (idx >= pdata.N)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals = tlist.n_dihedrals[idx];
    
    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    float4 idx_pos = tex1Dfetch(pdata_pos_tex, idx);  // we can be either a, b, or c in the a-b-c-d quartet
    float4 a_pos,b_pos,c_pos, d_pos; // allocate space for the a,b, and c atoms in the a-b-c-d quartet
    
    // initialize the force to 0
    float4 force_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // initialize the virial to 0
    float virial_idx = 0.0f;
    
    // loop over all dihedrals
    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        // the volatile fails to compile in device emulation mode (MEM TRANSFER: 8 bytes)
#ifdef _DEVICEEMU
        uint4 cur_dihedral = tlist.dihedrals[tlist.pitch*dihedral_idx + idx];
        uint1 cur_ABCD = tlist.dihedralABCD[tlist.pitch*dihedral_idx + idx];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint4 cur_dihedral = tlist.dihedrals[tlist.pitch*dihedral_idx + idx];
        volatile uint1 cur_ABCD = tlist.dihedralABCD[tlist.pitch*dihedral_idx + idx];
#endif
        
        int cur_dihedral_x_idx = cur_dihedral.x;
        int cur_dihedral_y_idx = cur_dihedral.y;
        int cur_dihedral_z_idx = cur_dihedral.z;
        int cur_dihedral_type = cur_dihedral.w;
        int cur_dihedral_abcd = cur_ABCD.x;
        
        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        float4 x_pos = tex1Dfetch(pdata_pos_tex, cur_dihedral_x_idx);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 y_pos = tex1Dfetch(pdata_pos_tex, cur_dihedral_y_idx);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 z_pos = tex1Dfetch(pdata_pos_tex, cur_dihedral_z_idx);
        
        if (cur_dihedral_abcd == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            d_pos = z_pos;
            }
        if (cur_dihedral_abcd == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            d_pos = z_pos;
            }
        if (cur_dihedral_abcd == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            d_pos = z_pos;
            }
        if (cur_dihedral_abcd == 3)
            {
            d_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            c_pos = z_pos;
            }
            
        // calculate dr for a-b,c-b,and a-c(FLOPS: 9)
        float dxab = a_pos.x - b_pos.x;
        float dyab = a_pos.y - b_pos.y;
        float dzab = a_pos.z - b_pos.z;
        
        float dxcb = c_pos.x - b_pos.x;
        float dycb = c_pos.y - b_pos.y;
        float dzcb = c_pos.z - b_pos.z;
        
        float dxdc = d_pos.x - c_pos.x;
        float dydc = d_pos.y - c_pos.y;
        float dzdc = d_pos.z - c_pos.z;
        
        dxab -= box.Lx * rintf(dxab * box.Lxinv);
        dxcb -= box.Lx * rintf(dxcb * box.Lxinv);
        dxdc -= box.Lx * rintf(dxdc * box.Lxinv);
        
        dyab -= box.Ly * rintf(dyab * box.Lyinv);
        dycb -= box.Ly * rintf(dycb * box.Lyinv);
        dydc -= box.Ly * rintf(dydc * box.Lyinv);
        
        dzab -= box.Lz * rintf(dzab * box.Lzinv);
        dzcb -= box.Lz * rintf(dzcb * box.Lzinv);
        dzdc -= box.Lz * rintf(dzdc * box.Lzinv);
        
        float dxcbm = -dxcb;
        float dycbm = -dycb;
        float dzcbm = -dzcb;
        
        dxcbm -= box.Lx * rintf(dxcbm * box.Lxinv);
        dycbm -= box.Ly * rintf(dycbm * box.Lyinv);
        dzcbm -= box.Lz * rintf(dzcbm * box.Lzinv);
        
        // get the dihedral parameters (MEM TRANSFER: 12 bytes)
        float4 params = tex1Dfetch(dihedral_params_tex, cur_dihedral_type);
        float K = params.x;
        float sign = params.y;
        float multi = params.z;
        
        // printf("IN CUDA CODE: k = %f sign = %f multi = %f \n",K,sign,multi);
        
        float aax = dyab*dzcbm - dzab*dycbm;
        float aay = dzab*dxcbm - dxab*dzcbm;
        float aaz = dxab*dycbm - dyab*dxcbm;
        
        float bbx = dydc*dzcbm - dzdc*dycbm;
        float bby = dzdc*dxcbm - dxdc*dzcbm;
        float bbz = dxdc*dycbm - dydc*dxcbm;
        
        float raasq = aax*aax + aay*aay + aaz*aaz;
        float rbbsq = bbx*bbx + bby*bby + bbz*bbz;
        float rgsq = dxcbm*dxcbm + dycbm*dycbm + dzcbm*dzcbm;
        float rg = sqrtf(rgsq);
        
        float rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = 0.0f;
        if (rg > 0.0f) rginv = 1.0f/rg;
        if (raasq > 0.0f) raa2inv = 1.0f/raasq;
        if (rbbsq > 0.0f) rbb2inv = 1.0f/rbbsq;
        float rabinv = sqrtf(raa2inv*rbb2inv);
        
        float c_abcd = (aax*bbx + aay*bby + aaz*bbz)*rabinv;
        float s_abcd = rg*rabinv*(aax*dxdc + aay*dydc + aaz*dzdc);
        
        if (c_abcd > 1.0f) c_abcd = 1.0f;
        if (c_abcd < -1.0f) c_abcd = -1.0f;
        
        
        float p = 1.0f;
        float ddfab;
        float dfab = 0.0f;
        int m = __float2int_rn(multi);
        
        for (int jj = 0; jj < m; jj++)
            {
            ddfab = p*c_abcd - dfab*s_abcd;
            dfab = p*s_abcd + dfab*c_abcd;
            p = ddfab;
            }
            
/////////////////////////
// FROM LAMMPS: sin_shift is always 0... so dropping all sin_shift terms!!!!
/////////////////////////
        p *= sign;
        dfab *= sign;
        dfab *= -multi;
        p += 1.0f;
        
        if (multi < 1.0f)
            {
            p =  1.0f + sign;
            dfab = 0.0f;
            }
            
        float fg = dxab*dxcbm + dyab*dycbm + dzab*dzcbm;
        float hg = dxdc*dxcbm + dydc*dycbm + dzdc*dzcbm;
        
        float fga = fg*raa2inv*rginv;
        float hgb = hg*rbb2inv*rginv;
        float gaa = -raa2inv*rg;
        float gbb = rbb2inv*rg;
        
        float dtfx = gaa*aax;
        float dtfy = gaa*aay;
        float dtfz = gaa*aaz;
        float dtgx = fga*aax - hgb*bbx;
        float dtgy = fga*aay - hgb*bby;
        float dtgz = fga*aaz - hgb*bbz;
        float dthx = gbb*bbx;
        float dthy = gbb*bby;
        float dthz = gbb*bbz;
        
        //float df = -K * dfab;
        float df = -K * dfab * float(0.500); // the 0.5 term is for 1/2K in the forces
        
        float sx2 = df*dtgx;
        float sy2 = df*dtgy;
        float sz2 = df*dtgz;
        
        float ffax = df*dtfx;
        float ffay = df*dtfy;
        float ffaz = df*dtfz;
        
        float ffbx = sx2 - ffax;
        float ffby = sy2 - ffay;
        float ffbz = sz2 - ffaz;
        
        float ffdx = df*dthx;
        float ffdy = df*dthy;
        float ffdz = df*dthz;
        
        float ffcx = -sx2 - ffdx;
        float ffcy = -sy2 - ffdy;
        float ffcz = -sz2 - ffdz;
        
        // Now, apply the force to each individual atom a,b,c,d
        // and accumlate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        //float dihedral_eng = p*K*float(1.0/4.0);
        float dihedral_eng = p*K*float(1.0/8.0); // the 1/8th term is (1/2)K * 1/4
        
        float vx = (dxab*ffax) + (dxcb*ffcx) + (dxdc+dxcb)*ffdx;
        float vy = (dyab*ffay) + (dycb*ffcy) + (dydc+dycb)*ffdy;
        float vz = (dzab*ffaz) + (dzcb*ffcz) + (dzdc+dzcb)*ffdz;
        
        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        float dihedral_virial = float(1.0/12.0)*(vx + vy + vz);
        
        if (cur_dihedral_abcd == 0)
            {
            force_idx.x += ffax;
            force_idx.y += ffay;
            force_idx.z += ffaz;
            }
        if (cur_dihedral_abcd == 1)
            {
            force_idx.x += ffbx;
            force_idx.y += ffby;
            force_idx.z += ffbz;
            }
        if (cur_dihedral_abcd == 2)
            {
            force_idx.x += ffcx;
            force_idx.y += ffcy;
            force_idx.z += ffcz;
            }
        if (cur_dihedral_abcd == 3)
            {
            force_idx.x += ffdx;
            force_idx.y += ffdy;
            force_idx.z += ffdz;
            }
            
        force_idx.w += dihedral_eng;
        virial_idx += dihedral_virial;
        }
        
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    d_virial[idx] = virial_idx;
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param ttable List of dihedrals stored on the GPU
    \param d_params K, sign,multiplicity params packed as padded float4 variables
    \param n_dihedral_types Number of dihedral types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float4 element per dihedral type. The x component contains K the spring constant
    and the y component contains sign, and the z component the multiplicity.
*/
cudaError_t gpu_compute_harmonic_dihedral_forces(float4* d_force,
                                                 float* d_virial,
                                                 const gpu_pdata_arrays &pdata,
                                                 const gpu_boxsize &box,
                                                 const gpu_dihedraltable_array &ttable,
                                                 float4 *d_params,
                                                 unsigned int n_dihedral_types,
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
        
    error = cudaBindTexture(0, dihedral_params_tex, d_params, sizeof(float4) * n_dihedral_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_harmonic_dihedral_forces_kernel<<< grid, threads>>>(d_force, d_virial, pdata, box, ttable);
    
    return cudaSuccess;
    }

