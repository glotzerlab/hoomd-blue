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

#include "HarmonicImproperForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicImproperForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic improper forces. Used by HarmonicImproperForceComputeGPU.
*/

//! Texture for reading improper parameters
texture<float2, 1, cudaReadModeElementType> improper_params_tex;

//! Kernel for caculating harmonic improper forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial
    \param N number of particles
    \param d_pos Device memory of particle positions
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Improper data to use in calculating the forces
*/
extern "C" __global__ 
void gpu_compute_harmonic_improper_forces_kernel(float4* d_force,
                                                 float* d_virial,
                                                 const unsigned int virial_pitch,
                                                 unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 gpu_boxsize box,
                                                 gpu_dihedraltable_array tlist)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= N)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_impropers = tlist.n_dihedrals[idx];
    
    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    float4 idx_pos = d_pos[idx];  // we can be either a, b, or c in the a-b-c-d quartet
    float4 pos_a,pos_b,pos_c, pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet
    
    // initialize the force to 0
    float4 force_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // initialize the virial to 0
    float virial_idx[6];
    for (int i = 0; i < 6; i++)
        virial_idx[i] = 0.0f;
    
    // loop over all impropers
    for (int improper_idx = 0; improper_idx < n_impropers; improper_idx++)
        {
        // the volatile fails to compile in device emulation mode (MEM TRANSFER: 8 bytes)
#ifdef _DEVICEEMU
        uint4 cur_improper = tlist.dihedrals[tlist.pitch*improper_idx + idx];
        uint1 cur_ABCD = tlist.dihedralABCD[tlist.pitch*improper_idx + idx];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint4 cur_improper = tlist.dihedrals[tlist.pitch*improper_idx + idx];
        volatile uint1 cur_ABCD = tlist.dihedralABCD[tlist.pitch*improper_idx + idx];
#endif
        
        int cur_improper_x_idx = cur_improper.x;
        int cur_improper_y_idx = cur_improper.y;
        int cur_improper_z_idx = cur_improper.z;
        int cur_improper_type = cur_improper.w;
        int cur_improper_abcd = cur_ABCD.x;
        
        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        float4 x_pos = d_pos[cur_improper_x_idx];
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 y_pos = d_pos[cur_improper_y_idx];
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 z_pos = d_pos[cur_improper_z_idx];
        
        if (cur_improper_abcd == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 3)
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }
            
        // calculate dr for a-b,c-b,and a-c(FLOPS: 9)
        float dxab = pos_a.x - pos_b.x;
        float dyab = pos_a.y - pos_b.y;
        float dzab = pos_a.z - pos_b.z;
        
        float dxcb = pos_c.x - pos_b.x;
        float dycb = pos_c.y - pos_b.y;
        float dzcb = pos_c.z - pos_b.z;
        
        float dxdc = pos_d.x - pos_c.x;
        float dydc = pos_d.y - pos_c.y;
        float dzdc = pos_d.z - pos_c.z;
        
        dxab -= box.Lx * rintf(dxab * box.Lxinv);
        dxcb -= box.Lx * rintf(dxcb * box.Lxinv);
        dxdc -= box.Lx * rintf(dxdc * box.Lxinv);
        
        dyab -= box.Ly * rintf(dyab * box.Lyinv);
        dycb -= box.Ly * rintf(dycb * box.Lyinv);
        dydc -= box.Ly * rintf(dydc * box.Lyinv);
        
        dzab -= box.Lz * rintf(dzab * box.Lzinv);
        dzcb -= box.Lz * rintf(dzcb * box.Lzinv);
        dzdc -= box.Lz * rintf(dzdc * box.Lzinv);
        
        
        // get the improper parameters (MEM TRANSFER: 12 bytes)
        float2 params = tex1Dfetch(improper_params_tex, cur_improper_type);
        float K = params.x;
        float chi = params.y;
        
        float r1 = rsqrtf(dxab*dxab + dyab*dyab + dzab*dzab);
        float r2 = rsqrtf(dxcb*dxcb + dycb*dycb + dzcb*dzcb);
        float r3 = rsqrtf(dxdc*dxdc + dydc*dydc + dzdc*dzdc);
        
        float ss1 = r1 * r1;
        float ss2 = r2 * r2;
        float ss3 = r3 * r3;
        
        // Cosine and Sin of the angle between the planes
        float c0 = (dxab*dxdc + dyab*dydc + dzab*dzdc)* r1 * r3;
        float c1 = (dxab*dxcb + dyab*dycb + dzab*dzcb)* r1 * r2;
        float c2 = -(dxdc*dxcb + dydc*dycb + dzdc*dzcb)* r3 * r2;
        
        float s1 = 1.0f - c1*c1;
        if (s1 < SMALL) s1 = SMALL;
        s1 = 1.0f / s1;
        
        float s2 = 1.0f - c2*c2;
        if (s2 < SMALL) s2 = SMALL;
        s2 = 1.0f / s2;
        
        float s12 = sqrt(s1*s2);
        float c = (c1*c2 + c0) * s12;
        
        if (c > 1.0f) c = 1.0f;
        if (c < -1.0f) c = -1.0f;
        
        float s = sqrt(1.0f - c*c);
        if (s < SMALL) s = SMALL;
        
        float domega = acosf(c) - chi;
        float a = K * domega;
        
        // calculate the energy, 1/4th for each atom
        //float improper_eng = 0.25*a*domega;
        float improper_eng = 0.125f*a*domega;  // the .125 term is 1/2 * 1/4
        
        //a = -a * 2.0/s;
        a = -a /s; // the missing 2.0 factor is to ensure K/2 is factored in for the forces
        c = c * a;
        s12 = s12 * a;
        float a11 = c*ss1*s1;
        float a22 = -ss2 * (2.0f*c0*s12 - c*(s1+s2));
        float a33 = c*ss3*s2;
        
        float a12 = -r1*r2*(c1*c*s1 + c2*s12);
        float a13 = -r1*r3*s12;
        float a23 = r2*r3*(c2*c*s2 + c1*s12);
        
        float sx2  = a22*dxcb + a23*dxdc + a12*dxab;
        float sy2  = a22*dycb + a23*dydc + a12*dyab;
        float sz2  = a22*dzcb + a23*dzdc + a12*dzab;
        
        // calculate the forces for each particle
        float ffax = a12*dxcb + a13*dxdc + a11*dxab;
        float ffay = a12*dycb + a13*dydc + a11*dyab;
        float ffaz = a12*dzcb + a13*dzdc + a11*dzab;
        
        float ffbx = -sx2 - ffax;
        float ffby = -sy2 - ffay;
        float ffbz = -sz2 - ffaz;
        
        float ffdx = a23*dxcb + a33*dxdc + a13*dxab;
        float ffdy = a23*dycb + a33*dydc + a13*dyab;
        float ffdz = a23*dzcb + a33*dzdc + a13*dzab;
        
        float ffcx = sx2 - ffdx;
        float ffcy = sy2 - ffdy;
        float ffcz = sz2 - ffdz;
        
        // and calculate the virial (symmetrized version)
        float improper_virial[6];
        improper_virial[0] = float(1./4.)*(dxab*ffax + dxcb*ffcx + (dxdc+dxcb)*ffdx);
        improper_virial[1] = float(1./8.)*((dxab*ffay + dxcb*ffcy + (dxdc+dxcb)*ffdy)
                                     +(dyab*ffax + dycb*ffcx + (dydc+dycb)*ffdx));
        improper_virial[2] = float(1./8.)*((dxab*ffaz + dxcb*ffcz + (dxdc+dxcb)*ffdz)
                                     +(dzab*ffax + dzcb*ffcx + (dzdc+dzcb)*ffdx));
        improper_virial[3] = float(1./4.)*(dyab*ffay + dycb*ffcy + (dydc+dycb)*ffdy);
        improper_virial[4] = float(1./8.)*((dyab*ffaz + dycb*ffcz + (dydc+dycb)*ffdz)
                                     +(dzab*ffay + dzcb*ffcy + (dzdc+dzcb)*ffdy));
        improper_virial[5] = float(1./4.)*(dzab*ffaz + dzcb*ffcz + (dzdc+dzcb)*ffdz);

        
        if (cur_improper_abcd == 0)
            {
            force_idx.x += ffax;
            force_idx.y += ffay;
            force_idx.z += ffaz;
            }
        if (cur_improper_abcd == 1)
            {
            force_idx.x += ffbx;
            force_idx.y += ffby;
            force_idx.z += ffbz;
            }
        if (cur_improper_abcd == 2)
            {
            force_idx.x += ffcx;
            force_idx.y += ffcy;
            force_idx.z += ffcz;
            }
        if (cur_improper_abcd == 3)
            {
            force_idx.x += ffdx;
            force_idx.y += ffdy;
            force_idx.z += ffdz;
            }
            
        force_idx.w += improper_eng;
        for (int k = 0; k < 6; k++)
            virial_idx[k] += improper_virial[k];
        }
        
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int k = 0; k < 6; k++)
        d_virial[k*virial_pitch+idx] = virial_idx[k];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param ttable List of impropers stored on the GPU
    \param d_params K, sign,multiplicity params packed as padded float4 variables
    \param n_improper_types Number of improper types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float4 element per improper type. The x component contains K the spring constant
    and the y component contains sign, and the z component the multiplicity.
*/
cudaError_t gpu_compute_harmonic_improper_forces(float4* d_force,
                                                 float* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const gpu_boxsize &box,
                                                 const gpu_dihedraltable_array &ttable,
                                                 float2 *d_params,
                                                 unsigned int n_improper_types,
                                                 int block_size)
    {
    assert(d_params);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the texture
    cudaError_t error = cudaBindTexture(0, improper_params_tex, d_params, sizeof(float2) * n_improper_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_harmonic_improper_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, box, ttable);
    
    return cudaSuccess;
    }

