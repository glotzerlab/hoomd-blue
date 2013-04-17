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

#include "HarmonicDihedralForceGPU.cuh"
#include "DihedralData.cuh" // SERIOUSLY, DO I NEED THIS HERE??

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic dihedral forces. Used by HarmonicDihedralForceComputeGPU.
*/

//! Texture for reading dihedral parameters
texture<float4, 1, cudaReadModeElementType> dihedral_params_tex;

//! Kernel for caculating harmonic dihedral forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
*/
extern "C" __global__ 
void gpu_compute_harmonic_dihedral_forces_kernel(float4* d_force,
                                                 float* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 BoxDim box,
                                                 const uint4 *tlist,
                                                 const uint1 *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals = n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c-d quartet
    Scalar3 idx_pos = make_float3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a,pos_b,pos_c, pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet

    // initialize the force to 0
    float4 force_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // initialize the virial to 0
    float virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = 0.0f;

    // loop over all dihedrals
    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        uint4 cur_dihedral = tlist[pitch*dihedral_idx + idx];
        uint1 cur_ABCD = dihedral_ABCD[pitch*dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.x;
        int cur_dihedral_y_idx = cur_dihedral.y;
        int cur_dihedral_z_idx = cur_dihedral.z;
        int cur_dihedral_type = cur_dihedral.w;
        int cur_dihedral_abcd = cur_ABCD.x;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        float4 x_postype = d_pos[cur_dihedral_x_idx];
        float3 x_pos = make_float3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 y_postype = d_pos[cur_dihedral_y_idx];
        float3 y_pos = make_float3(y_postype.x, y_postype.y, y_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        float4 z_postype = d_pos[cur_dihedral_z_idx];
        float3 z_pos = make_float3(z_postype.x, z_postype.y, z_postype.z);

        if (cur_dihedral_abcd == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 3)
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        float3 dab = pos_a - pos_b;
        float3 dcb = pos_c - pos_b;
        float3 ddc = pos_d - pos_c;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        float3 dcbm = -dcb;
        dcbm = box.minImage(dcbm);

        // get the dihedral parameters (MEM TRANSFER: 12 bytes)
        float4 params = tex1Dfetch(dihedral_params_tex, cur_dihedral_type);
        float K = params.x;
        float sign = params.y;
        float multi = params.z;

        float aax = dab.y*dcbm.z - dab.z*dcbm.y;
        float aay = dab.z*dcbm.x - dab.x*dcbm.z;
        float aaz = dab.x*dcbm.y - dab.y*dcbm.x;

        float bbx = ddc.y*dcbm.z - ddc.z*dcbm.y;
        float bby = ddc.z*dcbm.x - ddc.x*dcbm.z;
        float bbz = ddc.x*dcbm.y - ddc.y*dcbm.x;

        float raasq = aax*aax + aay*aay + aaz*aaz;
        float rbbsq = bbx*bbx + bby*bby + bbz*bbz;
        float rgsq = dcbm.x*dcbm.x + dcbm.y*dcbm.y + dcbm.z*dcbm.z;
        float rg = sqrtf(rgsq);

        float rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = 0.0f;
        if (rg > 0.0f) rginv = 1.0f/rg;
        if (raasq > 0.0f) raa2inv = 1.0f/raasq;
        if (rbbsq > 0.0f) rbb2inv = 1.0f/rbbsq;
        float rabinv = sqrtf(raa2inv*rbb2inv);

        float c_abcd = (aax*bbx + aay*bby + aaz*bbz)*rabinv;
        float s_abcd = rg*rabinv*(aax*ddc.x + aay*ddc.y + aaz*ddc.z);

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

        float fg = dab.x*dcbm.x + dab.y*dcbm.y + dab.z*dcbm.z;
        float hg = ddc.x*dcbm.x + ddc.y*dcbm.y + ddc.z*dcbm.z;

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
        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        float dihedral_virial[6];
        dihedral_virial[0] = float(1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        dihedral_virial[1] = float(1./4.)*(dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        dihedral_virial[2] = float(1./4.)*(dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        dihedral_virial[3] = float(1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        dihedral_virial[4] = float(1./4.)*(dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        dihedral_virial[5] = float(1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);

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
        for (int k = 0; k < 6; k++)
            virial_idx[k] += dihedral_virial[k];
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
    \param d_pos particle positions on the GPU
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
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
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const BoxDim& box,
                                                 const uint4 *tlist,
                                                 const uint1 *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list,
                                                 float4 *d_params,
                                                 unsigned int n_dihedral_types,
                                                 int block_size)
    {
    assert(d_params);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the texture
    cudaError_t error = cudaBindTexture(0, dihedral_params_tex, d_params, sizeof(float4) * n_dihedral_types);
    if (error != cudaSuccess)
        return error;

    // run the kernel
    gpu_compute_harmonic_dihedral_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, box, tlist, dihedral_ABCD, pitch, n_dihedrals_list);

    return cudaSuccess;
    }

