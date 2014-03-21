/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#include "CGCMMAngleForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// small number. cutoff for igoring the angle as being ill defined.
#define SMALL Scalar(0.001)

/*! \file CGCMMAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the CGCMM angle forces. Used by CGCMMAngleForceComputeGPU.
*/

//! Texture for reading angle parameters
scalar2_tex_t angle_params_tex;

//! Texture for reading angle CGCMM S-R parameters
scalar2_tex_t angle_CGCMMsr_tex; // MISSING EPSILON!!! sigma=.x, rcut=.y

//! Texture for reading angle CGCMM Epsilon-pow/pref parameters
scalar4_tex_t angle_CGCMMepow_tex; // now with EPSILON=.x, pow1=.y, pow2=.z, pref=.w

//! Kernel for caculating CGCMM angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/
extern "C" __global__ void gpu_compute_CGCMM_angle_forces_kernel(Scalar4* d_force,
                                                                 Scalar* d_virial,
                                                                 const unsigned int virial_pitch,
                                                                 const unsigned int N,
                                                                 const Scalar4 *d_pos,
                                                                 BoxDim box,
                                                                 const group_storage<3> *alist,
                                                                 const unsigned int *apos_list,
                                                                 const unsigned int pitch,
                                                                 const unsigned int *n_angles_list,
                                                                 Scalar2 *d_params,
                                                                 Scalar2 *d_CGCMMsr,
                                                                 Scalar4 *d_CGCMMepow)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles =n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar fab[3], fcb[3];
    Scalar fac, eac, vac[6];

    // initialize the virial to 0
    Scalar virial_idx[6];
    for (int i = 0; i < 6; i++)
        virial_idx[i] = Scalar(0.0);

    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];

        // store the a and c positions to accumlate their forces
        int cur_angle_type = cur_angle.idx[2];
        int cur_angle_abc = apos_list[pitch*angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

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

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = a_pos - b_pos;
        Scalar3 dcb = c_pos - b_pos;
        Scalar3 dac = a_pos - c_pos;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // get the angle parameters (MEM TRANSFER: 8 bytes)
        Scalar2 params = texFetchScalar2(d_params, angle_params_tex, cur_angle_type);
        Scalar K = params.x;
        Scalar t_0 = params.y;

        Scalar rsqab = dot(dab, dab);
        Scalar rab = sqrtf(rsqab);
        Scalar rsqcb = dot(dcb, dcb);;
        Scalar rcb = sqrtf(rsqcb);
        Scalar rsqac = dot(dac, dac);
        Scalar rac = sqrtf(rsqac);

        Scalar c_abbc = dot(dab, dcb);
        c_abbc /= rab*rcb;

        if (c_abbc > Scalar(1.0)) c_abbc = Scalar(1.0);
        if (c_abbc < -Scalar(1.0)) c_abbc = -Scalar(1.0);

        Scalar s_abbc = sqrtf(Scalar(1.0) - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = Scalar(1.0)/s_abbc;

        //////////////////////////////////////////
        // THIS CODE DOES THE 1-3 LJ repulsions //
        //////////////////////////////////////////////////////////////////////////////
        fac = Scalar(0.0);
        eac = Scalar(0.0);
        for (int i=0; i < 6; i++)
            vac[i] = Scalar(0.0);

        // get the angle E-S-R parameters (MEM TRANSFER: 12 bytes)
        const Scalar2 cgSR = texFetchScalar2(d_CGCMMsr, angle_CGCMMsr_tex, cur_angle_type);

        Scalar cgsigma = cgSR.x;
        Scalar cgrcut = cgSR.y;

        if (rac < cgrcut)
            {
            const Scalar4 cgEPOW = texFetchScalar4(d_CGCMMepow, angle_CGCMMepow_tex, cur_angle_type);

            // get the angle pow/pref parameters (MEM TRANSFER: 12 bytes)
            Scalar cgeps = cgEPOW.x;
            Scalar cgpow1 = cgEPOW.y;
            Scalar cgpow2 = cgEPOW.z;
            Scalar cgpref = cgEPOW.w;

            Scalar cgratio = cgsigma/rac;
            // INTERESTING NOTE: POW has weird behavior depending
            // on the inputted parameters.  Try sigma=2.05, versus sigma=0.05
            // in cgcmm_angle_force_test.cc 4 particle test
            fac = cgpref*cgeps / rsqac * (cgpow1*fast::pow(cgratio,cgpow1) - cgpow2*fast::pow(cgratio,cgpow2));
            eac = cgeps + cgpref*cgeps * (fast::pow(cgratio,cgpow1) - fast::pow(cgratio,cgpow2));

            vac[0] = fac * dac.x*dac.x;
            vac[1] = fac * dac.x*dac.y;
            vac[2] = fac * dac.x*dac.z;
            vac[3] = fac * dac.y*dac.y;
            vac[4] = fac * dac.y*dac.z;
            vac[5] = fac * dac.z*dac.z;
            }
        //////////////////////////////////////////////////////////////////////////////

        // actually calculate the force
        Scalar dth = fast::acos(c_abbc) - t_0;
        Scalar tk = K*dth;

        Scalar a = -Scalar(1.0) * tk * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;

        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;

        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (Scalar(0.5)*tk*dth + eac)*Scalar(Scalar(1.0)/Scalar(3.0));

        Scalar angle_virial[6];
        angle_virial[0] = (1.f/3.f) * ( dab.x*fab[0] + dcb.x*fcb[0] );
        angle_virial[1] = (1.f/3.f) * ( dab.y*fab[0] + dcb.y*fcb[0] );
        angle_virial[2] = (1.f/3.f) * ( dab.z*fab[0] + dcb.z*fcb[0] );
        angle_virial[3] = (1.f/3.f) * ( dab.y*fab[1] + dcb.y*fcb[1] );
        angle_virial[4] = (1.f/3.f) * ( dab.z*fab[1] + dcb.z*fcb[1] );
        angle_virial[5] = (1.f/3.f) * ( dab.z*fab[2] + dcb.z*fcb[2] );

        for (int i = 0; i < 6; i++)
            angle_virial[i] += (1.f/3.f)*vac[i];

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0] + fac*dac.x;
            force_idx.y += fab[1] + fac*dac.y;
            force_idx.z += fab[2] + fac*dac.z;
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0] - fac*dac.x;
            force_idx.y += fcb[1] - fac*dac.y;
            force_idx.z += fcb[2] - fac*dac.z;
            }

        force_idx.w += angle_eng;
        for (int i = 0; i < 6; i++)
            virial_idx[i] += angle_virial[i];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        d_virial[i*virial_pitch+idx] = virial_idx[i];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params K and t_0 params packed as Scalar2 variables
    \param d_CGCMMsr sigma, and rcut packed as a Scalar2
    \param d_CGCMMepow epsilon, pow1, pow2, and prefactor packed as a Scalar4
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
cudaError_t gpu_compute_CGCMM_angle_forces(Scalar4* d_force,
                                           Scalar* d_virial,
                                           const unsigned int virial_pitch,
                                           const unsigned int N,
                                           const Scalar4 *d_pos,
                                           const BoxDim& box,
                                           const group_storage<3> *atable,
                                           const unsigned int *apos_list,
                                           const unsigned int pitch,
                                           const unsigned int *n_angles_list,
                                           Scalar2 *d_params,
                                           Scalar2 *d_CGCMMsr,
                                           Scalar4 *d_CGCMMepow,
                                           unsigned int n_angle_types,
                                           int block_size,
                                           const unsigned int compute_capability)
    {
    assert(d_params);
    assert(d_CGCMMsr);
    assert(d_CGCMMepow);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_compute_CGCMM_angle_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)run_block_size), 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the textures on pre sm 35 arches
    if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMsr_tex, d_CGCMMsr, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMepow_tex, d_CGCMMepow, sizeof(Scalar4) * n_angle_types);
        if (error != cudaSuccess)
            return error;
        }

    // run the kernel
    gpu_compute_CGCMM_angle_forces_kernel<<< grid, threads>>>(d_force,
                                                              d_virial,
                                                              virial_pitch,
                                                              N,
                                                              d_pos,
                                                              box,
                                                              atable,
                                                              apos_list,
                                                              pitch,
                                                              n_angles_list,
                                                              d_params,
                                                              d_CGCMMsr,
                                                              d_CGCMMepow);

    return cudaSuccess;
    }
