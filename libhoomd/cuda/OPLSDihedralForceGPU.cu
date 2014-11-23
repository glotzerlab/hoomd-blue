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

// Maintainer: ksil

#include "OPLSDihedralForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// SMALL a relatively small number
#define SMALL     0.001
#define SMALLER   0.00001

/*! \file OPLSDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating OPLS dihedral forces. Used by OPLSDihedralForceComputeGPU.
*/

//! Texture for reading dihedral parameters
scalar4_tex_t dihedral_params_tex;

//! Kernel for caculating OPLS dihedral forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param d_params Array of OPLS parameters k1/2, k2/2, k3/2, and k4/2
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
*/
extern "C" __global__
void gpu_compute_opls_dihedral_forces_kernel(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const Scalar4 *d_params,
                                                 BoxDim box,
                                                 const group_storage<4> *tlist,
                                                 const unsigned int *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals = n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c-d set. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c-d quartet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a,pos_b,pos_c, pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = Scalar(0.0);

    // loop over all dihedrals
    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = tlist[pitch*dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch*dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];
        int cur_dihedral_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_dihedral_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_dihedral_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 z_postype = d_pos[cur_dihedral_z_idx];
        Scalar3 z_pos = make_scalar3(z_postype.x, z_postype.y, z_postype.z);

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

        // the three bonds
        
        Scalar3 vb1 = pos_a - pos_b;
        Scalar3 vb2 = pos_c - pos_b;
        Scalar3 vb3 = pos_d - pos_c;
        
        // apply periodic boundary conditions
        vb1 = box.minImage(vb1);
        vb2 = box.minImage(vb2);
        vb3 = box.minImage(vb3);
        
        Scalar3 vb2m = -vb2;
        vb2m = box.minImage(vb2m);

        // c0 calculation

        Scalar sb1 = 1.0 / (vb1.x*vb1.x + vb1.y*vb1.y + vb1.z*vb1.z);
        Scalar sb2 = 1.0 / (vb2.x*vb2.x + vb2.y*vb2.y + vb2.z*vb2.z);
        Scalar sb3 = 1.0 / (vb3.x*vb3.x + vb3.y*vb3.y + vb3.z*vb3.z);

        Scalar rb1 = fast::sqrt(sb1);
        Scalar rb3 = fast::sqrt(sb3);

        Scalar c0 = (vb1.x*vb3.x + vb1.y*vb3.y + vb1.z*vb3.z) * rb1*rb3;

        // 1st and 2nd angle

        Scalar b1mag2 = vb1.x*vb1.x + vb1.y*vb1.y + vb1.z*vb1.z;
        Scalar b1mag = fast::sqrt(b1mag2);
        Scalar b2mag2 = vb2.x*vb2.x + vb2.y*vb2.y + vb2.z*vb2.z;
        Scalar b2mag = fast::sqrt(b2mag2);
        Scalar b3mag2 = vb3.x*vb3.x + vb3.y*vb3.y + vb3.z*vb3.z;
        Scalar b3mag = fast::sqrt(b3mag2);

        Scalar ctmp = vb1.x*vb2.x + vb1.y*vb2.y + vb1.z*vb2.z;
        Scalar r12c1 = 1.0 / (b1mag*b2mag);
        Scalar c1mag = ctmp * r12c1;

        ctmp = vb2m.x*vb3.x + vb2m.y*vb3.y + vb2m.z*vb3.z;
        Scalar r12c2 = 1.0 / (b2mag*b3mag);
        Scalar c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        Scalar sin2 = 1.0 - c1mag*c1mag;
        if (sin2 < 0.0) sin2 = 0.0;
        Scalar sc1 = fast::sqrt(sin2);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = 1.0/sc1;

        sin2 = 1.0 - c2mag*c2mag;
        if (sin2 < 0.0) sin2 = 0.0;
        Scalar sc2 = fast::sqrt(sin2);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = 1.0/sc2;

        Scalar s1 = sc1 * sc1;
        Scalar s2 = sc2 * sc2;
        Scalar s12 = sc1 * sc2;
        Scalar c = (c0 + c1mag*c2mag) * s12;

        Scalar cx = vb1.y*vb2.z - vb1.z*vb2.y;
        Scalar cy = vb1.z*vb2.x - vb1.x*vb2.z;
        Scalar cz = vb1.x*vb2.y - vb1.y*vb2.x;
        Scalar cmag = fast::sqrt(cx*cx + cy*cy + cz*cz);
        Scalar dx = (cx*vb3.x + cy*vb3.y + cz*vb3.z)/cmag/b3mag;

        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        // force & energy
        // p = sum (i=1,4) k_i * (1 + (-1)**(i+1)*cos(i*phi) )
        // pd = dp/dc

        Scalar phi = acos(c);
        if (dx < 0.0) phi *= -1.0;
        Scalar si = sin(phi);
        if (fabs(si) < SMALLER) si = SMALLER;
        Scalar siinv = 1.0/si;
        
        // get values for k1/2 through k4/2 (MEM TRANSFER: 16 bytes)
        // ----- The 1/2 factor is already stored in the parameters --------
        Scalar4 params = texFetchScalar4(d_params, dihedral_params_tex, cur_dihedral_type);
        Scalar k1 = params.x;
        Scalar k2 = params.y;
        Scalar k3 = params.z;
        Scalar k4 = params.w;
        

        // the potential energy of the dihedral
        Scalar p = k1*(1.0 + c) + k2*(1.0 - cos(2.0*phi)) + k3*(1.0 + cos(3.0*phi)) + k4*(1.0 - cos(4.0*phi));
        Scalar pd = k1 - 2.0*k2*sin(2.0*phi)*siinv + 3.0*k3*sin(3.0*phi)*siinv - 4.0*k4*sin(4.0*phi)*siinv;

        // compute 1/4 of energy for each atom
        Scalar dihedral_eng = 0.25*p;


        // compute forces

        Scalar a = pd;
        c = c * a;
        s12 = s12 * a;
        Scalar a11 = c*sb1*s1;
        Scalar a22 = -sb2 * (2.0*c0*s12 - c*(s1+s2));
        Scalar a33 = c*sb3*s2;
        Scalar a12 = -r12c1 * (c1mag*c*s1 + c2mag*s12);
        Scalar a13 = -rb1*rb3*s12;
        Scalar a23 = r12c2 * (c2mag*c*s2 + c1mag*s12);

        Scalar3 ss2 = a12*vb1 + a22*vb2 + a23*vb3;

        Scalar3 f1 = a11*vb1 + a12*vb2 + a13*vb3;
        Scalar3 f2 = -ss2 - f1;
        Scalar3 f4 = a13*vb1 + a23*vb2 + a33*vb3;
        Scalar3 f3 = ss2 - f4;
        
        // Compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        
        Scalar dihedral_virial[6];
        dihedral_virial[0] = 0.25*(vb1.x*f1.x + vb2.x*f3.x + (vb3.x+vb2.x)*f4.x);
        dihedral_virial[1] = 0.25*(vb1.y*f1.x + vb2.y*f3.x + (vb3.y+vb2.y)*f4.x);
        dihedral_virial[2] = 0.25*(vb1.z*f1.x + vb2.z*f3.x + (vb3.z+vb2.z)*f4.x);
        dihedral_virial[3] = 0.25*(vb1.y*f1.y + vb2.y*f3.y + (vb3.y+vb2.y)*f4.y);
        dihedral_virial[4] = 0.25*(vb1.z*f1.y + vb2.z*f3.y + (vb3.z+vb2.z)*f4.y);
        dihedral_virial[5] = 0.25*(vb1.z*f1.z + vb2.z*f3.z + (vb3.z+vb2.z)*f4.z);

        if (cur_dihedral_abcd == 0)
            {
            force_idx.x += f1.x;
            force_idx.y += f1.y;
            force_idx.z += f1.z;
            }
        if (cur_dihedral_abcd == 1)
            {
            force_idx.x += f2.x;
            force_idx.y += f2.y;
            force_idx.z += f2.z;
            }
        if (cur_dihedral_abcd == 2)
            {
            force_idx.x += f3.x;
            force_idx.y += f3.y;
            force_idx.z += f3.z;
            }
        if (cur_dihedral_abcd == 3)
            {
            force_idx.x += f4.x;
            force_idx.y += f4.y;
            force_idx.z += f4.z;
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
    \param d_params Array of OPLS parameters k1/2, k2/2, k3/2, and k4/2
    \param n_dihedral_types Number of dihedral types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar4 element per dihedral type. The x component contains K the spring constant
    and the y component contains sign, and the z component the multiplicity.
*/
cudaError_t gpu_compute_opls_dihedral_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const BoxDim& box,
                                                 const group_storage<4> *tlist,
                                                 const unsigned int *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list,
                                                 Scalar4 *d_params,
                                                 unsigned int n_dihedral_types,
                                                 int block_size,
                                                 const unsigned int compute_capability)
    {
    assert(d_params);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_opls_dihedral_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the texture on pre sm35 devices
    if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, dihedral_params_tex, d_params, sizeof(Scalar4) * n_dihedral_types);
        if (error != cudaSuccess)
            return error;
        }

    // run the kernel
    gpu_compute_opls_dihedral_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, d_params, box, tlist, dihedral_ABCD, pitch, n_dihedrals_list);

    return cudaSuccess;
    }

#undef SMALL
#undef SMALLER