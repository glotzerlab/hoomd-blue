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

// Maintainer: phillicl

#include "TableDihedralForceGPU.cuh"
#include "TextureTools.h"


#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// SMALL a relatively small number
#define SMALL 0.001f

/*! \file TableDihedralForceGPU.cu
    \brief Defines GPU kernel code for calculating the table dihedral forces. Used by TableDihedralForceComputeGPU.
*/


//! Texture for reading table values
scalar2_tex_t tables_tex;

/*!  This kernel is called to calculate the table dihedral forces on all triples this is defined or

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param device_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param dlist List of dihedrals stored on the GPU
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals stored on the GPU
    \param n_dihedral_type number of dihedral types
    \param d_tables Tables of the potential and force
    \param table_value index helper function
    \param delta_phi dihedral delta of the table

    See TableDihedralForceCompute for information on the memory layout.

    \b Details:
    * Table entries are read from tables_tex. Note that currently this is bound to a 1D memory region. Performance tests
      at a later date may result in this changing.
*/
__global__ void gpu_compute_table_dihedral_forces_kernel(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *device_pos,
                                     const BoxDim box,
                                     const group_storage<4> *dlist,
                                     const unsigned int *dihedral_ABCD,
                                     const unsigned int pitch,
                                     const unsigned int *n_dihedrals_list,
                                     const Scalar2 *d_tables,
                                     const Index2D table_value,
                                     const Scalar delta_phi)
    {


    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals =n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = device_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a,pos_b,pos_c, pos_d; // allocate space for the a,b,c, and d atom in the a-b-c-d set


    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);

    // initialize the virial tensor to 0
    Scalar virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = 0;

    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = dlist[pitch*dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch*dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];
        int cur_dihedral_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = device_pos[cur_dihedral_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = device_pos[cur_dihedral_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);
        // get the d-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 z_postype = device_pos[cur_dihedral_z_idx];
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

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = pos_a - pos_b;
        Scalar3 dcb = pos_c - pos_b;
        Scalar3 ddc = pos_d - pos_c;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        Scalar3 dcbm = -dcb;
        dcbm = box.minImage(dcbm);

        // c0 calculation
        Scalar sb1 = Scalar(1.0) / (dab.x*dab.x + dab.y*dab.y + dab.z*dab.z);
        Scalar sb2 = Scalar(1.0) / (dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z);
        Scalar sb3 = Scalar(1.0) / (ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z);

        Scalar rb1 = sqrt(sb1);
        Scalar rb3 = sqrt(sb3);

        Scalar c0 = (dab.x*ddc.x + dab.y*ddc.y + dab.z*ddc.z) * rb1*rb3;

        // 1st and 2nd angle

        Scalar b1mag2 = dab.x*dab.x + dab.y*dab.y + dab.z*dab.z;
        Scalar b1mag = sqrt(b1mag2);
        Scalar b2mag2 = dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z;
        Scalar b2mag = sqrt(b2mag2);
        Scalar b3mag2 = ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z;
        Scalar b3mag = sqrt(b3mag2);

        Scalar ctmp = dab.x*dcb.x + dab.y*dcb.y + dab.z*dcb.z;
        Scalar r12c1 = Scalar(1.0) / (b1mag*b2mag);
        Scalar c1mag = ctmp * r12c1;

        ctmp = dcbm.x*ddc.x + dcbm.y*ddc.y + dcbm.z*ddc.z;
        Scalar r12c2 = Scalar(1.0) / (b2mag*b3mag);
        Scalar c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        Scalar sin2 = Scalar(1.0) - c1mag*c1mag;
        if (sin2 < 0.0f) sin2 = 0.0f;
        Scalar sc1 = sqrtf(sin2);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = Scalar(1.0)/sc1;

        sin2 = Scalar(1.0) - c2mag*c2mag;
        if (sin2 < 0.0f) sin2 = 0.0f;
        Scalar sc2 = sqrtf(sin2);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = Scalar(1.0)/sc2;

        Scalar s1 = sc1 * sc1;
        Scalar s2 = sc2 * sc2;
        Scalar s12 = sc1 * sc2;
        Scalar c = (c0 + c1mag*c2mag) * s12;

        if (c > Scalar(1.0)) c = Scalar(1.0);
        if (c < -Scalar(1.0)) c = -Scalar(1.0);

        //phi
        Scalar phi = acosf(c);
        // precomputed term
        Scalar value_f = phi / delta_phi;

        // compute index into the table and read in values
        unsigned int value_i = floor(value_f);
        Scalar2 VT0 = texFetchScalar2(d_tables, tables_tex, table_value(value_i, cur_dihedral_type));
        Scalar2 VT1 = texFetchScalar2(d_tables, tables_tex, table_value(value_i+1, cur_dihedral_type));
        // unpack the data
        Scalar V0 = VT0.x;
        Scalar V1 = VT1.x;
        Scalar T0 = VT0.y;
        Scalar T1 = VT1.y;

        // compute the linear interpolation coefficient
        Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and T;
        Scalar V = V0 + f * (V1 - V0);
        Scalar T = T0 + f * (T1 - T0);


        Scalar a = T;
        c = c * a;
        s12 = s12 * a;
        Scalar a11 = c*sb1*s1;
        Scalar a22 = -sb2 * (Scalar(2.0)*c0*s12 - c*(s1+s2));
        Scalar a33 = c*sb3*s2;
        Scalar a12 = -r12c1*(c1mag*c*s1 + c2mag*s12);
        Scalar a13 = -rb1*rb3*s12;
        Scalar a23 = r12c2*(c2mag*c*s2 + c1mag*s12);

        Scalar sx2  = a12*dab.x + a22*dcb.x + a23*ddc.x;
        Scalar sy2  = a12*dab.y + a22*dcb.y + a23*ddc.y;
        Scalar sz2  = a12*dab.z + a22*dcb.z + a23*ddc.z;

        Scalar ffax = a11*dab.x + a12*dcb.x + a13*ddc.x;
        Scalar ffay = a11*dab.y + a12*dcb.y + a13*ddc.y;
        Scalar ffaz = a11*dab.z + a12*dcb.z + a13*ddc.z;

        Scalar ffbx = -sx2 - ffax;
        Scalar ffby = -sy2 - ffay;
        Scalar ffbz = -sz2 - ffaz;

        Scalar ffdx = a13*dab.x + a23*dcb.x + a33*ddc.x;
        Scalar ffdy = a13*dab.y + a23*dcb.y + a33*ddc.y;
        Scalar ffdz = a13*dab.z + a23*dcb.z + a33*ddc.z;

        Scalar ffcx = sx2 - ffdx;
        Scalar ffcy = sy2 - ffdy;
        Scalar ffcz = sz2 - ffdz;

        // Now, apply the force to each individual atom a,b,c,d
        // and accumlate the energy/virial

        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        Scalar dihedral_eng = V*Scalar(1.0/4.0);

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // symmetrized version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = Scalar(1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        dihedral_virial[1] = Scalar(1./8.)*(dab.x*ffay + dcb.x*ffcy + (ddc.x+dcb.x)*ffdy
                                     +dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        dihedral_virial[2] = Scalar(1./8.)*(dab.x*ffaz + dcb.x*ffcz + (ddc.x+dcb.x)*ffdz
                                     +dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        dihedral_virial[3] = Scalar(1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        dihedral_virial[4] = Scalar(1./8.)*(dab.y*ffaz + dcb.y*ffcz + (ddc.y+dcb.y)*ffdz
                                     +dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        dihedral_virial[5] = Scalar(1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);

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
    \param device_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param dlist List of dihedrals stored on the GPU
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals stored on the GPU
    \param n_dihedral_type number of dihedral types
    \param d_tables Tables of the potential and force
    \param table_width Number of points in each table
    \param table_value indexer helper
    \param block_size Block size at which to run the kernel
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \note This is just a kernel driver. See gpu_compute_table_dihedral_forces_kernel for full documentation.
*/
cudaError_t gpu_compute_table_dihedral_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *device_pos,
                                     const BoxDim &box,
                                     const group_storage<4> *dlist,
                                     const unsigned int *dihedral_ABCD,
                                     const unsigned int pitch,
                                     const unsigned int *n_dihedrals_list,
                                     const Scalar2 *d_tables,
                                     const unsigned int table_width,
                                     const Index2D &table_value,
                                     const unsigned int block_size,
                                     const unsigned int compute_capability)
    {
    assert(d_tables);
    assert(table_width > 1);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_compute_table_dihedral_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)run_block_size), 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the tables texture on pre sm35 devices
    if (compute_capability < 350)
        {
        tables_tex.normalized = false;
        tables_tex.filterMode = cudaFilterModePoint;
        cudaError_t error = cudaBindTexture(0, tables_tex, d_tables, sizeof(Scalar2) * table_value.getNumElements());
        if (error != cudaSuccess)
            return error;
        }

    Scalar delta_phi = M_PI/(table_width - 1.0f);

    gpu_compute_table_dihedral_forces_kernel<<< grid, threads>>>
            (d_force,
             d_virial,
             virial_pitch,
             N,
             device_pos,
             box,
             dlist,
             dihedral_ABCD,
             pitch,
             n_dihedrals_list,
             d_tables,
             table_value,
             delta_phi);

    return cudaSuccess;
    }

// vim:syntax=cpp
