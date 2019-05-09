// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "HarmonicImproperForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file HarmonicImproperForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic improper forces. Used by HarmonicImproperForceComputeGPU.
*/

//! Kernel for calculating harmonic improper forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial
    \param N number of particles
    \param d_pos Device memory of particle positions
    \param d_params Force field parameters
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Improper data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
*/
extern "C" __global__
void gpu_compute_harmonic_improper_forces_kernel(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const Scalar2 *d_params,
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
    int n_impropers = n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c-d quartet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a,pos_b,pos_c, pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial_idx[6];
    for (int i = 0; i < 6; i++)
        virial_idx[i] = Scalar(0.0);

    // loop over all impropers
    for (int improper_idx = 0; improper_idx < n_impropers; improper_idx++)
        {
        group_storage<4> cur_improper = tlist[pitch*improper_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch*improper_idx + idx];

        int cur_improper_x_idx = cur_improper.idx[0];
        int cur_improper_y_idx = cur_improper.idx[1];
        int cur_improper_z_idx = cur_improper.idx[2];
        int cur_improper_type = cur_improper.idx[3];
        int cur_improper_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_improper_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_improper_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 z_postype = d_pos[cur_improper_z_idx];
        Scalar3 z_pos = make_scalar3(z_postype.x, z_postype.y, z_postype.z);

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

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = pos_a - pos_b;
        Scalar3 dcb = pos_c - pos_b;
        Scalar3 ddc = pos_d - pos_c;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        // get the improper parameters (MEM TRANSFER: 12 bytes)
        Scalar2 params = __ldg(d_params + cur_improper_type);
        Scalar K = params.x;
        Scalar chi = params.y;

        Scalar r1 = rsqrtf(dot(dab, dab));
        Scalar r2 = rsqrtf(dot(dcb, dcb));
        Scalar r3 = rsqrtf(dot(ddc, ddc));

        Scalar ss1 = r1 * r1;
        Scalar ss2 = r2 * r2;
        Scalar ss3 = r3 * r3;

        // Cosine and Sin of the angle between the planes
        Scalar c0 = dot(dab, ddc) * r1 * r3;
        Scalar c1 = dot(dab, dcb) * r1 * r2;
        Scalar c2 = -dot(ddc, dcb) * r3 * r2;

        Scalar s1 = Scalar(1.0) - c1*c1;
        if (s1 < SMALL) s1 = SMALL;
        s1 = Scalar(1.0) / s1;

        Scalar s2 = Scalar(1.0) - c2*c2;
        if (s2 < SMALL) s2 = SMALL;
        s2 = Scalar(1.0) / s2;

        Scalar s12 = sqrtf(s1*s2);
        Scalar c = (c1*c2 + c0) * s12;

        if (c > Scalar(1.0)) c = Scalar(1.0);
        if (c < -Scalar(1.0)) c = -Scalar(1.0);

        Scalar s = sqrtf(Scalar(1.0) - c*c);
        if (s < SMALL) s = SMALL;

        Scalar domega = fast::acos(c) - chi;
        Scalar a = K * domega;

        // calculate the energy, 1/4th for each atom
        //Scalar improper_eng = 0.25*a*domega;
        Scalar improper_eng = Scalar(0.125)*a*domega;  // the .125 term is 1/2 * 1/4

        //a = -a * 2.0/s;
        a = -a /s; // the missing 2.0 factor is to ensure K/2 is factored in for the forces
        c = c * a;
        s12 = s12 * a;
        Scalar a11 = c*ss1*s1;
        Scalar a22 = -ss2 * (Scalar(2.0)*c0*s12 - c*(s1+s2));
        Scalar a33 = c*ss3*s2;

        Scalar a12 = -r1*r2*(c1*c*s1 + c2*s12);
        Scalar a13 = -r1*r3*s12;
        Scalar a23 = r2*r3*(c2*c*s2 + c1*s12);

        Scalar sx2  = a22*dcb.x + a23*ddc.x + a12*dab.x;
        Scalar sy2  = a22*dcb.y + a23*ddc.y + a12*dab.y;
        Scalar sz2  = a22*dcb.z + a23*ddc.z + a12*dab.z;

        // calculate the forces for each particle
        Scalar ffax = a12*dcb.x + a13*ddc.x + a11*dab.x;
        Scalar ffay = a12*dcb.y + a13*ddc.y + a11*dab.y;
        Scalar ffaz = a12*dcb.z + a13*ddc.z + a11*dab.z;

        Scalar ffbx = -sx2 - ffax;
        Scalar ffby = -sy2 - ffay;
        Scalar ffbz = -sz2 - ffaz;

        Scalar ffdx = a23*dcb.x + a33*ddc.x + a13*dab.x;
        Scalar ffdy = a23*dcb.y + a33*ddc.y + a13*dab.y;
        Scalar ffdz = a23*dcb.z + a33*ddc.z + a13*dab.z;

        Scalar ffcx = sx2 - ffdx;
        Scalar ffcy = sy2 - ffdy;
        Scalar ffcz = sz2 - ffdz;

        // and calculate the virial (upper triangular version)
        Scalar improper_virial[6];
        improper_virial[0] = Scalar(1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        improper_virial[1] = Scalar(1./4.)*(dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        improper_virial[2] = Scalar(1./4.)*(dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        improper_virial[3] = Scalar(1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        improper_virial[4] = Scalar(1./4.)*(dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        improper_virial[5] = Scalar(1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);

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
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
    \param d_params K, sign,multiplicity params packed as padded Scalar4 variables
    \param n_improper_types Number of improper types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar4 element per improper type. The x component contains K the spring constant
    and the y component contains sign, and the z component the multiplicity.
*/
cudaError_t gpu_compute_harmonic_improper_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const BoxDim& box,
                                                 const group_storage<4> *tlist,
                                                 const unsigned int *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list,
                                                 Scalar2 *d_params,
                                                 unsigned int n_improper_types,
                                                 int block_size)
    {
    assert(d_params);

    if (N == 0)
        return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_harmonic_improper_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    gpu_compute_harmonic_improper_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, d_params, box, tlist, dihedral_ABCD, pitch, n_dihedrals_list);

    return cudaSuccess;
    }
