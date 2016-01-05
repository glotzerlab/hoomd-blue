/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

// Maintainer: joaander

#include "ActiveForceComputeGPU.cuh"
#include "saruprngCUDA.h"
#include "EvaluatorConstraintEllipsoid.h"

#include <assert.h>

/*! \file ActiveForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveForceComputeGPU.
*/

//! Kernel for adjusting active force vectors to align parallel to an ellipsoid surface constraint on the GPU
/*! \param N number of particles
    \param d_rtag particle tag
    \param d_pos particle positions on device
    \param d_actVec particle active force unit vector
    \param d_actMag particle active force vector magnitude
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
*/
extern "C" __global__
void gpu_compute_active_force_set_constraints_kernel(const unsigned int N,
                                                   const unsigned int *d_rtag,
                                                   const Scalar4 *d_pos,
                                                   Scalar3 *d_actVec,
                                                   const Scalar *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
    
/////////////////////////////////////
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
/////////////////////////////////////

    
    EvaluatorConstraintEllipsoid Ellipsoid(P, rx, ry, rz);
    unsigned int idx = d_rtag[i]; // recover original tag for particle indexing
    Scalar3 current_pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);
                
    Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
    vec3<Scalar> norm;
    norm = vec3<Scalar>(norm_scalar3);
    Scalar dot_prod = d_actVec[i].x * norm.x + d_actVec[i].y * norm.y + d_actVec[i].z * norm.z;

    d_actVec[i].x -= norm.x * dot_prod;
    d_actVec[i].y -= norm.y * dot_prod;
    d_actVec[i].z -= norm.z * dot_prod;

    Scalar new_norm = sqrt(d_actVec[i].x * d_actVec[i].x
                        + d_actVec[i].y * d_actVec[i].y
                        + d_actVec[i].z * d_actVec[i].z);

    d_actVec[i].x /= new_norm;
    d_actVec[i].y /= new_norm;
    d_actVec[i].z /= new_norm;
}

//! Kernel for applying rotational diffusion to active force vectors on the GPU
/*! \param N number of particles
    \param d_rtag particle tag
    \param d_pos particle positions on device
    \param d_actVec particle active force unit vector
    \param d_actMag particle active force vector magnitude
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param is2D check if simulation is 2D or 3D
    \param rotationDiff particle rotational diffusion constant
    \param deltaT step size from the Integrator
    \param timestep current timestep
    \param seed seed for random number generator
*/
__global__ void gpu_compute_active_force_rotational_diffusion_kernel(const unsigned int N,
                                                   const unsigned int *d_rtag,
                                                   const Scalar4 *d_pos,
                                                   Scalar3 *d_actVec,
                                                   const Scalar *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   bool is2D,
                                                   const Scalar rotationDiff,
                                                   const Scalar deltaT,
                                                   const unsigned int timestep,
                                                   const int seed)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
    
/////////////////////////////////////
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
/////////////////////////////////////
    
    
    if (is2D) // 2D
    {
        //USE VECTOR MATH TO SIMPLIFY THINGS? CHECK UNITS AND MAGNITUDES, ALL CHECK OUT?
        SaruGPU saru(i, timestep, seed);
        Scalar delta_theta; // rotational diffusion angle
        delta_theta = deltaT * rotationDiff * gaussian_rng(saru, 1.0);
        Scalar theta; // angle on plane defining orientation of active force vector
        theta = atan2(d_actVec[i].y, d_actVec[i].x);
        theta += delta_theta;
        d_actVec[i].x = cos(theta);
        d_actVec[i].y = sin(theta);

    } else // 3D: Following Stenhammar, Soft Matter, 2014
    {
        if (rx == 0) // if no constraint
        {
            //USE VECTOR MATH TO SIMPLIFY THINGS? CHECK UNITS AND MAGNITUDES OF DIFFUSION CONSTANT, ALL CHECK OUT?
            SaruGPU saru(i, timestep, seed);
            Scalar u = saru.d(0, 1.0); // generates an even distribution of random unit vectors in 3D
            Scalar v = saru.d(0, 1.0);
            Scalar theta = 2.0 * M_PI * u;
            Scalar phi = acos(2.0 * v - 1.0);
            vec3<Scalar> rand_vec;
            rand_vec.x = sin(phi) * cos(theta);
            rand_vec.y = sin(phi) * sin(theta);
            rand_vec.z = cos(phi);
            Scalar diffusion_mag = deltaT * rotationDiff * gaussian_rng(saru, 1.0);
            vec3<Scalar> delta_vec;
            delta_vec.x = d_actVec[i].y * rand_vec.z - d_actVec[i].z * rand_vec.y;
            delta_vec.y = d_actVec[i].z * rand_vec.x - d_actVec[i].x * rand_vec.z;
            delta_vec.z = d_actVec[i].x * rand_vec.y - d_actVec[i].y * rand_vec.x;
            d_actVec[i].x += delta_vec.x * diffusion_mag;
            d_actVec[i].y += delta_vec.y * diffusion_mag;
            d_actVec[i].z += delta_vec.z * diffusion_mag;
            Scalar new_mag = sqrt(d_actVec[i].x * d_actVec[i].x + d_actVec[i].y * d_actVec[i].y + d_actVec[i].z * d_actVec[i].z);
            d_actVec[i].x /= new_mag;
            d_actVec[i].y /= new_mag;
            d_actVec[i].z /= new_mag;

        } else // if constraint
        {
            EvaluatorConstraintEllipsoid Ellipsoid(P, rx, ry, rz);

            SaruGPU saru(i, timestep, seed);
            unsigned int idx = d_rtag[i]; // recover original tag for particle indexing
            Scalar3 current_pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);
            Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.

            vec3<Scalar> norm;
            norm = vec3<Scalar> (norm_scalar3);

            vec3<Scalar> current_vec;
            current_vec.x = d_actVec[i].x;
            current_vec.y = d_actVec[i].y;
            current_vec.z = d_actVec[i].z;
            vec3<Scalar> aux_vec = cross(current_vec, norm); // aux vec for defining direction that active force vector rotates towards.

            Scalar delta_theta; // rotational diffusion angle
            delta_theta = deltaT * rotationDiff * gaussian_rng(saru, 1.0);

            d_actVec[i].x = cos(delta_theta) * current_vec.x + sin(delta_theta) * aux_vec.x;
            d_actVec[i].y = cos(delta_theta) * current_vec.y + sin(delta_theta) * aux_vec.y;
            d_actVec[i].z = cos(delta_theta) * current_vec.z + sin(delta_theta) * aux_vec.z;
        }
    }
}

//! Kernel for setting active force vectors on the GPU
/*! \param N number of particles
    \param d_rtag particle tag
    \param d_force particle force on device
    \param d_orientation particle orientation on device
    \param d_actVec particle active force unit vector
    \param d_actMag particle active force vector magnitude
    \param P position of the ellipsoid constraint
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param orientationLink check if particle orientation is linked to active force vector
*/
__global__ void gpu_compute_active_force_set_forces_kernel(const unsigned int N,
                                                   const unsigned int *d_rtag, 
                                                   Scalar4 *d_force,
                                                   const Scalar4 *d_orientation,
                                                   const Scalar3 *d_actVec,
                                                   const Scalar *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   bool orientationLink)
{
    //FILL ME IN, FINISH ACTIVE FORCE GPU CODE
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

//////////////////////////////////
    if (i >= N)
        return;

    // why use groups?
    // unsigned int i = d_group_members[id];
    unsigned int idx = d_rtag[i];
//////////////////////////////////
    
    Scalar3 f;
    // unsigned int idx = h_rtag[i]; // recover original tag for particle indexing
    // rotate force according to particle orientation only if orientation is linked to active force vector and there are rigid bodies
    if (orientationLink)
    {
        vec3<Scalar> fi;
        f = make_scalar3(d_actMag[i] * d_actVec[i].x, d_actMag[i] * d_actVec[i].y, d_actMag[i] * d_actVec[i].z);
        quat<Scalar> quati(d_orientation[idx]);
        fi = rotate(quati, vec3<Scalar>(f));
        d_force[idx].x = fi.x;
        d_force[idx].y = fi.y;
        d_force[idx].z = fi.z;
    } else // no orientation link
    {
        f = make_scalar3(d_actMag[i] * d_actVec[i].x, d_actMag[i] * d_actVec[i].y, d_actMag[i] * d_actVec[i].z);
        d_force[idx].x = f.x;
        d_force[idx].y = f.y;
        d_force[idx].z = f.z;
    }
}



cudaError_t gpu_compute_active_force_set_constraints(const unsigned int N,
                                                   const unsigned int *d_rtag,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_force,
                                                   Scalar3 *d_actVec,
                                                   const Scalar *d_actMag,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size)
{
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_set_constraints_kernel<<< grid, threads>>>(N,
                                                                    d_rtag,
                                                                    d_pos,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);

    return cudaSuccess;
}

cudaError_t gpu_compute_active_force_rotational_diffusion(const unsigned int N,
                                                       const unsigned int *d_rtag,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_force,
                                                       Scalar3 *d_actVec,
                                                       const Scalar *d_actMag,
                                                       const Scalar3& P,
                                                       Scalar rx,
                                                       Scalar ry,
                                                       Scalar rz,
                                                       bool is2D,
                                                       const Scalar rotationDiff,
                                                       const Scalar deltaT,
                                                       const unsigned int timestep,
                                                       const int seed,
                                                       unsigned int block_size)
{
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_rotational_diffusion_kernel<<< grid, threads>>>(N,
                                                                    d_rtag,
                                                                    d_pos,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz,
                                                                    is2D,
                                                                    rotationDiff,
                                                                    deltaT,
                                                                    timestep,
                                                                    seed);

    return cudaSuccess;
}

cudaError_t gpu_compute_active_force_set_forces(const unsigned int N,
                                           const unsigned int *d_rtag,
                                           Scalar4 *d_force,
                                           const Scalar4 *d_orientation,
                                           const Scalar3 *d_actVec,
                                           const Scalar *d_actMag,
                                           const Scalar3& P,
                                           Scalar rx,
                                           Scalar ry,
                                           Scalar rz,
                                           bool orientationLink,
                                           unsigned int block_size)
{
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    gpu_compute_active_force_set_forces_kernel<<< grid, threads>>>( N,
                                                                    d_rtag,
                                                                    d_force,
                                                                    d_orientation,
                                                                    d_actVec,
                                                                    d_actMag,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz,
                                                                    orientationLink);

    return cudaSuccess;
}










