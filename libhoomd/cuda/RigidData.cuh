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


// Maintainer: ndtrung

#ifndef _RIGIDDATA_CUH_
#define _RIGIDDATA_CUH_

#include <cuda_runtime.h>
#include "QuaternionMath.h"
#include "ParticleData.cuh"

/*! \file RigidData.cuh
    \brief Declares GPU kernel code and data structure functions used by RigidData
*/

//! Arrays of the rigid data as it resides on the GPU
/*! 
    All the pointers in this structure are allocated on the device.

   \ingroup gpu_data_structs
*/
struct gpu_rigid_data_arrays
    {
    unsigned int n_bodies;  //!< Number of rigid bodies in the rigid body arrays
    unsigned int n_group_bodies;   //!< Number of rigid bodies in the body group 
    unsigned int nmax;      //!< Maximum number of particles in a rigid body
    unsigned int local_beg; //!< Index of the first body local to this GPU
    unsigned int local_num; //!< Number of particles local to this GPU
    
    unsigned int *body_indices; //!< Body indices
    Scalar  *body_mass;      //!< Body mass
    Scalar4 *moment_inertia; //!< Body principle moments in \c x, \c y, \c z, nothing in \c w
    Scalar4 *com;            //!< Body position in \c x,\c y,\c z, particle type as an int in \c w
    Scalar4 *vel;            //!< Body velocity in \c x, \c y, \c z, nothing in \c w
    Scalar4 *angvel;         //!< Angular velocity in \c x, \c y, \c z, nothing in \c w
    Scalar4 *angmom;         //!< Angular momentum in \c x, \c y, \c z, nothing in \c w
    Scalar4 *orientation;    //!< Quaternion in \c x, \c y, \c z, nothing in \c w
    int3   *body_image;     //!< Body box image location
    Scalar4 *force;          //!< Body force in \c x, \c y, \c z, nothing in \c w
    Scalar4 *torque;         //!< Body torque in \c x, \c y, \c z, nothing in \c w
    Scalar *virial;          //!< Virial contribution from the first integration part
    Scalar4 *conjqm;         //!< Conjugate quaternion momentum 
    unsigned int *particle_offset; //!< Per particle array listing the index offset in its body
    
    Scalar4 *particle_orientation;   //!< Particle orientation relative to the body frame
    Scalar4 *particle_pos;           //!< Particle relative position to the body frame
    Scalar4 *particle_oldpos;        //!< Particle position from the previous step
    Scalar4 *particle_oldvel;        //!< Particle velocity from the previous step
    unsigned int *particle_indices; //!< Particle indices: actual particle index in the particle data arrays
    unsigned int *particle_tags;    //!< Particle tags   
    };


//! sets RV on the GPU for rigid bodies
cudaError_t gpu_rigid_setRV(Scalar4 *d_pos,
                            Scalar4 *d_vel,
                            int3 *d_image,
                            unsigned int *d_body,
                                   const gpu_rigid_data_arrays& rigid_data,
                                   Scalar4 *d_pdata_orientation,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const BoxDim& box, 
                                   bool set_x);

//! Computes the virial correction from the rigid body constraints
cudaError_t gpu_compute_virial_correction_end(Scalar *d_net_virial,
                                              const unsigned int virial_pitch,
                                              const Scalar4 *d_net_force,
                                              const Scalar4 *d_oldpos,
                                              const Scalar4 *d_oldvel,
                                              const Scalar4 *d_vel,
                                              const unsigned int *d_body,
                                              Scalar deltaT,
                                              unsigned int N);


#endif

