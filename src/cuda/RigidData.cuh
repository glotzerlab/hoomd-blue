/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#ifndef _RIGIDDATA_CUH_
#define _RIGIDDATA_CUH_

#include <cuda_runtime.h>

/*! \file RigidData.cuh
    \brief Declares GPU kernel code and data structure functions used by RigidData
*/

//! Structure of arrays of the rigid data as it resides on the GPU
/*! Stores pointers to the particles positions, velocities, acceleartions, and particle tags.
    Particle type information is most likely needed along with the position, so the type
    is encoded in the 4th float in the position float4 as an integer. Device code
    can decode this type data with __float_as_int();

    All the pointers in this structure are allocated on the device.

    This structure is about to be rewritten. Consider it being documented as poorly documented
    for now.

    \ingroup gpu_data_structs
*/
struct gpu_rigid_data_arrays
    {
    unsigned int n_bodies;  //!< Number of rigid bodies in the arrays
    unsigned int nmax;      //!< Maximum number of particles in a rigid body
    unsigned int local_beg; //!< Index of the first body local to this GPU
    unsigned int local_num; //!< Number of particles local to this GPU
    
    float  *body_mass;      //!< Body mass
    float4 *moment_inertia; //!< Body principle moments in \c x, \c y, \c z, nothing in \c w
    float4 *com;            //!< Body position in \c x,\c y,\c z, particle type as an int in \c w
    float4 *vel;            //!< Body velocity in \c x, \c y, \c z, nothing in \c w
    float4 *angvel;         //!< Angular velocity in \c x, \c y, \c z, nothing in \c w
    float4 *angmom;         //!< Angular momentum in \c x, \c y, \c z, nothing in \c w
    float4 *orientation;    //!< Quaternion in \c x, \c y, \c z, nothing in \c w
    float4 *ex_space;       //!< Body frame x axis in the world space in \c x, \c y, \c z, nothing in \c w
    float4 *ey_space;       //!< Body frame y axis in the world space in \c x, \c y, \c z, nothing in \c w
    float4 *ez_space;       //!< Body frame z axis in the world space in \c x, \c y, \c z, nothing in \c w
    int    *body_imagex;    //!< Body box image location in \c x.
    int    *body_imagey;    //!< Body box image location in \c y.
    int    *body_imagez;    //!< Body box image location in \c z.
    float4 *force;          //!< Body force in \c x, \c y, \c z, nothing in \c w
    float4 *torque;         //!< Body torque in \c x, \c y, \c z, nothing in \c w
    
    float4 *particle_pos;           //!< Particle relative position to the body frame
    unsigned int *particle_indices; //!< Particle indices: actual particle index in the particle data arrays
    unsigned int *particle_tags;    //!< Particle tags
    };

struct gpu_nvt_rigid_data_arrays
    {
    unsigned int n_bodies;
    
    float  *q_t;
    float  *q_r;
    float  *eta_t;
    float  *eta_r;
    float  *eta_dot_t;
    float  *eta_dot_r;
    float  *f_eta_t;
    float  *f_eta_r;
    float  *w;
    float  *wdti1;
    float  *wdti2;
    float  *wdti4;
    float4 *conjqm;
    
    float   *partial_ke;
    float   *ke;
    };

#endif

