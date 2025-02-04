// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef ENABLE_HIP
#include "BoxDim.h"
#include "GPUPartition.cuh"

#include "hoomd/CachedAllocator.h"

/*! \file ParticleData.cuh
    \brief Declares GPU kernel code and data structure functions used by ParticleData
*/

#ifdef __HIPCC__
//! Sentinel value in \a body to signify that this particle does not belong to a body
const unsigned int NO_BODY = 0xffffffff;

//! Unsigned value equivalent to a sign flip in a signed int. All larger values of the \a body flag
//! indicate a floppy body (forces between are ignored, but they are integrated independently).
const unsigned int MIN_FLOPPY = 0x80000000;

//! Sentinel value in \a r_tag to signify that this particle is not currently present on the local
//! processor
const unsigned int NOT_LOCAL = 0xffffffff;
#endif

namespace hoomd
    {
namespace detail
    {
#ifdef __HIPCC__
//! Compact particle data storage
struct pdata_element
    {
    Scalar4 pos;          //!< Position
    Scalar4 vel;          //!< Velocity
    Scalar3 accel;        //!< Acceleration
    Scalar charge;        //!< Charge
    Scalar diameter;      //!< Diameter
    int3 image;           //!< Image
    unsigned int body;    //!< Body id
    Scalar4 orientation;  //!< Orientation
    Scalar4 angmom;       //!< Angular momentum
    Scalar3 inertia;      //!< Moments of inertia
    unsigned int tag;     //!< global tag
    Scalar4 net_force;    //!< net force
    Scalar4 net_torque;   //!< net torque
    Scalar net_virial[6]; //!< net virial
    };
#else
//! Forward declaration
struct pdata_element;
#endif

    } // end namespace detail

namespace kernel
    {
//! Pack particle data into output buffer and remove marked particles
unsigned int gpu_pdata_remove(const unsigned int N,
                              const Scalar4* d_pos,
                              const Scalar4* d_vel,
                              const Scalar3* d_accel,
                              const Scalar* d_charge,
                              const Scalar* d_diameter,
                              const int3* d_image,
                              const unsigned int* d_body,
                              const Scalar4* d_orientation,
                              const Scalar4* d_angmom,
                              const Scalar3* d_inertia,
                              const Scalar4* d_net_force,
                              const Scalar4* d_net_torque,
                              const Scalar* d_net_virial,
                              unsigned int net_virial_pitch,
                              const unsigned int* d_tag,
                              unsigned int* d_rtag,
                              Scalar4* d_pos_alt,
                              Scalar4* d_vel_alt,
                              Scalar3* d_accel_alt,
                              Scalar* d_charge_alt,
                              Scalar* d_diameter_alt,
                              int3* d_image_alt,
                              unsigned int* d_body_alt,
                              Scalar4* d_orientation_alt,
                              Scalar4* d_angmom_alt,
                              Scalar3* d_inertia_alt,
                              Scalar4* d_net_force_alt,
                              Scalar4* d_net_torque_alt,
                              Scalar* d_net_virial_alt,
                              unsigned int* d_tag_alt,
                              detail::pdata_element* d_out,
                              unsigned int* d_comm_flags,
                              unsigned int* d_comm_flags_out,
                              unsigned int max_n_out,
                              unsigned int* d_tmp,
                              CachedAllocator& alloc,
                              GPUPartition& gpu_partition);

//! Update particle data with new particles
void gpu_pdata_add_particles(const unsigned int old_nparticles,
                             const unsigned int num_add_ptls,
                             Scalar4* d_pos,
                             Scalar4* d_vel,
                             Scalar3* d_accel,
                             Scalar* d_charge,
                             Scalar* d_diameter,
                             int3* d_image,
                             unsigned int* d_body,
                             Scalar4* d_orientation,
                             Scalar4* d_angmom,
                             Scalar3* d_inertia,
                             Scalar4* d_net_force,
                             Scalar4* d_net_torque,
                             Scalar* d_net_virial,
                             unsigned int net_virial_pitch,
                             unsigned int* d_tag,
                             unsigned int* d_rtag,
                             const detail::pdata_element* d_in,
                             unsigned int* d_comm_flags);
    } // end namespace kernel

    } // end namespace hoomd

#endif
