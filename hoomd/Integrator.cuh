// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Integrator.cuh
    \brief Declares methods and data structures used by the Integrator class on the GPU
*/

#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "ParticleData.cuh"
#include "GPUPartition.cuh"

//! struct to pack up several force and virial arrays for addition
/*! To keep the argument count down to gpu_integrator_sum_accel, up to 6 force/virial array pairs are packed up in this
    struct for addition to the net force/virial in a single kernel call. If there is not a multiple of 5 forces to sum,
    set some of the pointers to NULL and they will be ignored.
*/
struct gpu_force_list
    {
    //! Initializes to NULL
    gpu_force_list()
        : f0(NULL), f1(NULL), f2(NULL), f3(NULL), f4(NULL), f5(NULL),
          t0(NULL), t1(NULL), t2(NULL), t3(NULL), t4(NULL), t5(NULL),
          v0(NULL), v1(NULL), v2(NULL), v3(NULL), v4(NULL), v5(NULL),
          vpitch0(0), vpitch1(0), vpitch2(0), vpitch3(0), vpitch4(0), vpitch5(0)
          {
          }

    Scalar4 *f0; //!< Pointer to force array 0
    Scalar4 *f1; //!< Pointer to force array 1
    Scalar4 *f2; //!< Pointer to force array 2
    Scalar4 *f3; //!< Pointer to force array 3
    Scalar4 *f4; //!< Pointer to force array 4
    Scalar4 *f5; //!< Pointer to force array 5

    Scalar4 *t0; //!< Pointer to torque array 0
    Scalar4 *t1; //!< Pointer to torque array 1
    Scalar4 *t2; //!< Pointer to torque array 2
    Scalar4 *t3; //!< Pointer to torque array 3
    Scalar4 *t4; //!< Pointer to torque array 4
    Scalar4 *t5; //!< Pointer to torque array 5

    Scalar *v0;  //!< Pointer to virial array 0
    Scalar *v1;  //!< Pointer to virial array 1
    Scalar *v2;  //!< Pointer to virial array 2
    Scalar *v3;  //!< Pointer to virial array 3
    Scalar *v4;  //!< Pointer to virial array 4
    Scalar *v5;  //!< Pointer to virial array 5

    unsigned int vpitch0; //!< Pitch of virial array 0
    unsigned int vpitch1; //!< Pitch of virial array 1
    unsigned int vpitch2; //!< Pitch of virial array 2
    unsigned int vpitch3; //!< Pitch of virial array 3
    unsigned int vpitch4; //!< Pitch of virial array 4
    unsigned int vpitch5; //!< Pitch of virial array 5
 };

//! Driver for gpu_integrator_sum_net_force_kernel()
cudaError_t gpu_integrator_sum_net_force(Scalar4 *d_net_force,
                                         Scalar *d_net_virial,
                                         const unsigned int virial_pitch,
                                         Scalar4 *d_net_torque,
                                         const gpu_force_list& force_list,
                                         unsigned int nparticles,
                                         bool clear,
                                         bool compute_virial,
                                         const GPUPartition& gpu_partition);

#endif
