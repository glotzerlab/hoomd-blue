// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace hpmc {

namespace gpu {

//! Wraps arguments to kernel::hpmc_insert_depletants_phase(n)
/*! \ingroup hpmc_data_structs */
struct hpmc_auxilliary_args_t
    {
    //! Construct a hpmc_auxilliary_args_t
    hpmc_auxilliary_args_t(const unsigned int *_d_tag,
                           const Scalar4 *_d_vel,
                           const Scalar4 *_d_trial_vel,
                           const unsigned int _ntrial,
                           const unsigned int *_d_n_depletants_ntrial,
                           int *_d_deltaF_int,
                           const hipStream_t *_streams_phase1,
                           const hipStream_t *_streams_phase2)
                : d_tag(_d_tag),
                  d_vel(_d_vel),
                  d_trial_vel(_d_trial_vel),
                  ntrial(_ntrial),
                  d_n_depletants_ntrial(_d_n_depletants_ntrial),
                  d_deltaF_int(_d_deltaF_int),
                  streams_phase1(_streams_phase1),
                  streams_phase2(_streams_phase2)
        { };

    const unsigned int *d_tag;          //!< Particle tags
    const Scalar4 *d_vel;               //!< Particle velocities (.x component is the auxilliary variable)
    const Scalar4 *d_trial_vel;         //!< Particle velocities after trial move (.x component is the auxilliary variable)
    const unsigned int ntrial;          //!< Number of trial insertions per depletant
    const unsigned int *d_n_depletants_ntrial;     //!< Number of depletants per particle, depletant type pair and trial insertion
    int *d_deltaF_int;                  //!< Free energy difference rescaled to integer units
    const hipStream_t *streams_phase1;             //!< Stream for this depletant type, phase1 kernel
    const hipStream_t *streams_phase2;             //!< Stream for this depletant type, phase2 kernel
    };

} // end namespace gpu

} // end namespace hpmc
