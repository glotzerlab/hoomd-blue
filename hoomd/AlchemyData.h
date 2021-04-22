// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

/*! \file AlchemyData.h
    \brief Contains declarations for AlchemyData.
 */

#include "hoomd/HOOMDMath.h"
#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ALCHEMYDATA_H__
#define __ALCHEMYDATA_H__

#include <string>
#include <memory>

#include "HOOMDMPI.h"
class AlchemicalParticle:
    {
    public:
    protected:
        Scalar m_value; //!< Alpha space dimensionless position of the particle
        // TODO: decide if velocity or momentum would typically be better for numerical stability
        Scalar3 m_kineticValues; //!< x=mass, y=velocity/momentum, z=netForce
        std::shared_ptr<ForceCompute> //!< the associated Alchemical Force Compute
    }



#endif
