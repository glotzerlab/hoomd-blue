// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AlchemyData.h
    \brief Contains declarations for AlchemyData.
 */

#pragma once

#include "hoomd/ExecutionConfiguration.h"
#include <algorithm>
#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <string>

#include <pybind11/stl_bind.h>

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/HOOMDMath.h"

namespace hoomd
    {

namespace md
    {

struct AlchemicalParticle
    {
    AlchemicalParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : value(Scalar(1.0)), m_attached(true), m_exec_conf(exec_conf) {};

    void notifyDetach()
        {
        m_attached = false;
        }

    Scalar value; //!< Alpha space dimensionless position of the particle
    uint64_t m_nextTimestep;

    protected:
    bool m_attached;
    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf;                 //!< Stored shared ptr to the execution configuration
    std::shared_ptr<Compute> m_base; //!< the associated Alchemical Compute
    };

struct AlchemicalMDParticle : AlchemicalParticle
    {
    AlchemicalMDParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf)

        : AlchemicalParticle(exec_conf) {};

    void inline zeroForces()
        {
        ArrayHandle<Scalar> h_forces(m_alchemical_derivatives,
                                     access_location::host,
                                     access_mode::overwrite);
        memset((void*)h_forces.data, 0, sizeof(Scalar) * m_alchemical_derivatives.getNumElements());
        }

    void resizeForces(unsigned int N)
        {
        GlobalArray<Scalar> new_forces(N, m_exec_conf);
        m_alchemical_derivatives.swap(new_forces);
        }

    void zeroNetForce(uint64_t timestep)
        {
        zeroForces();
        m_timestep_net_force.first = timestep;
        }

    void setNetForce()
        {
        Scalar netForce(0.0);
        ArrayHandle<Scalar> h_forces(m_alchemical_derivatives,
                                     access_location::host,
                                     access_mode::read);
        for (unsigned int i = 0; i < m_alchemical_derivatives.getNumElements(); i++)
            netForce += h_forces.data[i];
        netForce /= Scalar(m_alchemical_derivatives.getNumElements());
        m_timestep_net_force.second = netForce;
        }

    Scalar getNetForce(uint64_t timestep)
        {
        assert(m_timestep_net_force.first == timestep);
        return m_timestep_net_force.second;
        }

    Scalar getNetForce()
        {
        return m_timestep_net_force.second;
        }

    void setMass(Scalar new_mass)
        {
        mass.x = new_mass;
        mass.y = Scalar(1.) / new_mass;
        }

    Scalar getMass()
        {
        return mass.x;
        }

    Scalar getValue()
        {
        return value;
        }

    pybind11::array_t<Scalar> getDAlphas()
        {
        ArrayHandle<Scalar> h_forces(m_alchemical_derivatives,
                                     access_location::host,
                                     access_mode::read);
        return pybind11::array(m_alchemical_derivatives.getNumElements(), h_forces.data);
        }

    Scalar momentum = 0.; // the momentum of the particle
    Scalar2 mass
        = make_scalar2(1.0,
                       1.0); // mass (x) and it's inverse (y) (don't have to recompute constantly)
    Scalar mu = 0.;          //!< the alchemical potential of the particle
    GlobalArray<Scalar> m_alchemical_derivatives; //!< Per particle alchemical forces
    protected:
    // the timestep the net force was computed and the netforce
    std::pair<uint64_t, Scalar> m_timestep_net_force;
    };

struct AlchemicalPairParticle : AlchemicalMDParticle
    {
    AlchemicalPairParticle(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                           int3 type_pair_param)
        : AlchemicalMDParticle(exec_conf), m_type_pair_param(type_pair_param) {};
    int3 m_type_pair_param;
    };

struct AlchemicalNormalizedPairParticle : AlchemicalPairParticle
    {
    using AlchemicalPairParticle::AlchemicalPairParticle;

    Scalar alchemical_derivative_normalization_value = 0.0;

    void NormalizeNetForce(Scalar norm_value, Scalar energy_value)
        {
        m_timestep_net_force.second *= norm_value;
        energy_value /= Scalar(m_alchemical_derivatives.getNumElements());
        m_timestep_net_force.second += energy_value * alchemical_derivative_normalization_value;
        }
    };
    } // namespace md
    } // namespace hoomd
