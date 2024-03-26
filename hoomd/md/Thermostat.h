// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_THERMOSTAT_H
#define HOOMD_THERMOSTAT_H

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include <array>
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleGroup.h>
#include <hoomd/Variant.h>
#include <hoomd/md/ComputeThermo.h>

namespace hoomd::md
    {

/** Velocity rescaling thermostat.

    Define the base class for velocity-rescaling thermostats to be used with ConstantVolume and
    ConstantPressure integration methods.
*/
class Thermostat
    {
    public:
    /** Construct the thermostat.

        @param T Temperature set point over time.
        @param group Group of particles this thermostat is applied to.
        @param thermo Use to compute the thermodynamic properties of the group.
        @param sysdef Used to access the simulation seed and MPI communicator.
    */
    Thermostat(std::shared_ptr<Variant> T,
               std::shared_ptr<ParticleGroup> group,
               std::shared_ptr<ComputeThermo> thermo,
               std::shared_ptr<SystemDefinition> sysdef)
        : m_group(group), m_thermo(thermo), m_T(T), m_sysdef(sysdef)
        {
        }

    /// Destructor.
    virtual ~Thermostat() { }

    /** Get the rescaling factors to employ in the first half step of the integration.

        @param timestep Current simulation timestep.
        @param deltaT Simulation step size.
        @returns [translation rescale factor, rotation rescale factor]
    */
    virtual std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    /** Get the rescaling factors to employ in the second half step of the integration.

        @param timestep Current simulation timestep.
        @param deltaT Simulation step size.
        @returns [translation rescale factor, rotation rescale factor]
    */
    virtual std::array<Scalar, 2> getRescalingFactorsTwo(uint64_t timestep, Scalar deltaT)
        {
        return {Scalar(1.0), Scalar(1.0)};
        }

    /** Advance the thermostat degrees of freedom (if present) forward one timestep.

        @param timestep Current simulation timestep.
        @param deltaT Simulation step size.
        @param aniso Set to true when the integration method is applied to rotational degrees of
                     freedom.
    */
    virtual void advanceThermostat(uint64_t timestep, Scalar deltaT, bool aniso) { }

    /// Get the temperature variant.
    std::shared_ptr<Variant> getT()
        {
        return m_T;
        }

    /** Set the temperature variant.

        @param T New temperature variant.
    */
    void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    /** Evaluate the temperature variant at the given timestep.

        @param timestep The simulation timestep.
        @returns The value of the temperature variant at the given timestep.
    */
    Scalar getTimestepTemperature(uint64_t timestep)
        {
        return m_T->operator()(timestep);
        }

    protected:
    /// The particle group to thermostat.
    std::shared_ptr<ParticleGroup> m_group;

    /// Compute the thermodynamic properties of the group.
    std::shared_ptr<ComputeThermo> m_thermo;

    /// The temperature set point of the thermostat.
    std::shared_ptr<Variant> m_T;

    /// The system definition.
    std::shared_ptr<SystemDefinition> m_sysdef;
    };

/** Implement the MTTK thermostat.

    See: G. J. Martyna, D. J. Tobias, M. L. Klein 1994 <http://dx.doi.org/10.1063/1.467468> and
    J. Cao, G. J. Martyna 1996 <http://dx.doi.org/10.1063/1.470959>.
*/
class MTTKThermostat : public Thermostat
    {
    public:
    /** Construct the thermostat.

        @param T Temperature set point over time.
        @param group Group of particles this thermostat is applied to.
        @param thermo Use to compute the thermodynamic properties of the group.
        @param sysdef Used to access the simulation seed and MPI communicator.
        @param tau Thermostat time constant.
    */
    MTTKThermostat(std::shared_ptr<Variant> T,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermo> thermo,
                   std::shared_ptr<SystemDefinition> sysdef,
                   Scalar tau)
        : Thermostat(T, group, thermo, sysdef), m_tau(tau)
        {
        }

    /// Structure that describes the internal degrees of freedom of the MTTK thermostat.
    struct state
        {
        Scalar xi = 0;
        Scalar eta = 0;
        Scalar xi_rot = 0;
        Scalar eta_rot = 0;
        };

    std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT) override
        {
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * m_state.xi * deltaT);
        Scalar exp_thermo_fac_rot = exp(-m_state.xi_rot * deltaT / Scalar(2.0));
        return {exp_thermo_fac, exp_thermo_fac_rot};
        }

    std::array<Scalar, 2> getRescalingFactorsTwo(uint64_t timestep, Scalar deltaT) override
        {
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * m_state.xi * deltaT);
        Scalar exp_thermo_fac_rot = exp(-m_state.xi_rot * deltaT / Scalar(2.0));
        return {exp_thermo_fac, exp_thermo_fac_rot};
        }

    void advanceThermostat(uint64_t timestep, Scalar deltaT, bool aniso = true) override
        {
        // compute the current thermodynamic properties
        m_thermo->compute(timestep);

        Scalar curr_T_trans = m_thermo->getTranslationalTemperature();
        Scalar T = m_T->operator()(timestep);

        // update the state variables Xi and eta
        Scalar xi_prime
            = m_state.xi
              + Scalar(1.0 / 2.0) * deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
        m_state.xi
            = xi_prime
              + Scalar(1.0 / 2.0) * deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
        m_state.eta += xi_prime * deltaT;

        if (aniso)
            {
            // update thermostat for rotational DOF
            Scalar curr_ke_rot = m_thermo->getRotationalKineticEnergy();
            Scalar ndof_rot = m_group->getRotationalDOF();

            Scalar xi_prime_rot = m_state.xi_rot
                                  + Scalar(1.0 / 2.0) * deltaT / m_tau / m_tau
                                        * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));
            m_state.xi_rot = xi_prime_rot
                             + Scalar(1.0 / 2.0) * deltaT / m_tau / m_tau
                                   * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));

            m_state.eta_rot += xi_prime_rot * deltaT;
            }
        }

    /** Get the thermostat's contribution to the total Hamiltonian of the system.

        @param timestep Current simulation timestep.
    */
    Scalar getThermostatEnergy(uint64_t timestep)
        {
        Scalar translation_dof = m_group->getTranslationalDOF();
        Scalar thermostat_energy
            = static_cast<Scalar>(translation_dof) * m_T->operator()(timestep)
              * ((m_state.xi * m_state.xi * m_tau * m_tau / Scalar(2.0)) + m_state.eta);
        thermostat_energy += static_cast<Scalar>(m_group->getRotationalDOF())
                             * m_T->operator()(timestep)
                             * (m_state.eta_rot
                                + (m_tau * m_tau * m_state.xi_rot * m_state.xi_rot / Scalar(2.0)));

        return thermostat_energy;
        }

    /** Set the translational degrees of freedom from Python.

        @param v A Python tuple: (xi, eta).
    */
    void setTranslationalDOF(pybind11::tuple v)
        {
        if (pybind11::len(v) != 2)
            {
            throw std::length_error("translational_thermostat_dof must have length 2");
            }
        m_state.xi = v[0].cast<Scalar>();
        m_state.eta = v[1].cast<Scalar>();
        }

    /** Get the translational degrees of freedom from Python.

        @returns A Python tuple: (xi, eta).
    */
    pybind11::tuple getTranslationalDOF()
        {
        return pybind11::make_tuple(m_state.xi, m_state.eta);
        }

    /** Set the rotational degrees of freedom from Python.

        @param v A Python tuple: (xi, eta).
    */
    void setRotationalDOF(pybind11::tuple v)
        {
        if (pybind11::len(v) != 2)
            {
            throw std::length_error("rotational_thermostat_dof must have length 2");
            }
        m_state.xi_rot = v[0].cast<Scalar>();
        m_state.eta_rot = v[1].cast<Scalar>();
        }

    /** Get the rotational degrees of freedom from Python.

        @returns A Python tuple: (xi, eta).
    */
    pybind11::tuple getRotationalDOF()
        {
        return pybind11::make_tuple(m_state.xi_rot, m_state.eta_rot);
        };

    /// Get the thermostat time constant.
    Scalar getTau()
        {
        return m_tau;
        }

    /// Set the thermostat timeconstant.
    void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    /// Choose random initial values for the thermostat internal degrees of freedom.
    void thermalizeThermostat(uint64_t timestep)
        {
        auto exec_conf = m_sysdef->getParticleData()->getExecConf();
        exec_conf->msg->notice(6) << "TwoStepNVTMTK randomizing thermostat DOF" << std::endl;

        Scalar g = m_group->getTranslationalDOF();
        Scalar sigmasq_t = Scalar(1.0) / (static_cast<Scalar>(g) * m_tau * m_tau);

        bool root = exec_conf->getRank() == 0;

        unsigned int instance_id = 0;
        if (m_group->getNumMembersGlobal() > 0)
            instance_id = m_group->getMemberTag(0);

        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::MTTKThermostat, timestep, m_sysdef->getSeed()),
            hoomd::Counter(instance_id));

        if (root)
            {
            // draw a random Gaussian thermostat variable on rank 0
            m_state.xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
            }

#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            // broadcast integrator variables from rank 0 to other processors
            MPI_Bcast(&m_state.xi, 1, MPI_HOOMD_SCALAR, 0, exec_conf->getMPICommunicator());
            }
#endif

        if (m_group->getRotationalDOF() > 0)
            {
            // update thermostat for rotational DOF
            Scalar sigmasq_r = Scalar(1.0) / ((Scalar)m_group->getRotationalDOF() * m_tau * m_tau);

            if (root)
                {
                m_state.xi_rot = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_r))(rng);
                }

#ifdef ENABLE_MPI
            if (m_sysdef->isDomainDecomposed())
                {
                // broadcast integrator variables from rank 0 to other processors
                MPI_Bcast(&m_state.xi_rot, 1, MPI_HOOMD_SCALAR, 0, exec_conf->getMPICommunicator());
                }
#endif
            }
        }

    protected:
    /// The thermostat time constant.
    Scalar m_tau = 0;

    /// The thermostat internal degrees of freedom.
    MTTKThermostat::state m_state {};
    };

/** Implement the Bussi stochastic velocity rescaling thermostat.

    See: Bussi et. al. 2007 <https://doi.org/10.1063/1.2408420>.
*/
class BussiThermostat : public Thermostat
    {
    public:
    /** Construct the thermostat.

        @param T Temperature set point over time.
        @param group Group of particles this thermostat is applied to.
        @param thermo Use to compute the thermodynamic properties of the group.
        @param sysdef Used to access the simulation seed and MPI communicator.
        @param tau Thermostat time constant.
    */
    BussiThermostat(std::shared_ptr<Variant> T,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<ComputeThermo> thermo,
                    std::shared_ptr<SystemDefinition> sysdef,
                    Scalar tau)
        : Thermostat(T, group, thermo, sysdef), m_tau(tau)
        {
        }

    std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT) override
        {
        if (deltaT == 0.0)
            {
            return {1.0, 1.0};
            }

        m_thermo->compute(timestep);

        const auto translational_dof = m_thermo->getTranslationalDOF();
        const auto rotational_dof = m_thermo->getRotationalDOF();
        const auto translational_kinetic_energy = m_thermo->getTranslationalKineticEnergy();
        const auto rotational_kinetic_energy = m_thermo->getRotationalKineticEnergy();
        if ((translational_dof != 0 && translational_kinetic_energy == 0)
            || (rotational_dof != 0 && rotational_kinetic_energy == 0))
            {
            throw std::runtime_error("Bussi thermostat requires non-zero initial momenta.");
            }

        unsigned int instance_id = 0;
        if (m_group->getNumMembersGlobal() > 0)
            instance_id = m_group->getMemberTag(0);
        RandomGenerator rng(Seed(RNGIdentifier::BussiThermostat, timestep, m_sysdef->getSeed()),
                            instance_id);

        const auto set_T = m_T->operator()(timestep);

        return {
            compute_rescale_factor(translational_kinetic_energy,
                                   translational_dof,
                                   deltaT,
                                   set_T,
                                   rng),
            compute_rescale_factor(rotational_kinetic_energy, rotational_dof, deltaT, set_T, rng)};
        }

    /// Get the thermostat time constant.
    Scalar getTau()
        {
        return m_tau;
        }

    /// Set the thermostat time constant.
    void setTau(Scalar tau)
        {
        m_tau = tau;
        }

    protected:
    Scalar m_tau;

    /** Compute the rescaling factor

        @param K kinetic energy
        @param degrees_of_freedom Number of degrees of freedom with this energy.
        @param deltaT Time step size.
        @param set_T Temperature set point.
        @param rng Random number generator.
    **/
    Scalar compute_rescale_factor(Scalar K,
                                  double degrees_of_freedom,
                                  Scalar deltaT,
                                  Scalar set_T,
                                  RandomGenerator& rng)
        {
        if (degrees_of_freedom == 0)
            return Scalar(1.0);

        double time_decay_factor = 0.0;
        if (m_tau != 0.0)
            {
            time_decay_factor = exp(-deltaT / m_tau);
            }

        NormalDistribution<double> normal(1.0);
        Scalar r_normal_one = normal(rng);

        GammaDistribution<double> gamma((degrees_of_freedom - 1.0) / Scalar(2.0), Scalar(1.0));
        double r_gamma = 0.0;
        if (degrees_of_freedom > 1.0)
            {
            r_gamma = 2.0 * gamma(rng);
            }

        double v = set_T / 2.0 / K;
        double term1 = v * (1.0 - time_decay_factor) * (r_gamma + r_normal_one * r_normal_one);
        double term2 = 2.0 * r_normal_one * sqrt(v * (1.0 - time_decay_factor) * time_decay_factor);

        return Scalar(sqrt(time_decay_factor + term1 + term2));
        }
    };

/// Implement the Berendsen velocity recalcing thermostat.
class BerendsenThermostat : public Thermostat
    {
    public:
    /** Construct the thermostat.

        @param T Temperature set point over time.
        @param group Group of particles this thermostat is applied to.
        @param thermo Use to compute the thermodynamic properties of the group.
        @param sysdef Used to access the simulation seed and MPI communicator.
        @param tau Thermostat time constant.
    */
    BerendsenThermostat(std::shared_ptr<Variant> T,
                        std::shared_ptr<ParticleGroup> group,
                        std::shared_ptr<ComputeThermo> thermo,
                        std::shared_ptr<SystemDefinition> sysdef,
                        Scalar tau)
        : Thermostat(T, group, thermo, sysdef), m_tau(tau)
        {
        }
    std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, hoomd::Scalar deltaT) override
        {
        m_thermo->compute(timestep);
        Scalar current_translation_T = m_thermo->getTranslationalTemperature();
        Scalar current_rotational_T = m_thermo->getRotationalTemperature();

        if ((m_thermo->getTranslationalDOF() != 0 && m_thermo->getTranslationalKineticEnergy() == 0)
            || (m_thermo->getRotationalDOF() != 0 && m_thermo->getRotationalKineticEnergy() == 0))
            {
            throw std::runtime_error("Berendsen thermostat requires non-zero initial temperatures");
            }

        Scalar lambda_T = sqrt(
            Scalar(1.0)
            + deltaT / m_tau * (m_T->operator()(timestep) / current_translation_T - Scalar(1.0)));
        Scalar lambda_R = sqrt(
            Scalar(1.0)
            + deltaT / m_tau * (m_T->operator()(timestep) / current_rotational_T - Scalar(1.0)));

        return {lambda_T, lambda_R};
        }

    protected:
    Scalar m_tau;
    };

    } // namespace hoomd::md

#endif // HOOMD_THERMOSTAT_H
