// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.


#ifndef HOOMD_THERMOSTAT_H
#define HOOMD_THERMOSTAT_H

#include <array>
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleGroup.h>
#include <hoomd/md/ComputeThermo.h>
#include <hoomd/Variant.h>
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd::md {
    class Thermostat {
    public:
        Thermostat(std::shared_ptr<Variant> T) :
        m_T(T)
        {}

        virtual ~Thermostat() {}

        virtual std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT) {
            return {Scalar(1.0), Scalar(1.0)};
        }

        virtual std::array<Scalar, 2> getRescalingFactorsTwo(uint64_t timestep, Scalar deltaT) {
            return {Scalar(1.0), Scalar(1.0)};
        }

        virtual void advanceThermostat(uint64_t timestep, Scalar deltaT, bool aniso) {}

        virtual void thermalizeThermostat() {}

        virtual Scalar getThermostatEnergy(uint64_t timestep) {return Scalar(0.0);}

        std::shared_ptr<Variant> getT(){
            return m_T;
        }

        void setT(std::shared_ptr<Variant> T){
            m_T = T;
        }

        Scalar getTemperature(uint64_t timestep){
            return (*m_T)(timestep);
        }

        void attach(std::shared_ptr<ParticleGroup> group, std::shared_ptr<ComputeThermo> thermo, uint16_t seed){
            if(is_attached)
                throw std::runtime_error("Thermostats can only be attached to a single integrator");
            setSeed(seed);
            is_attached = true;
            m_group = group;
            m_thermo = thermo;
        }

        void detach(){
            is_attached = false;
            m_group = nullptr;
            m_thermo = nullptr;
        }

        bool attached() const{
            return is_attached;
        }

        virtual bool requiresSeed(){
            return false;
        }

        virtual void setSeed(uint16_t seed){}

#ifdef ENABLE_MPI
        virtual void broadcastThermostat(MPI_Comm){}
#endif

    protected:
        bool is_attached = false;
        std::shared_ptr<ParticleGroup> m_group;
        std::shared_ptr<ComputeThermo> m_thermo;
        std::shared_ptr<Variant> m_T;
    };


    class MTTKThermostat : public Thermostat{
    public:
        MTTKThermostat(std::shared_ptr<Variant> T,
                       Scalar tau) :
                Thermostat(T), m_tau(tau) {}


        struct state{
            Scalar xi = 0;
            Scalar eta = 0;
            Scalar xi_rot = 0;
            Scalar eta_rot = 0;
        };

        std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT) override{
            Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * m_state.xi * deltaT);
            Scalar exp_thermo_fac_rot = exp(-m_state.xi_rot * deltaT / Scalar(2.0));
            return { exp_thermo_fac, exp_thermo_fac_rot };
        }

        std::array<Scalar, 2> getRescalingFactorsTwo(uint64_t timestep, Scalar deltaT) override{
            Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * m_state.xi * deltaT);
            Scalar exp_thermo_fac_rot = exp(-m_state.xi_rot * deltaT / Scalar(2.0));
            return { exp_thermo_fac, exp_thermo_fac_rot };
        }

        void advanceThermostat(uint64_t timestep, Scalar deltaT, bool aniso = true) override{
            // compute the current thermodynamic properties
            m_thermo->compute(timestep);

            Scalar curr_T_trans = m_thermo->getTranslationalTemperature();
            Scalar T = (*m_T)(timestep);

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

        Scalar getThermostatEnergy(uint64_t timestep) override{
            Scalar translation_dof = m_group->getTranslationalDOF();
            Scalar thermostat_energy
                    = static_cast<Scalar>(translation_dof) * (*m_T)(timestep)
                      * ((m_state.xi * m_state.xi * m_tau * m_tau / Scalar(2.0)) + m_state.eta);
            thermostat_energy += static_cast<Scalar>(m_group->getRotationalDOF()) * (*m_T)(timestep)
                           * (m_state.eta_rot + (m_tau * m_tau * m_state.xi_rot * m_state.xi_rot / Scalar(2.0)));

            return thermostat_energy;
        }

        void setTranslationalDOF(pybind11::tuple v){
            if (pybind11::len(v) != 2)
            {
                throw std::length_error("translational_thermostat_dof must have length 2");
            }
            m_state.xi = v[0].cast<Scalar>();
            m_state.eta = v[1].cast<Scalar>();
        }

        pybind11::tuple getTranslationalDOF(){
            return pybind11::make_tuple(m_state.xi, m_state.eta);
        }

        void setRotationalDOF(pybind11::tuple v){
            if (pybind11::len(v) != 2)
            {
                throw std::length_error("rotational_thermostat_dof must have length 2");
            }
            m_state.xi_rot = v[0].cast<Scalar>();
            m_state.eta_rot = v[1].cast<Scalar>();
        }

        pybind11::tuple getRotationalDOF(){
            return pybind11::make_tuple(m_state.xi_rot, m_state.eta_rot);
        };

        Scalar getTau(){return m_tau;}

        void setTau(Scalar tau){m_tau = tau;}

#ifdef ENABLE_MPI
    void broadcastThermostat(MPI_Comm comm) override{
        MPI_Bcast(&m_state, 4, MPI_HOOMD_SCALAR, 0, comm);
        }
#endif

    protected:
        Scalar m_tau = 0;
        MTTKThermostat::state m_state{};
    };

    class BussiThermostat : public Thermostat{
    public:
        BussiThermostat(std::shared_ptr<Variant> T) : Thermostat(T) {}

        std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, Scalar deltaT) override{
            m_thermo->compute(timestep);

            const auto ntdof = m_thermo->getTranslationalDOF();
            const auto nrdof = m_thermo->getRotationalDOF();
            if((ntdof != 0 && m_thermo->getTranslationalKineticEnergy() == 0) ||
               (nrdof != 0 && m_thermo->getRotationalKineticEnergy() == 0))
            {
                throw std::runtime_error("Bussi thermostat requires non-zero initial temperatures");
            }
            unsigned int instance_id = 0;
            if (m_group->getNumMembersGlobal() > 0)
                instance_id = m_group->getMemberTag(0);
            RandomGenerator rng(Seed(RNGIdentifier::StochasticVRescale, timestep, m_seed), instance_id);

            const auto set_T = (*m_T)(timestep);
            GammaDistribution<double> gamma_translation(ntdof / 2.0, set_T);
            GammaDistribution<double> gamma_rotation(nrdof / 2.0, set_T);

            return{
                    std::sqrt(gamma_translation(rng) / m_thermo->getTranslationalKineticEnergy()),
                    std::sqrt(gamma_rotation(rng) / m_thermo->getRotationalKineticEnergy())
            };
        }

        bool requiresSeed() override{
            return true;
        }

        void setSeed(uint16_t seed) override{
            m_seed = seed;
        }

    protected:
        uint16_t m_seed;
    };

    class BerendsenThermostat : public Thermostat{
    public:
        BerendsenThermostat(std::shared_ptr<Variant> T, Scalar tau) :
                Thermostat(T), m_tau(tau) {}
        std::array<Scalar, 2> getRescalingFactorsOne(uint64_t timestep, hoomd::Scalar deltaT) override{
            m_thermo->compute(timestep);
            Scalar current_translation_T = m_thermo->getTranslationalTemperature();
            Scalar current_rotational_T = m_thermo->getRotationalTemperature();

            if((m_thermo->getTranslationalDOF() != 0 && m_thermo->getTranslationalKineticEnergy() == 0) ||
               (m_thermo->getRotationalDOF() != 0 && m_thermo->getRotationalKineticEnergy() == 0))
            {
                throw std::runtime_error("Berendsen thermostat requires non-zero initial temperatures");
            }

            Scalar lambda_T = sqrt(Scalar(1.0) + deltaT / m_tau * ((*m_T)(timestep) / current_translation_T - Scalar(1.0)));
            Scalar lambda_R = sqrt(Scalar(1.0) + deltaT / m_tau * ((*m_T)(timestep) / current_rotational_T - Scalar(1.0)));

            return {lambda_T, lambda_R};
        }

    protected:
        Scalar m_tau;
    };

}


#endif // HOOMD_THERMOSTAT_H
