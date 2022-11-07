// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVTMTK.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

/*! \file TwoStepNVTMTK.h
    \brief Contains code for the TwoStepNVTMTK class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
*/
TwoStepNVTMTK::TwoStepNVTMTK(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo,
                             Scalar tau,
                             std::shared_ptr<Variant> T)
    : TwoStepNVTBase(sysdef, group, thermo, T), m_tau(tau), m_exp_thermo_fac(1.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTMTK" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt: tau set less than 0.0 in NVTUpdater" << endl;
    }

TwoStepNVTMTK::~TwoStepNVTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTMTK" << endl;
    }

void TwoStepNVTMTK::thermalizeThermostatDOF(uint64_t timestep)
    {
    m_exec_conf->msg->notice(6) << "TwoStepNVTMTK randomizing thermostat DOF" << std::endl;

    Scalar g = m_group->getTranslationalDOF();
    Scalar sigmasq_t = Scalar(1.0) / ((Scalar)g * m_tau * m_tau);

    bool root = m_exec_conf->getRank() == 0;

    unsigned int instance_id = 0;
    if (m_group->getNumMembersGlobal() > 0)
        instance_id = m_group->getMemberTag(0);

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::TwoStepNVTMTK, timestep, m_sysdef->getSeed()),
        hoomd::Counter(instance_id));

    if (root)
        {
        // draw a random Gaussian thermostat variable on rank 0
        m_thermostat.xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_thermostat.xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar sigmasq_r = Scalar(1.0) / ((Scalar)m_group->getRotationalDOF() * m_tau * m_tau);

        if (root)
            {
            m_thermostat.xi_rot = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_r))(rng);
            }

#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            // broadcast integrator variables from rank 0 to other processors
            MPI_Bcast(&m_thermostat.xi_rot,
                      1,
                      MPI_HOOMD_SCALAR,
                      0,
                      m_exec_conf->getMPICommunicator());
            }
#endif
        }
    }

pybind11::tuple TwoStepNVTMTK::getTranslationalThermostatDOF()
    {
    return pybind11::make_tuple(m_thermostat.xi, m_thermostat.eta);
    }

void TwoStepNVTMTK::setTranslationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("translational_thermostat_dof must have length 2");
        }
    m_thermostat.xi = v[0].cast<Scalar>();
    m_thermostat.eta = v[1].cast<Scalar>();
    }

pybind11::tuple TwoStepNVTMTK::getRotationalThermostatDOF()
    {
    return pybind11::make_tuple(m_thermostat.xi_rot, m_thermostat.eta_rot);
    }

void TwoStepNVTMTK::setRotationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("rotational_thermostat_dof must have length 2");
        }
    m_thermostat.xi_rot = v[0].cast<Scalar>();
    m_thermostat.eta_rot = v[1].cast<Scalar>();
    }

Scalar TwoStepNVTMTK::getThermostatEnergy(uint64_t timestep)
    {
    Scalar translation_dof = m_group->getTranslationalDOF();
    Scalar thermostat_energy
        = static_cast<Scalar>(translation_dof) * (*m_T)(timestep)
          * ((m_thermostat.xi * m_thermostat.xi * m_tau * m_tau / Scalar(2.0)) + m_thermostat.eta);

    if (m_aniso)
        {
        thermostat_energy
            += static_cast<Scalar>(m_group->getRotationalDOF()) * (*m_T)(timestep)
               * (m_thermostat.eta_rot
                  + (m_tau * m_tau * m_thermostat.xi_rot * m_thermostat.xi_rot / Scalar(2.0)));
        }

    return thermostat_energy;
    }

void TwoStepNVTMTK::advanceThermostat(uint64_t timestep, bool broadcast)
    {
    // compute the current thermodynamic properties
    TwoStepNVTMTK::m_thermo->compute(timestep + 1);

    Scalar curr_T_trans = TwoStepNVTMTK::m_thermo->getTranslationalTemperature();

    // update the state variables Xi and eta
    Scalar xi_prime = TwoStepNVTMTK::m_thermostat.xi
                      + Scalar(1.0 / 2.0) * m_deltaT / TwoStepNVTMTK::m_tau / TwoStepNVTMTK::m_tau
                            * (curr_T_trans / (*m_T)(timestep)-Scalar(1.0));
    TwoStepNVTMTK::m_thermostat.xi = xi_prime
                                     + Scalar(1.0 / 2.0) * m_deltaT / TwoStepNVTMTK::m_tau / TwoStepNVTMTK::m_tau
                                           * (curr_T_trans / (*m_T)(timestep)-Scalar(1.0));
    TwoStepNVTMTK::m_thermostat.eta += xi_prime * m_deltaT;

    // update loop-invariant quantity
    TwoStepNVTMTK::m_exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * TwoStepNVTMTK::m_thermostat.xi * m_deltaT);

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar curr_ke_rot = TwoStepNVTMTK::m_thermo->getRotationalKineticEnergy();
        Scalar ndof_rot = m_group->getRotationalDOF();

        Scalar xi_prime_rot
            = TwoStepNVTMTK::m_thermostat.xi_rot
              + Scalar(1.0 / 2.0) * m_deltaT / TwoStepNVTMTK::m_tau / TwoStepNVTMTK::m_tau
                    * (Scalar(2.0) * curr_ke_rot / ndof_rot / (*m_T)(timestep)-Scalar(1.0));
        TwoStepNVTMTK::m_thermostat.xi_rot
            = xi_prime_rot
              + Scalar(1.0 / 2.0) * m_deltaT / TwoStepNVTMTK::m_tau / TwoStepNVTMTK::m_tau
                    * (Scalar(2.0) * curr_ke_rot / ndof_rot / (*m_T)(timestep)-Scalar(1.0));

        TwoStepNVTMTK::m_thermostat.eta_rot += xi_prime_rot * m_deltaT;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed() && broadcast)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_thermostat, 4, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

namespace detail
    {
void export_TwoStepNVTMTK(pybind11::module& m)
    {
    pybind11::class_<TwoStepNVTMTK, TwoStepNVTBase, std::shared_ptr<TwoStepNVTMTK>>(
        m,
        "TwoStepNVTMTK",
        pybind11::multiple_inheritance())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>())
        .def("setTau", &TwoStepNVTMTK::setTau)
        .def_property("tau", &TwoStepNVTMTK::getTau, &TwoStepNVTMTK::setTau)
        .def("thermalizeThermostatDOF", &TwoStepNVTMTK::thermalizeThermostatDOF)
        .def_property("translational_thermostat_dof",
                      &TwoStepNVTMTK::getTranslationalThermostatDOF,
                      &TwoStepNVTMTK::setTranslationalThermostatDOF)
        .def_property("rotational_thermostat_dof",
                      &TwoStepNVTMTK::getRotationalThermostatDOF,
                      &TwoStepNVTMTK::setRotationalThermostatDOF)
        .def("getThermostatEnergy", &TwoStepNVTMTK::getThermostatEnergy);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
