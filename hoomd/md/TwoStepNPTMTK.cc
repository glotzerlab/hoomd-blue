// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNPTMTK.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

using namespace std;

/*! \file TwoStepNPTMTK.cc
    \brief Contains code for the TwoStepNPTMTK class
*/

namespace hoomd::md
    {


/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_half_step Own ComputeThermo to compute thermo properties of the integrated \a
   group (at half time step) \param thermo_half_step_full_tstep ComputeThermo to compute thermo
   properties of the integrated \a group at full time step \param tau NPT temperature period \param
   tauS NPT pressure period \param T Temperature set point \param S Pressure or Stress set point.
   Pressure if one value, Stress if a list of 6. Stress should be ordered as [xx, yy, zz, yz, xz,
   xy] \param couple Coupling mode \param flags Barostatted simulation box degrees of freedom
*/
TwoStepNPTMTK::TwoStepNPTMTK(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo_half_step,
                             std::shared_ptr<ComputeThermo> thermo_full_step,
                             Scalar tau,
                             Scalar tauS,
                             std::shared_ptr<Variant> T,
                             const std::vector<std::shared_ptr<Variant>>& S,
                             const std::string& couple,
                             const std::vector<bool>& flags,
                             const bool nph) : TwoStepNPTMTTKBase(sysdef, group, thermo_half_step, thermo_full_step, tauS, T, S, couple, flags, nph),
    m_tau(tau), m_gamma(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTK" << endl;
    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tau set less than 0.0" << endl;
    if (m_tauS <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tauS set less than 0.0" << endl;
    }

TwoStepNPTMTK::~TwoStepNPTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTK" << endl;
    }


std::array<Scalar, 2> TwoStepNPTMTK::NPT_thermo_rescale_factor_one(uint64_t timestep)
    {
        Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;
        const Scalar xi_trans = m_thermostat.xi;
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);
        const Scalar xi_rot = m_thermostat.xi_rot;
        Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));
        return { exp_thermo_fac, exp_thermo_fac_rot };
    }

std::array<Scalar, 2> TwoStepNPTMTK::NPT_thermo_rescale_factor_two(uint64_t timestep)
    {
    Scalar xi_trans = m_thermostat.xi;
    Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;
    Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);
    const Scalar xi_rot = m_thermostat.xi_rot;
    Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));
    return {exp_thermo_fac, exp_thermo_fac_rot};
    }

//! Helper function to advance the barostat parameters
void TwoStepNPTMTK::advanceBarostat(uint64_t timestep)
    {
    // compute thermodynamic properties at full time step
    m_thermo_full_step->compute(timestep);

    // compute pressure for the next half time step
    PressureTensor P = m_thermo_full_step->getPressureTensor();

    if (std::isnan(P.xx) || std::isnan(P.xy) || std::isnan(P.xz) || std::isnan(P.yy)
        || std::isnan(P.yz) || std::isnan(P.zz))
        {
        P.xx = (*m_S[0])(timestep);
        P.yy = (*m_S[1])(timestep);
        P.zz = (*m_S[2])(timestep);
        P.yz = (*m_S[3])(timestep);
        P.xz = (*m_S[4])(timestep);
        P.xy = (*m_S[5])(timestep);
        }

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    // Martyna-Tobias-Klein correction
    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = (Scalar)(m_ndof + d) / (Scalar)d * (*m_T)(timestep)*m_tauS * m_tauS;
    Scalar mtk_term = Scalar(2.0) * m_thermo_full_step->getTranslationalKineticEnergy();
    mtk_term *= Scalar(1.0 / 2.0) * m_deltaT / (Scalar)m_ndof / W;

    couplingMode couple = getRelevantCouplings();

    // couple diagonal elements of pressure tensor together
    Scalar3 P_diag = make_scalar3(0.0, 0.0, 0.0);

    if (couple == couple_none)
        {
        P_diag.x = P.xx;
        P_diag.y = P.yy;
        P_diag.z = P.zz;
        }
    else if (couple == couple_xy)
        {
        P_diag.x = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        P_diag.y = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        P_diag.z = P.zz;
        }
    else if (couple == couple_xz)
        {
        P_diag.x = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        P_diag.y = P.yy;
        P_diag.z = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        }
    else if (couple == couple_yz)
        {
        P_diag.x = P.xx;
        P_diag.y = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        P_diag.z = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        }
    else if (couple == couple_xyz)
        {
        Scalar P_iso = Scalar(1.0 / 3.0) * (P.xx + P.yy + P.zz);
        P_diag.x = P_diag.y = P_diag.z = P_iso;
        }
    else
        {
        throw std::runtime_error("Invalid NPT coupling mode.");
        }

    // update barostat matrix

    if (m_flags & baro_x)
        {
        m_barostat.nu_xx
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.x - (*m_S[0])(timestep)) + mtk_term;
        m_barostat.nu_xx -= m_gamma * m_barostat.nu_xx;
        }

    if (m_flags & baro_xy)
        {
        m_barostat.nu_xy += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xy - (*m_S[5])(timestep));
        m_barostat.nu_xy -= m_gamma * m_barostat.nu_xy;
        }

    if (m_flags & baro_xz)
        {
        m_barostat.nu_xz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xz - (*m_S[4])(timestep));
        m_barostat.nu_xz -= m_gamma * m_barostat.nu_xz;
        }

    if (m_flags & baro_y)
        {
        m_barostat.nu_yy
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.y - (*m_S[1])(timestep)) + mtk_term;
        m_barostat.nu_yy -= m_gamma * m_barostat.nu_yy;
        }

    if (m_flags & baro_yz)
        {
        m_barostat.nu_yz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.yz - (*m_S[3])(timestep));
        m_barostat.nu_yz -= m_gamma * m_barostat.nu_yz;
        }

    if (m_flags & baro_z)
        {
        m_barostat.nu_zz
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.z - (*m_S[2])(timestep)) + mtk_term;
        m_barostat.nu_zz -= m_gamma * m_barostat.nu_zz;
        }

    // store integrator variables
    }

void TwoStepNPTMTK::advanceThermostat(uint64_t timestep)
    {
    // compute the current thermodynamic properties
    m_thermo_half_step->compute(timestep);

    Scalar curr_T_trans = m_thermo_half_step->getTranslationalTemperature();
    Scalar T = (*m_T)(timestep);

    // update the state variables Xi and eta
    Scalar xi_prime
        = m_thermostat.xi
          + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
    m_thermostat.xi
        = xi_prime
          + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
    m_thermostat.eta += xi_prime * m_deltaT;

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar curr_ke_rot = m_thermo_half_step->getRotationalKineticEnergy();
        Scalar ndof_rot = m_group->getRotationalDOF();

        Scalar xi_prime_rot = m_thermostat.xi_rot
                              + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau
                                    * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));
        m_thermostat.xi_rot = xi_prime_rot
                              + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau
                                    * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));

        m_thermostat.eta_rot += xi_prime_rot * m_deltaT;
        }
    }

void TwoStepNPTMTK::thermalizeThermostatAndBarostatDOF(uint64_t timestep)
    {

    TwoStepNPTMTTKBase::thermalizeBarostatDOF(timestep);

    m_exec_conf->msg->notice(6) << "TwoStepNPTMTK randomizing thermostat DOF"  << std::endl;

    unsigned int instance_id = 0;
    if (m_group->getNumMembersGlobal() > 0)
        instance_id = m_group->getMemberTag(0);

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::TwoStepNPTMTK, timestep, m_sysdef->getSeed()),
        hoomd::Counter(instance_id));

    bool master = m_exec_conf->getRank() == 0;

    if (!m_nph)
        {
        // randomize thermostat variables
        Scalar g = m_group->getTranslationalDOF();
        Scalar sigmasq_t = Scalar(1.0) / (g * m_tau * m_tau);

        if (master)
            {
            // draw a random Gaussian thermostat variable on rank 0
            m_thermostat.xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
            }

        if (m_aniso)
            {
            // update thermostat for rotational DOF
            Scalar sigmasq_r = Scalar(1.0) / ((Scalar)m_group->getRotationalDOF() * m_tau * m_tau);

            if (master)
                {
                m_thermostat.xi_rot = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_r))(rng);
                }
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_thermostat, 4, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

pybind11::tuple TwoStepNPTMTK::getTranslationalThermostatDOF()
    {
    return pybind11::make_tuple(m_thermostat.xi, m_thermostat.eta);
    }

void TwoStepNPTMTK::setTranslationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("translational_thermostat_dof must have length 2");
        }
    m_thermostat.xi = v[0].cast<Scalar>();
    m_thermostat.eta = v[1].cast<Scalar>();
    }

pybind11::tuple TwoStepNPTMTK::getRotationalThermostatDOF()
    {
    return pybind11::make_tuple(m_thermostat.xi_rot, m_thermostat.eta_rot);
    }

void TwoStepNPTMTK::setRotationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("rotational_thermostat_dof must have length 2");
        }
    m_thermostat.xi_rot = v[0].cast<Scalar>();
    m_thermostat.eta_rot = v[1].cast<Scalar>();
    }

Scalar TwoStepNPTMTK::getThermostatEnergy(uint64_t timestep)
    {
    const Scalar xi = m_thermostat.xi;
    const Scalar eta = m_thermostat.eta;

    Scalar thermostat_energy = m_group->getTranslationalDOF() * (*m_T)(timestep)
                               * (eta + m_tau * m_tau * xi * xi / Scalar(2.0));

    if (m_aniso)
        {
        const Scalar xi_rot = m_thermostat.xi_rot;
        const Scalar eta_rot = m_thermostat.eta_rot;
        thermostat_energy += m_group->getRotationalDOF() * (*m_T)(timestep)
                             * (eta_rot + m_tau * m_tau * xi_rot * xi_rot / Scalar(2.0));
        }

    return thermostat_energy;
    }


namespace detail
    {
void export_TwoStepNPTMTK(pybind11::module& m)
    {
    pybind11::class_<TwoStepNPTMTK, IntegrationMethodTwoStep, std::shared_ptr<TwoStepNPTMTK>>
        twostepnptmtk(m, "TwoStepNPTMTK");
    twostepnptmtk
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            Scalar,
                            std::shared_ptr<Variant>,
                            const std::vector<std::shared_ptr<Variant>>&,
                            const string&,
                            const std::vector<bool>&,
                            const bool>())

        .def_property("gamma", &TwoStepNPTMTK::getGamma, &TwoStepNPTMTK::setGamma)
        .def_property("tauS", &TwoStepNPTMTK::getTau, &TwoStepNPTMTK::setTau)
        .def("thermalizeThermostatAndBarostatDOF",
             &TwoStepNPTMTK::thermalizeThermostatAndBarostatDOF)
        .def_property("translational_thermostat_dof",
                      &TwoStepNPTMTK::getTranslationalThermostatDOF,
                      &TwoStepNPTMTK::setTranslationalThermostatDOF)
        .def_property("rotational_thermostat_dof",
                      &TwoStepNPTMTK::getRotationalThermostatDOF,
                      &TwoStepNPTMTK::setRotationalThermostatDOF)
        .def("getThermostatEnergy", &TwoStepNPTMTK::getThermostatEnergy);
    }

    } // end namespace detail
    } // end namespace hoomd
