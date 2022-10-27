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
//! Coefficients of f(x) = sinh(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 *
//! x^10
const Scalar f_coeff[] = {Scalar(1.0),
                          Scalar(1.0 / 6.0),
                          Scalar(1.0 / 120.0),
                          Scalar(1.0 / 5040.0),
                          Scalar(1.0 / 362880.0),
                          Scalar(1.0 / 39916800.0)};

//! Coefficients of g(x) = coth(x) - 1/x =  a_1 * x + a_3 * x^3 + a_5 * x^5 + a_7 * x^7 + a_9 * x^9
const Scalar g_coeff[] = {Scalar(1.0 / 3.0),
                          Scalar(-1.0 / 45.0),
                          Scalar(2.0 / 945.0),
                          Scalar(-1.0 / 4725.0),
                          Scalar(1.0 / 93555.0)};

//! Coefficients of h(x) = (-1/sinh^2(x)+1/x^2) = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 *
//! x^8 + a_10 * x^10
const Scalar h_coeff[] = {Scalar(1.0 / 3.0),
                          Scalar(-1.0 / 15.0),
                          Scalar(2.0 / 189.0),
                          Scalar(-1.0 / 675.0),
                          Scalar(2.0 / 10395.0),
                          Scalar(-1382.0 / 58046625.0)};

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
                             const bool nph) :
      TwoStepNPTBase(sysdef, group, thermo_half_step, thermo_full_step, T, S, couple, flags, nph),
    m_tau(tau), m_tauS(tauS), m_gamma(0.0)
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

void TwoStepNPTMTK::updatePropagator()
    {
    // calculate some factors needed for the update matrix
    Scalar3 v_fac = make_scalar3(-Scalar(1.0 / 4.0) * m_barostat.nu_xx,
                                 -Scalar(1.0 / 4.0) * m_barostat.nu_yy,
                                 -Scalar(1.0 / 4.0) * m_barostat.nu_zz);
    Scalar3 exp_v_fac_2 = make_scalar3(exp(Scalar(2.0) * v_fac.x * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.y * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.z * m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0 / 2.0) * m_barostat.nu_xx,
                                 Scalar(1.0 / 2.0) * m_barostat.nu_yy,
                                 Scalar(1.0 / 2.0) * m_barostat.nu_zz);
    Scalar3 exp_r_fac = make_scalar3(exp(Scalar(1.0 / 2.0) * m_barostat.nu_xx * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * m_barostat.nu_yy * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * m_barostat.nu_zz * m_deltaT));
    Scalar3 exp_r_fac_2 = make_scalar3(exp(m_barostat.nu_xx * m_deltaT),
                                       exp(m_barostat.nu_yy * m_deltaT),
                                       exp(m_barostat.nu_zz * m_deltaT));

    // Calculate power series approximations of analytical functions entering the update equations

    Scalar3 arg_v = v_fac * m_deltaT;
    Scalar3 arg_r = r_fac * m_deltaT;

    // Calculate function f = sinh(x)/x
    Scalar3 f_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 f_r = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 term_v = make_scalar3(1.0, 1.0, 1.0);
    Scalar3 term_r = make_scalar3(1.0, 1.0, 1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        f_v += f_coeff[i] * term_v;
        f_r += f_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function g = cth(x) - 1/x
    Scalar3 g_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 g_r = make_scalar3(0.0, 0.0, 0.0);

    term_v = arg_v;
    term_r = arg_r;

    for (unsigned int i = 0; i < 5; i++)
        {
        g_v += g_coeff[i] * term_v;
        g_r += g_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function h = -1/sinh^2(x) + 1/x^2
    Scalar3 h_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 h_r = make_scalar3(0.0, 0.0, 0.0);

    term_v = term_r = make_scalar3(1.0, 1.0, 1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        h_v += h_coeff[i] * term_v;
        h_r += h_coeff[i] * term_r;

        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate matrix exponentials for upper triangular barostat matrix
    /* These are approximations accurate up to and including delta_t^2.
       They are fully time reversible  */

    // Matrix exp. for velocity update
    m_mat_exp_v[0] = exp_v_fac_2.x; // xx
    m_mat_exp_v[1]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_xy * (exp_v_fac_2.x + exp_v_fac_2.y); // xy
    m_mat_exp_v[2]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_xz * (exp_v_fac_2.x + exp_v_fac_2.z)
          + m_deltaT * m_deltaT * Scalar(1.0 / 32.0) * m_barostat.nu_xy * m_barostat.nu_yz
                * (exp_v_fac_2.x + Scalar(2.0) * exp_v_fac_2.y + exp_v_fac_2.z); // xz
    m_mat_exp_v[3] = exp_v_fac_2.y;                                              // yy
    m_mat_exp_v[4]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_yz * (exp_v_fac_2.y + exp_v_fac_2.z); // yz
    m_mat_exp_v[5] = exp_v_fac_2.z;                                                           // zz

    // Matrix exp. for position update
    m_mat_exp_r[0] = exp_r_fac_2.x; // xx
    m_mat_exp_r[1]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_xy * (exp_r_fac_2.x + exp_r_fac_2.y); // xy
    m_mat_exp_r[2]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_xz * (exp_r_fac_2.x + exp_r_fac_2.z)
          + m_deltaT * m_deltaT * Scalar(1.0 / 8.0) * m_barostat.nu_xy * m_barostat.nu_yz
                * (exp_r_fac_2.x + Scalar(2.0) * exp_r_fac_2.y + exp_r_fac_2.z); // xz
    m_mat_exp_r[3] = exp_r_fac_2.y;                                              // yy
    m_mat_exp_r[4]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_yz * (exp_r_fac_2.y + exp_r_fac_2.z); // yz
    m_mat_exp_r[5] = exp_r_fac_2.z;                                                          // zz

    // integrated matrix exp. for position update
    Scalar3 xz_fac_r = make_scalar3((Scalar(1.0) + g_r.x) * (Scalar(1.0) + g_r.x) + h_r.x,
                                    (Scalar(1.0) + g_r.y) * (Scalar(1.0) + g_r.y) + h_r.y,
                                    (Scalar(1.0) + g_r.z) * (Scalar(1.0) + g_r.z) + h_r.z);

    m_mat_exp_r_int[0] = m_deltaT * exp_r_fac.x * f_r.x; // xx
    m_mat_exp_r_int[1] = m_deltaT * m_deltaT * m_barostat.nu_xy * Scalar(1.0 / 4.0)
                         * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                            + exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)); // xy
    m_mat_exp_r_int[2]
        = m_deltaT * m_deltaT * m_barostat.nu_xz * Scalar(1.0 / 4.0)
              * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                 + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z))
          + m_deltaT * m_deltaT * m_deltaT * m_barostat.nu_xy * m_barostat.nu_yz
                * Scalar(1.0 / 32.0)
                * (exp_r_fac.x * f_r.x * xz_fac_r.x + Scalar(2.0) * exp_r_fac.y * f_r.y * xz_fac_r.y
                   + exp_r_fac.z * f_r.z * xz_fac_r.z);  // xz
    m_mat_exp_r_int[3] = m_deltaT * exp_r_fac.y * f_r.y; // yy
    m_mat_exp_r_int[4] = m_deltaT * m_deltaT * m_barostat.nu_yz * Scalar(1.0 / 4.0)
                         * (exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)
                            + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z)); // yz
    m_mat_exp_r_int[5] = m_deltaT * exp_r_fac.z * f_r.z;                    // zz
    }

std::array<Scalar, 2> TwoStepNPTMTK::NPT_thermo_rescale_factor_one(uint64_t timestep)
    {
        Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;
        Scalar xi_trans = m_thermostat.xi;
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);
        const Scalar xi_rot = m_thermostat.xi_rot;
        Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));
        return {
            exp_thermo_fac, exp_thermo_fac_rot
            };
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
    Scalar3 R_diag = make_scalar3(0., 0., 0.);

    RandomGenerator rng(Seed(RNGIdentifier::LangevinPiston, timestep, m_sysdef->getSeed()), 0);
    NormalDistribution<Scalar> noise;

    switch (couple)
        {
    case couple_none:
        P_diag.x = P.xx;
        R_diag.x = noise(rng);
        P_diag.y = P.yy;
        R_diag.y = noise(rng);
        P_diag.z = P.zz;
        R_diag.z = noise(rng);
        break;
    case couple_xy:
        P_diag.x = P_diag.y = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        R_diag.x = R_diag.y = noise(rng);
        P_diag.z = P.zz;
        R_diag.z = noise(rng);
        break;
    case couple_xz:
        P_diag.x = P_diag.z = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        R_diag.x = R_diag.z = noise(rng);
        P_diag.y = P.yy;
        R_diag.y = noise(rng);
        break;
    case couple_yz:
        P_diag.x = P.xx;
        R_diag.x = noise(rng);
        P_diag.y = P_diag.z = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        R_diag.y = R_diag.z = noise(rng);
        break;
    case couple_xyz:
        P_diag.x = P_diag.y = P_diag.z = Scalar(1.0 / 3.0) * (P.xx + P.yy + P.zz);
        R_diag.x = R_diag.y = R_diag.z = noise(rng);
        break;
    default:
        throw std::runtime_error("Invalid NPT coupling mode.");
        }

    // update barostat matrix

    Scalar noise_exp_integrate = exp(-m_gamma * m_deltaT / Scalar(2.0));
    Scalar coeff = sqrt((*m_T)(timestep) * (Scalar(1.0) - noise_exp_integrate * noise_exp_integrate) / W);

    if (m_flags & baro_x)
        {
        m_barostat.nu_xx  = m_barostat.nu_xx * noise_exp_integrate + coeff * R_diag.x;
        m_barostat.nu_xx += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.x - (*m_S[0])(timestep)) + mtk_term;
        }

    if (m_flags & baro_xy)
        {
        m_barostat.nu_xy = m_barostat.nu_xy * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_xy += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xy - (*m_S[5])(timestep));
        }

    if (m_flags & baro_xz)
        {
        m_barostat.nu_xz = m_barostat.nu_xz * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_xz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xz - (*m_S[4])(timestep));
        }

    if (m_flags & baro_y)
        {
        m_barostat.nu_yy = m_barostat.nu_yy * noise_exp_integrate + coeff * R_diag.y;
        m_barostat.nu_yy  += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.y - (*m_S[1])(timestep)) + mtk_term;
        }

    if (m_flags & baro_yz)
        {
        m_barostat.nu_yz = m_barostat.nu_yz * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_yz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.yz - (*m_S[3])(timestep));
        }

    if (m_flags & baro_z)
        {
        m_barostat.nu_zz = m_barostat.nu_zz * noise_exp_integrate + coeff * R_diag.z;
        m_barostat.nu_zz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.z - (*m_S[2])(timestep)) + mtk_term;
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
    m_exec_conf->msg->notice(6) << "TwoStepNPTMTK randomizing thermostat and barostat DOF"
                                << std::endl;

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

    // randomize barostat variables
    unsigned int d = m_sysdef->getNDimensions();
    Scalar sigmasq_baro = Scalar(1.0) / ((Scalar)(m_ndof + d) / (Scalar)d * m_tauS * m_tauS);

    if (master)
        {
        if (m_flags & baro_x)
            {
            m_barostat.nu_xx = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xy)
            {
            m_barostat.nu_xy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xz)
            {
            m_barostat.nu_xz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_y)
            {
            m_barostat.nu_yy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_yz)
            {
            m_barostat.nu_yz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_z)
            {
            m_barostat.nu_zz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        // couple box degrees of freedom
        couplingMode couple = getRelevantCouplings();

        switch (couple)
            {
        case couple_none:
            break;
        case couple_xy:
            m_barostat.nu_yy = m_barostat.nu_xx;
            break;
        case couple_xz:
            m_barostat.nu_zz = m_barostat.nu_xx;
            break;
        case couple_yz:
            m_barostat.nu_yy = m_barostat.nu_zz;
            break;
        case couple_xyz:
            m_barostat.nu_xx = m_barostat.nu_zz;
            m_barostat.nu_yy = m_barostat.nu_zz;
            break;
        default:
            throw std::runtime_error("Invalid NPT coupling mode.");
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_thermostat, 4, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&m_barostat, 6, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
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

pybind11::tuple TwoStepNPTMTK::getBarostatDOF()
    {
    return pybind11::make_tuple(m_barostat.nu_xx,
                                m_barostat.nu_xy,
                                m_barostat.nu_xz,
                                m_barostat.nu_yy,
                                m_barostat.nu_yz,
                                m_barostat.nu_zz);
    }

void TwoStepNPTMTK::setBarostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 6)
        {
        throw std::length_error("barostat_dof must have length 6");
        }
    m_barostat.nu_xx = v[0].cast<Scalar>();
    m_barostat.nu_xy = v[1].cast<Scalar>();
    m_barostat.nu_xz = v[2].cast<Scalar>();
    m_barostat.nu_yy = v[3].cast<Scalar>();
    m_barostat.nu_yz = v[4].cast<Scalar>();
    m_barostat.nu_zz = v[5].cast<Scalar>();
    }

Scalar TwoStepNPTMTK::getBarostatEnergy(uint64_t timestep)
    {
    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = static_cast<Scalar>(m_ndof + d) / static_cast<Scalar>(d) * (*m_T)(timestep)*m_tauS
               * m_tauS;

    Scalar barostat_energy
        = W
          * (m_barostat.nu_xx * m_barostat.nu_xx + m_barostat.nu_yy * m_barostat.nu_yy
             + m_barostat.nu_zz * m_barostat.nu_zz // Normal
             + m_barostat.nu_xy * m_barostat.nu_xy + m_barostat.nu_xz * m_barostat.nu_xz
             + m_barostat.nu_yz * m_barostat.nu_yz // Shear
             )
          / Scalar(2.0);

    return barostat_energy;
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
        .def_property("tauS", &TwoStepNPTMTK::getTauS, &TwoStepNPTMTK::setTauS)
        .def_property("gamma", &TwoStepNPTMTK::getGamma, &TwoStepNPTMTK::setGamma)
        .def("thermalizeThermostatAndBarostatDOF",
             &TwoStepNPTMTK::thermalizeThermostatAndBarostatDOF)
        .def_property("translational_thermostat_dof",
                      &TwoStepNPTMTK::getTranslationalThermostatDOF,
                      &TwoStepNPTMTK::setTranslationalThermostatDOF)
        .def_property("rotational_thermostat_dof",
                      &TwoStepNPTMTK::getRotationalThermostatDOF,
                      &TwoStepNPTMTK::setRotationalThermostatDOF)
        .def_property("barostat_dof",
                      &TwoStepNPTMTK::getBarostatDOF,
                      &TwoStepNPTMTK::setBarostatDOF)
        .def("getThermostatEnergy", &TwoStepNPTMTK::getThermostatEnergy)
        .def("getBarostatEnergy", &TwoStepNPTMTK::getBarostatEnergy);
    }

    } // end namespace detail
    } // end namespace hoomd
