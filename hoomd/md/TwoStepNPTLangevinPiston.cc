//
// Created by girard01 on 10/28/22.
//


#include "TwoStepNPTLangevinPiston.h"
#include "hoomd/RNGIdentifiers.h"
#include <hoomd/RandomNumbers.h>

namespace hoomd::md{

void TwoStepNPTLangevinPiston::advanceBarostat(uint64_t timestep)
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
    }

namespace detail{
void export_TwoStepNPTLangevinPiston(pybind11::module& m){
    pybind11::class_<TwoStepNPTLangevinPiston, TwoStepNPTMTK, std::shared_ptr<TwoStepNPTLangevinPiston>>(m, "TwoStepNPTLangevinPiston")
        .def(pybind11::init<std::shared_ptr<SystemDefinition> ,
                            std::shared_ptr<ParticleGroup> ,
                            std::shared_ptr<ComputeThermo> ,
                            std::shared_ptr<ComputeThermo> ,
                            Scalar ,
                            Scalar ,
                            std::shared_ptr<Variant> ,
                            const std::vector<std::shared_ptr<Variant>>& ,
                            const std::string& ,
                            const std::vector<bool>& ,
                            const bool >());
    }
    }

    }