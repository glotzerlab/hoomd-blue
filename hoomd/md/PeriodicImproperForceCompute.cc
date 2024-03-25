// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproperForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
PeriodicImproperForceCompute::PeriodicImproperForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing PeriodicImproperForceCompute" << endl;

    // access the improper data for later use
    m_improper_data = m_sysdef->getImproperData();

    // allocate the parameters
    GPUArray<periodic_improper_params> params(m_improper_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

PeriodicImproperForceCompute::~PeriodicImproperForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PeriodicImproperForceCompute" << endl;
    }

void PeriodicImproperForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    // make sure the type is valid
    auto typ = m_improper_data->getTypeByName(type);
    periodic_improper_params _params(params);
    ArrayHandle<periodic_improper_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::readwrite);
    h_params.data[typ] = _params;
    }

pybind11::dict PeriodicImproperForceCompute::getParams(std::string type)
    {
    auto typ = m_improper_data->getTypeByName(type);
    pybind11::dict params;

    ArrayHandle<periodic_improper_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::read);
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void PeriodicImproperForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<periodic_improper_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::read);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // for each of the impropers
    const unsigned int size = (unsigned int)m_improper_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the improper
        const ImproperData::members_t& improper = m_improper_data->getMembersByIndex(i);
        assert(improper.tag[0] <= m_pdata->getMaximumTag());
        assert(improper.tag[1] <= m_pdata->getMaximumTag());
        assert(improper.tag[2] <= m_pdata->getMaximumTag());
        assert(improper.tag[3] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        unsigned int idx_a = h_rtag.data[improper.tag[0]];
        unsigned int idx_b = h_rtag.data[improper.tag[1]];
        unsigned int idx_c = h_rtag.data[improper.tag[2]];
        unsigned int idx_d = h_rtag.data[improper.tag[3]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            std::ostringstream s;
            s << "improper " << improper.tag[0] << " " << improper.tag[1] << " " << improper.tag[2]
              << " " << improper.tag[3] << " is incomplete." << endl
              << endl;
            throw std::runtime_error(s.str());
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x;
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y;
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;

        dcbm = box.minImage(dcbm);

        Scalar aax = dab.y * dcbm.z - dab.z * dcbm.y;
        Scalar aay = dab.z * dcbm.x - dab.x * dcbm.z;
        Scalar aaz = dab.x * dcbm.y - dab.y * dcbm.x;

        Scalar bbx = ddc.y * dcbm.z - ddc.z * dcbm.y;
        Scalar bby = ddc.z * dcbm.x - ddc.x * dcbm.z;
        Scalar bbz = ddc.x * dcbm.y - ddc.y * dcbm.x;

        Scalar raasq = aax * aax + aay * aay + aaz * aaz;
        Scalar rbbsq = bbx * bbx + bby * bby + bbz * bbz;
        Scalar rgsq = dcbm.x * dcbm.x + dcbm.y * dcbm.y + dcbm.z * dcbm.z;
        Scalar rg = sqrt(rgsq);

        Scalar rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = Scalar(0.0);
        if (rg > Scalar(0.0))
            rginv = Scalar(1.0) / rg;
        if (raasq > Scalar(0.0))
            raa2inv = Scalar(1.0) / raasq;
        if (rbbsq > Scalar(0.0))
            rbb2inv = Scalar(1.0) / rbbsq;
        Scalar rabinv = sqrt(raa2inv * rbb2inv);

        Scalar c_abcd = (aax * bbx + aay * bby + aaz * bbz) * rabinv;
        Scalar s_abcd = rg * rabinv * (aax * ddc.x + aay * ddc.y + aaz * ddc.z);

        if (c_abcd > 1.0)
            c_abcd = 1.0;
        if (c_abcd < -1.0)
            c_abcd = -1.0;

        unsigned int improper_type = m_improper_data->getTypeByIndex(i);
        const periodic_improper_params& param = h_params.data[improper_type];
        int n = param.n;
        Scalar p = Scalar(1.0);
        Scalar dfab = Scalar(0.0);
        Scalar ddfab = Scalar(0.0);

        for (int j = 0; j < n; j++)
            {
            ddfab = p * c_abcd - dfab * s_abcd;
            dfab = p * s_abcd + dfab * c_abcd;
            p = ddfab;
            }

        /////////////////////////
        // FROM LAMMPS: sin_shift is always 0... so dropping all sin_shift terms!!!!
        // Adding charmm improper functionality, sin_shift not always 0,
        // cos_shift not always 1
        /////////////////////////

        Scalar d = param.d;
        Scalar chi_0 = param.chi_0;
        Scalar sin_chi_0 = fast::sin(chi_0);
        Scalar cos_chi_0 = fast::cos(chi_0);
        p = p * cos_chi_0 + dfab * sin_chi_0;
        p = p * d;
        dfab = dfab * cos_chi_0 - ddfab * sin_chi_0;
        dfab = dfab * d;
        dfab *= (Scalar)-n;
        p += Scalar(1.0);

        if (n == 0)
            {
            p = Scalar(1.0) + d;
            dfab = Scalar(0.0);
            }

        Scalar fg = dab.x * dcbm.x + dab.y * dcbm.y + dab.z * dcbm.z;
        Scalar hg = ddc.x * dcbm.x + ddc.y * dcbm.y + ddc.z * dcbm.z;

        Scalar fga = fg * raa2inv * rginv;
        Scalar hgb = hg * rbb2inv * rginv;
        Scalar gaa = -raa2inv * rg;
        Scalar gbb = rbb2inv * rg;

        Scalar dtfx = gaa * aax;
        Scalar dtfy = gaa * aay;
        Scalar dtfz = gaa * aaz;
        Scalar dtgx = fga * aax - hgb * bbx;
        Scalar dtgy = fga * aay - hgb * bby;
        Scalar dtgz = fga * aaz - hgb * bbz;
        Scalar dthx = gbb * bbx;
        Scalar dthy = gbb * bby;
        Scalar dthz = gbb * bbz;

        //      Scalar df = -m_K[improper.type] * dfab;
        Scalar df = -param.k * dfab * Scalar(0.500); // the 0.5 term is for 1/2K in the forces

        Scalar sx2 = df * dtgx;
        Scalar sy2 = df * dtgy;
        Scalar sz2 = df * dtgz;

        Scalar ffax = df * dtfx;
        Scalar ffay = df * dtfy;
        Scalar ffaz = df * dtfz;

        Scalar ffbx = sx2 - ffax;
        Scalar ffby = sy2 - ffay;
        Scalar ffbz = sz2 - ffaz;

        Scalar ffdx = df * dthx;
        Scalar ffdy = df * dthy;
        Scalar ffdz = df * dthz;

        Scalar ffcx = -sx2 - ffdx;
        Scalar ffcy = -sy2 - ffdy;
        Scalar ffcz = -sz2 - ffdz;

        // Now, apply the force to each individual atom a,b,c,d
        // and accumulate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the improper
        // Scalar improper_eng = p*m_K[improper.type]*Scalar(1.0/4.0);
        Scalar improper_eng = p * param.k * Scalar(0.125); // the .125 term is (1/2)K * 1/4

        // compute 1/4 of the virial, 1/4 for each atom in the improper
        // upper triangular version of virial tensor
        Scalar improper_virial[6];
        improper_virial[0] = (1. / 4.) * (dab.x * ffax + dcb.x * ffcx + (ddc.x + dcb.x) * ffdx);
        improper_virial[1] = (1. / 4.) * (dab.y * ffax + dcb.y * ffcx + (ddc.y + dcb.y) * ffdx);
        improper_virial[2] = (1. / 4.) * (dab.z * ffax + dcb.z * ffcx + (ddc.z + dcb.z) * ffdx);
        improper_virial[3] = (1. / 4.) * (dab.y * ffay + dcb.y * ffcy + (ddc.y + dcb.y) * ffdy);
        improper_virial[4] = (1. / 4.) * (dab.z * ffay + dcb.z * ffcy + (ddc.z + dcb.z) * ffdy);
        improper_virial[5] = (1. / 4.) * (dab.z * ffaz + dcb.z * ffcz + (ddc.z + dcb.z) * ffdz);

        h_force.data[idx_a].x += ffax;
        h_force.data[idx_a].y += ffay;
        h_force.data[idx_a].z += ffaz;
        h_force.data[idx_a].w += improper_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_a] += improper_virial[k];

        h_force.data[idx_b].x += ffbx;
        h_force.data[idx_b].y += ffby;
        h_force.data[idx_b].z += ffbz;
        h_force.data[idx_b].w += improper_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_b] += improper_virial[k];

        h_force.data[idx_c].x += ffcx;
        h_force.data[idx_c].y += ffcy;
        h_force.data[idx_c].z += ffcz;
        h_force.data[idx_c].w += improper_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_c] += improper_virial[k];

        h_force.data[idx_d].x += ffdx;
        h_force.data[idx_d].y += ffdy;
        h_force.data[idx_d].z += ffdz;
        h_force.data[idx_d].w += improper_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_d] += improper_virial[k];
        }
    }

namespace detail
    {
void export_PeriodicImproperForceCompute(pybind11::module& m)
    {
    pybind11::class_<PeriodicImproperForceCompute,
                     ForceCompute,
                     std::shared_ptr<PeriodicImproperForceCompute>>(m,
                                                                    "PeriodicImproperForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PeriodicImproperForceCompute::setParamsPython)
        .def("getParams", &PeriodicImproperForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
