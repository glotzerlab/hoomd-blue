// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichGeneralMeshForceCompute.h"
#include "BendingRigidityMeshForceCompute.h"

#include <iostream>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <memory>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)


/*! \file HelfrichGeneralMeshForceCompute.cc
    \brief Contains code for the HelfrichGeneralMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
HelfrichGeneralMeshForceCompute::HelfrichGeneralMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing HelfrichGeneralMeshForceCompute" << endl;

    // allocate the parameters
    GPUArray<helfrich_param_t> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_sigma_dash(m_pdata->getMaxN(), m_exec_conf);

    m_sigma_dash.swap(tmp_sigma_dash);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_normal(m_pdata->getMaxN(), m_exec_conf);

    m_normal.swap(tmp_normal);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar> tmp_sigma(m_pdata->getMaxN(), m_exec_conf);

    m_sigma.swap(tmp_sigma);
    }

HelfrichGeneralMeshForceCompute::~HelfrichGeneralMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying HelfrichGeneralMeshForceCompute" << endl;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void HelfrichGeneralMeshForceCompute::setParams(unsigned int type, const helfrich_param_t& params)
    {
    ArrayHandle<helfrich_param_t> h_params(m_params,
                                                             access_location::host,
                                                             access_mode::readwrite);
    h_params.data[type] = params;

    if (params.k <= 0)
        m_exec_conf->msg->warning() << "Helfrich: specified K <= 0" << endl;
    }

void HelfrichGeneralMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type);
    setParams(typ, helfrich_param_t(params));
    }

pybind11::dict HelfrichGeneralMeshForceCompute::getParams(std::string type)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type);
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.helfrich: Invalid particle type specified" << endl;
        throw runtime_error("Error setting parameters in HelfrichGeneralMeshForceCompute");
        }
    ArrayHandle<helfrich_param_t> h_params(m_params,
                                                             access_location::host,
                                                             access_mode::read);
    return h_params.data[typ].asDict();
    }


pybind11::object HelfrichGeneralMeshForceCompute::getCurvaturesPython()
    {
    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(1);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        }
    else
        {
        dims[0] = 0;
        }
    std::vector<double> global_curvature(dims[0]);

    // sort energies by particle tag
    sortLocalTags();
    std::vector<double> local_curvature;
    local_curvature.reserve(m_pdata->getN());
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_normal(m_normal, access_location::host, access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
	unsigned int idx = h_rtag.data[m_local_tag[i]];
        Scalar3 sigma_dash = h_sigma_dash.data[idx];

        Scalar sigma = h_sigma.data[idx];
					      
	Scalar sigma_dash2 = dot(sigma_dash,sigma_dash);

	Scalar factor =  h_normal.data[idx].x;
        local_curvature.push_back(factor*sqrt(sigma_dash2)/sigma);
        }

    if (m_sysdef->isDomainDecomposed())
        {
#ifdef ENABLE_MPI
        m_gather_tag_order.setLocalTagsSorted(m_local_tag);
        m_gather_tag_order.gatherArray(global_curvature, local_curvature);
#endif
        }
    else
        {
        global_curvature = std::move(local_curvature);
        }

    if (root)
        {
	pybind11::array_t<double> arr(global_curvature.size());
	std::memcpy(arr.mutable_data(), global_curvature.data(),
            global_curvature.size() * sizeof(double));
        return arr;
        }
    return pybind11::none();
    }



/*! Actually perform the force computation
    \param timestep Current time step
 */
void HelfrichGeneralMeshForceCompute::computeForces(uint64_t timestep)
    {
    precomputeParameter();

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();
    ArrayHandle<helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_normal(m_normal, access_location::host, access_mode::read);

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);
    assert(h_sigma.data);
    assert(h_sigma_dash.data);
    assert(h_normal.data);

    m_force.zeroFill();
    m_virial.zeroFill();

    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar helfrich_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        helfrich_virial[i] = Scalar(0.0);

    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);
        unsigned int btag_d = bond.tag[3];
        assert(btag_d < m_pdata->getMaximumTag() + 1);

        if (btag_c == btag_d)
            continue;

        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        Scalar3 dad;
        dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
        dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
        dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

        Scalar3 dbc;
        dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
        dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
        dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

        Scalar3 dbd;
        dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
        dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
        dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

	unsigned int type_a = __scalar_as_int(h_pos.data[idx_a].w);
	unsigned int type_b = __scalar_as_int(h_pos.data[idx_b].w);
	unsigned int type_c = __scalar_as_int(h_pos.data[idx_c].w);
	unsigned int type_d = __scalar_as_int(h_pos.data[idx_d].w);

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);
        Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
        Scalar rad = sqrt(rsqad);

        Scalar rsqbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
        Scalar rbc = sqrt(rsqbc);
        Scalar rsqbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
        Scalar rbd = sqrt(rsqbd);

        Scalar3 nab, nac, nad, nbc, nbd;
        nab = dab / rab;
        nac = dac / rac;
        nad = dad / rad;
        nbc = dbc / rbc;
        nbd = dbd / rbd;

        Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
        if (c_accb > 1.0)
            c_accb = 1.0;
        if (c_accb < -1.0)
            c_accb = -1.0;

        Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
        if (c_addb > 1.0)
            c_addb = 1.0;
        if (c_addb < -1.0)
            c_addb = -1.0;

        Scalar c_abbc = -nab.x * nbc.x - nab.y * nbc.y - nab.z * nbc.z;
        if (c_abbc > 1.0)
            c_abbc = 1.0;
        if (c_abbc < -1.0)
            c_abbc = -1.0;

        Scalar c_abbd = -nab.x * nbd.x - nab.y * nbd.y - nab.z * nbd.z;
        if (c_abbd > 1.0)
            c_abbd = 1.0;
        if (c_abbd < -1.0)
            c_abbd = -1.0;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;
        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar c_baad = nab.x * nad.x + nab.y * nad.y + nab.z * nad.z;
        if (c_baad > 1.0)
            c_baad = 1.0;
        if (c_baad < -1.0)
            c_baad = -1.0;

        Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
        if (inv_s_accb < SMALL)
            inv_s_accb = SMALL;
        inv_s_accb = 1.0 / inv_s_accb;

        Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
        if (inv_s_addb < SMALL)
            inv_s_addb = SMALL;
        inv_s_addb = 1.0 / inv_s_addb;

        Scalar inv_s_abbc = sqrt(1.0 - c_abbc * c_abbc);
        if (inv_s_abbc < SMALL)
            inv_s_abbc = SMALL;
        inv_s_abbc = 1.0 / inv_s_abbc;

        Scalar inv_s_abbd = sqrt(1.0 - c_abbd * c_abbd);
        if (inv_s_abbd < SMALL)
            inv_s_abbd = SMALL;
        inv_s_abbd = 1.0 / inv_s_abbd;

        Scalar inv_s_baac = sqrt(1.0 - c_baac * c_baac);
        if (inv_s_baac < SMALL)
            inv_s_baac = SMALL;
        inv_s_baac = 1.0 / inv_s_baac;

        Scalar inv_s_baad = sqrt(1.0 - c_baad * c_baad);
        if (inv_s_baad < SMALL)
            inv_s_baad = SMALL;
        inv_s_baad = 1.0 / inv_s_baad;

        Scalar cot_accb = c_accb * inv_s_accb;
        Scalar cot_addb = c_addb * inv_s_addb;

        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;

        Scalar3 sigma_dash_a = h_sigma_dash.data[idx_a]; // precomputed
        Scalar3 sigma_dash_b = h_sigma_dash.data[idx_b]; // precomputed
        Scalar3 sigma_dash_c = h_sigma_dash.data[idx_c]; // precomputed
        Scalar3 sigma_dash_d = h_sigma_dash.data[idx_d]; // precomputed

        Scalar sigma_a = h_sigma.data[idx_a]; // precomputed
        Scalar sigma_b = h_sigma.data[idx_b]; // precomputed
        Scalar sigma_c = h_sigma.data[idx_c]; // precomputed
        Scalar sigma_d = h_sigma.data[idx_d]; // precomputed
					      //
	Scalar sigma_dash_a2 = dot(sigma_dash_a,sigma_dash_a);
	Scalar sigma_dash_b2 = dot(sigma_dash_b,sigma_dash_b);
	Scalar sigma_dash_c2 = dot(sigma_dash_c,sigma_dash_c);
	Scalar sigma_dash_d2 = dot(sigma_dash_d,sigma_dash_d);

	Scalar H0_a = h_params.data[type_a].H0;
	Scalar H0_b = h_params.data[type_b].H0;
	Scalar H0_c = h_params.data[type_c].H0;
	Scalar H0_d = h_params.data[type_d].H0;

	Scalar factor_a =  h_normal.data[idx_a].x;
	Scalar factor_b =  h_normal.data[idx_b].x;
	Scalar factor_c =  h_normal.data[idx_c].x;
	Scalar factor_d =  h_normal.data[idx_d].x;

	Scalar Curv_a = factor_a*sqrt(sigma_dash_a2)-H0_a*sigma_a;
	Scalar Curv_b = factor_b*sqrt(sigma_dash_b2)-H0_b*sigma_b;
	Scalar Curv_c = factor_c*sqrt(sigma_dash_c2)-H0_c*sigma_c;
	Scalar Curv_d = factor_d*sqrt(sigma_dash_d2)-H0_d*sigma_d;
					      //
        Scalar3 dc_abbc, dc_abbd, dc_baac, dc_baad;
        dc_abbc = -nbc / rab - c_abbc / rab * nab;
        dc_abbd = -nbd / rab - c_abbd / rab * nab;
        dc_baac = nac / rab - c_baac / rab * nab;
        dc_baad = nad / rab - c_baad / rab * nab;

        Scalar3 dsigma_hat_ac, dsigma_hat_ad, dsigma_hat_bc, dsigma_hat_bd;
        dsigma_hat_ac = inv_s_abbc * inv_s_abbc * inv_s_abbc * dc_abbc / 2;
        dsigma_hat_ad = inv_s_abbd * inv_s_abbd * inv_s_abbd * dc_abbd / 2;
        dsigma_hat_bc = inv_s_baac * inv_s_baac * inv_s_baac * dc_baac / 2;
        dsigma_hat_bd = inv_s_baad * inv_s_baad * inv_s_baad * dc_baad / 2;

        Scalar3 dsigma_a, dsigma_b, dsigma_c, dsigma_d;
        dsigma_a = (dsigma_hat_ac * rsqac + dsigma_hat_ad * rsqad + 2 * sigma_hat_ab * dab) / 4;
        dsigma_b = (dsigma_hat_bc * rsqbc + dsigma_hat_bd * rsqbd + 2 * sigma_hat_ab * dab) / 4;
        dsigma_c = (dsigma_hat_ac * rsqac + dsigma_hat_bc * rsqbc) / 4;
        dsigma_d = (dsigma_hat_ad * rsqad + dsigma_hat_bd * rsqbd) / 4;

        Scalar dsigma_dash_a = dot(dsigma_hat_ac, dac) + dot(dsigma_hat_ad, dad) + sigma_hat_ab;
        Scalar dsigma_dash_b = dot(dsigma_hat_bc, dbc) + dot(dsigma_hat_bd, dbd) - sigma_hat_ab;
        Scalar dsigma_dash_c = -dot(dsigma_hat_ac, dac) - dot(dsigma_hat_bc, dbc);
        Scalar dsigma_dash_d = -dot(dsigma_hat_ad, dad) - dot(dsigma_hat_bd, dbd);

	Scalar3 dCurv_a, dCurv_b, dCurv_c, dCurv_d;

	dCurv_a = factor_a*dsigma_dash_a/sqrt(sigma_dash_a2)*sigma_dash_a - H0_a*dsigma_a;
	dCurv_b = factor_b*dsigma_dash_b/sqrt(sigma_dash_b2)*sigma_dash_b - H0_b*dsigma_b;
	dCurv_c = factor_c*dsigma_dash_c/sqrt(sigma_dash_c2)*sigma_dash_c - H0_c*dsigma_c;
	dCurv_d = factor_d*dsigma_dash_d/sqrt(sigma_dash_d2)*sigma_dash_d - H0_d*dsigma_d;

        Scalar inv_sigma_a = 1.0 / sigma_a;
        Scalar inv_sigma_b = 1.0 / sigma_b;
        Scalar inv_sigma_c = 1.0 / sigma_c;
        Scalar inv_sigma_d = 1.0 / sigma_d;

        Scalar Curv_a2 = 0.5 * Curv_a * Curv_a * inv_sigma_a * inv_sigma_a;
        Scalar Curv_b2 = 0.5 * Curv_b * Curv_b * inv_sigma_b * inv_sigma_b;
        Scalar Curv_c2 = 0.5 * Curv_c * Curv_c * inv_sigma_c * inv_sigma_c;
        Scalar Curv_d2 = 0.5 * Curv_d * Curv_d * inv_sigma_d * inv_sigma_d;

	Scalar K_a = h_params.data[type_a].k;
	Scalar K_b = h_params.data[type_b].k;
	Scalar K_c = h_params.data[type_c].k;
	Scalar K_d = h_params.data[type_d].k;

        Scalar3 Fa;

        Fa.x =  K_a*(Curv_a * inv_sigma_a * dCurv_a.x - Curv_a2 * dsigma_a.x);
        Fa.x += K_b*(Curv_b * inv_sigma_b * dCurv_b.x - Curv_b2 * dsigma_b.x);
        Fa.x += K_c*(Curv_c * inv_sigma_c * dCurv_c.x - Curv_c2 * dsigma_c.x);
        Fa.x += K_d*(Curv_d * inv_sigma_d * dCurv_d.x - Curv_d2 * dsigma_d.x);

        Fa.y =  K_a*(Curv_a * inv_sigma_a * dCurv_a.y - Curv_a2 * dsigma_a.y);
        Fa.y += K_b*(Curv_b * inv_sigma_b * dCurv_b.y - Curv_b2 * dsigma_b.y);
        Fa.y += K_c*(Curv_c * inv_sigma_c * dCurv_c.y - Curv_c2 * dsigma_c.y);
        Fa.y += K_d*(Curv_d * inv_sigma_d * dCurv_d.y - Curv_d2 * dsigma_d.y);

        Fa.z =  K_a*(Curv_a * inv_sigma_a * dCurv_a.z - Curv_a2 * dsigma_a.z);
        Fa.z += K_b*(Curv_b * inv_sigma_b * dCurv_b.z - Curv_b2 * dsigma_b.z);
        Fa.z += K_c*(Curv_c * inv_sigma_c * dCurv_c.z - Curv_c2 * dsigma_c.z);
        Fa.z += K_d*(Curv_d * inv_sigma_d * dCurv_d.z - Curv_d2 * dsigma_d.z);


        //Fa *= h_params.data[type_a].k;
        if (compute_virial)
            {
            helfrich_virial[0] = Scalar(1. / 2.) * dab.x * Fa.x; // xx
            helfrich_virial[1] = Scalar(1. / 2.) * dab.y * Fa.x; // xy
            helfrich_virial[2] = Scalar(1. / 2.) * dab.z * Fa.x; // xz
            helfrich_virial[3] = Scalar(1. / 2.) * dab.y * Fa.y; // yy
            helfrich_virial[4] = Scalar(1. / 2.) * dab.z * Fa.y; // yz
            helfrich_virial[5] = Scalar(1. / 2.) * dab.z * Fa.z; // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
	    
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w = K_a * 0.5
                                     * Curv_a * Curv_a * inv_sigma_a;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fa.x;
            h_force.data[idx_b].y -= Fa.y;
            h_force.data[idx_b].z -= Fa.z;
            h_force.data[idx_b].w = K_b * 0.5
                                     * Curv_b * Curv_b * inv_sigma_b;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }
        }
    }

void HelfrichGeneralMeshForceCompute::precomputeParameter()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_normal(m_normal, access_location::host, access_mode::overwrite);

    m_sigma.zeroFill();
    m_sigma_dash.zeroFill();

    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);
        unsigned int btag_d = bond.tag[3];
        assert(btag_d < m_pdata->getMaximumTag() + 1);

        if (btag_c == btag_d)
            continue;

        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        Scalar3 dad;
        dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
        dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
        dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

        Scalar3 dbc;
        dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
        dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
        dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

        Scalar3 dbd;
        dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
        dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
        dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        rac = sqrt(rac);
        Scalar rad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
        rad = sqrt(rad);

        Scalar rbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
        rbc = sqrt(rbc);
        Scalar rbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
        rbd = sqrt(rbd);

        Scalar3 nab, nac, nad, nbc, nbd;
        nab = dab / rab;
        nac = dac / rac;
        nad = dad / rad;
        nbc = dbc / rbc;
        nbd = dbd / rbd;

	Scalar3 normal_abcd;

	normal_abcd.x = nac.y*nab.z - nac.z*nab.y + nab.y*nad.z - nab.z*nad.y;
	normal_abcd.y = nac.z*nab.x - nac.x*nab.z + nab.z*nad.x - nab.x*nad.z;
	normal_abcd.z = nac.x*nab.y - nac.y*nab.x + nab.x*nad.y - nab.y*nad.x;

        Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
        if (c_accb > 1.0)
            c_accb = 1.0;
        if (c_accb < -1.0)
            c_accb = -1.0;

        Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
        if (c_addb > 1.0)
            c_addb = 1.0;
        if (c_addb < -1.0)
            c_addb = -1.0;

        Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
        if (inv_s_accb < SMALL)
            inv_s_accb = SMALL;
        inv_s_accb = 1.0 / inv_s_accb;

        Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
        if (inv_s_addb < SMALL)
            inv_s_addb = SMALL;
        inv_s_addb = 1.0 / inv_s_addb;

        Scalar cot_accb = c_accb * inv_s_accb;
        Scalar cot_addb = c_addb * inv_s_addb;

        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;

        Scalar sigma_a = sigma_hat_ab * rsqab * 0.25;

        if (idx_a < m_pdata->getN())
            {
            h_sigma.data[idx_a] += sigma_a;
            h_normal.data[idx_a].x += normal_abcd.x;
            h_normal.data[idx_a].y += normal_abcd.y;
            h_normal.data[idx_a].z += normal_abcd.z;
            h_sigma_dash.data[idx_a].x += sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_a].y += sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_a].z += sigma_hat_ab * dab.z;
            }

        if (idx_b < m_pdata->getN())
            {
            h_sigma.data[idx_b] += sigma_a;
            h_normal.data[idx_b].x += normal_abcd.x;
            h_normal.data[idx_b].y += normal_abcd.y;
            h_normal.data[idx_b].z += normal_abcd.z;
            h_sigma_dash.data[idx_b].x -= sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_b].y -= sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_b].z -= sigma_hat_ab * dab.z;
            }
        }

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
		if( dot(h_normal.data[i], h_sigma_dash.data[i]) < 0)
			h_normal.data[i].x = -1;
		else
			h_normal.data[i].x = 1;
	}
    }

Scalar HelfrichGeneralMeshForceCompute::energyDiff(unsigned int idx_a,
                                            unsigned int idx_b,
                                            unsigned int idx_c,
                                            unsigned int idx_d,
                                            unsigned int type_id)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_normal(m_normal, access_location::host, access_mode::read);

    ArrayHandle<helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();

    // calculate d\vec{r}
    Scalar3 dab;
    dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
    dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
    dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

    Scalar3 dac;
    dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
    dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
    dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

    Scalar3 dad;
    dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
    dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
    dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

    Scalar3 dbc;
    dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
    dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
    dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

    Scalar3 dbd;
    dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
    dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
    dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

    Scalar3 dcd;
    dcd.x = h_pos.data[idx_c].x - h_pos.data[idx_d].x;
    dcd.y = h_pos.data[idx_c].y - h_pos.data[idx_d].y;
    dcd.z = h_pos.data[idx_c].z - h_pos.data[idx_d].z;

    unsigned int type_a = __scalar_as_int(h_pos.data[idx_a].w);
    unsigned int type_b = __scalar_as_int(h_pos.data[idx_b].w);
    unsigned int type_c = __scalar_as_int(h_pos.data[idx_c].w);
    unsigned int type_d = __scalar_as_int(h_pos.data[idx_d].w);

    // apply minimum image conventions to all 3 vectors
    dab = box.minImage(dab);
    dac = box.minImage(dac);
    dad = box.minImage(dad);
    dbc = box.minImage(dbc);
    dbd = box.minImage(dbd);
    dcd = box.minImage(dcd);

    Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
    Scalar rab = sqrt(rsqab);
    Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
    Scalar rac = sqrt(rsqac);
    Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
    Scalar rad = sqrt(rsqad);

    Scalar rsqbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
    Scalar rbc = sqrt(rsqbc);
    Scalar rsqbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
    Scalar rbd = sqrt(rsqbd);
    Scalar rsqcd = dcd.x * dcd.x + dcd.y * dcd.y + dcd.z * dcd.z;
    Scalar rcd = sqrt(rsqcd);

    Scalar3 nab, nac, nad, nbc, nbd, ncd;
    nab = dab / rab;
    nac = dac / rac;
    nad = dad / rad;
    nbc = dbc / rbc;
    nbd = dbd / rbd;
    ncd = dcd / rcd;

    Scalar3 normal_cdb, normal_cda;
    normal_cdb.x = ncd.y*nbc.z - ncd.z*nbc.y;
    normal_cdb.y = ncd.z*nbc.x - ncd.x*nbc.z;
    normal_cdb.z = ncd.x*nbc.y - ncd.y*nbc.x;

    normal_cda.x = nac.y*ncd.z - nac.z*ncd.y;
    normal_cda.y = nac.z*ncd.x - nac.x*ncd.z;
    normal_cda.z = nac.x*ncd.y - nac.y*ncd.x;

    Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
    if (c_accb > 1.0)
        c_accb = 1.0;
    if (c_accb < -1.0)
        c_accb = -1.0;

    Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
    if (inv_s_accb < SMALL)
        inv_s_accb = SMALL;
    inv_s_accb = 1.0 / inv_s_accb;

    Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
    if (c_addb > 1.0)
        c_addb = 1.0;
    if (c_addb < -1.0)
        c_addb = -1.0;

    Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
    if (inv_s_addb < SMALL)
        inv_s_addb = SMALL;
    inv_s_addb = 1.0 / inv_s_addb;

    Scalar c_abbc = -(nab.x * nbc.x + nab.y * nbc.y + nab.z * nbc.z);
    if (c_abbc > 1.0)
        c_abbc = 1.0;
    if (c_abbc < -1.0)
        c_abbc = -1.0;

    Scalar inv_s_abbc = sqrt(1.0 - c_abbc * c_abbc);
    if (inv_s_abbc < SMALL)
        inv_s_abbc = SMALL;
    inv_s_abbc = 1.0 / inv_s_abbc;

    Scalar c_abbd = -(nab.x * nbd.x + nab.y * nbd.y + nab.z * nbd.z);
    if (c_abbd > 1.0)
        c_abbd = 1.0;
    if (c_abbd < -1.0)
        c_abbd = -1.0;

    Scalar inv_s_abbd = sqrt(1.0 - c_abbd * c_abbd);
    if (inv_s_abbd < SMALL)
        inv_s_abbd = SMALL;
    inv_s_abbd = 1.0 / inv_s_abbd;

    Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;
    if (c_baac > 1.0)
        c_baac = 1.0;
    if (c_baac < -1.0)
        c_baac = -1.0;

    Scalar inv_s_baac = sqrt(1.0 - c_baac * c_baac);
    if (inv_s_baac < SMALL)
        inv_s_baac = SMALL;
    inv_s_baac = 1.0 / inv_s_baac;

    Scalar c_baad = nab.x * nad.x + nab.y * nad.y + nab.z * nad.z;
    if (c_baad > 1.0)
        c_baad = 1.0;
    if (c_baad < -1.0)
        c_baad = -1.0;

    Scalar inv_s_baad = sqrt(1.0 - c_baad * c_baad);
    if (inv_s_baad < SMALL)
        inv_s_baad = SMALL;
    inv_s_baad = 1.0 / inv_s_baad;

    Scalar c_caad = nac.x * nad.x + nac.y * nad.y + nac.z * nad.z;
    if (c_caad > 1.0)
        c_caad = 1.0;
    if (c_caad < -1.0)
        c_caad = -1.0;

    Scalar inv_s_caad = sqrt(1.0 - c_caad * c_caad);
    if (inv_s_caad < SMALL)
        inv_s_caad = SMALL;
    inv_s_caad = 1.0 / inv_s_caad;

    Scalar c_cbbd = nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z;
    if (c_cbbd > 1.0)
        c_cbbd = 1.0;
    if (c_cbbd < -1.0)
        c_cbbd = -1.0;

    Scalar inv_s_cbbd = sqrt(1.0 - c_cbbd * c_cbbd);
    if (inv_s_cbbd < SMALL)
        inv_s_cbbd = SMALL;
    inv_s_cbbd = 1.0 / inv_s_cbbd;

    Scalar c_accd = -(nac.x * ncd.x + nac.y * ncd.y + nac.z * ncd.z);
    if (c_accd > 1.0)
        c_accd = 1.0;
    if (c_accd < -1.0)
        c_accd = -1.0;

    Scalar inv_s_accd = sqrt(1.0 - c_accd * c_accd);
    if (inv_s_accd < SMALL)
        inv_s_accd = SMALL;
    inv_s_accd = 1.0 / inv_s_accd;

    Scalar c_addc = nad.x * ncd.x + nad.y * ncd.y + nad.z * ncd.z;
    if (c_addc > 1.0)
        c_addc = 1.0;
    if (c_addc < -1.0)
        c_addc = -1.0;

    Scalar inv_s_addc = sqrt(1.0 - c_addc * c_addc);
    if (inv_s_addc < SMALL)
        inv_s_addc = SMALL;
    inv_s_addc = 1.0 / inv_s_addc;

    Scalar c_bccd = -(nbc.x * ncd.x + nbc.y * ncd.y + nbc.z * ncd.z);
    if (c_bccd > 1.0)
        c_bccd = 1.0;
    if (c_bccd < -1.0)
        c_bccd = -1.0;

    Scalar inv_s_bccd = sqrt(1.0 - c_bccd * c_bccd);
    if (inv_s_bccd < SMALL)
        inv_s_bccd = SMALL;
    inv_s_bccd = 1.0 / inv_s_bccd;

    Scalar c_bddc = nbd.x * ncd.x + nbd.y * ncd.y + nbd.z * ncd.z;
    if (c_bddc > 1.0)
        c_bddc = 1.0;
    if (c_bddc < -1.0)
        c_bddc = -1.0;

    Scalar inv_s_bddc = sqrt(1.0 - c_bddc * c_bddc);
    if (inv_s_bddc < SMALL)
        inv_s_bddc = SMALL;
    inv_s_bddc = 1.0 / inv_s_bddc;

    Scalar cot_accb = c_accb * inv_s_accb;
    Scalar cot_addb = c_addb * inv_s_addb;
    Scalar cot_baac = c_baac * inv_s_baac;
    Scalar cot_baad = c_baad * inv_s_baad;
    Scalar cot_abbc = c_abbc * inv_s_abbc;
    Scalar cot_abbd = c_abbd * inv_s_abbd;

    Scalar cot_caad = c_caad * inv_s_caad;
    Scalar cot_cbbd = c_cbbd * inv_s_cbbd;
    Scalar cot_accd = c_accd * inv_s_accd;
    Scalar cot_addc = c_addc * inv_s_addc;
    Scalar cot_bccd = c_bccd * inv_s_bccd;
    Scalar cot_bddc = c_bddc * inv_s_bddc;

    Scalar sigma_hat_ab = -(cot_accb + cot_addb) * 0.5;
    Scalar sigma_hat_cd = (cot_caad + cot_cbbd) * 0.5;
    Scalar sigma_hat_ac = (cot_addc - cot_abbc) * 0.5;
    Scalar sigma_hat_ad = (cot_accd - cot_abbd) * 0.5;
    Scalar sigma_hat_bc = (cot_bddc - cot_baac) * 0.5;
    Scalar sigma_hat_bd = (cot_bccd - cot_baad) * 0.5;

    Scalar sigma_a = h_sigma.data[idx_a]; // precomputed
    Scalar sigma_b = h_sigma.data[idx_b]; // precomputed
    Scalar sigma_c = h_sigma.data[idx_c]; // precomputed
    Scalar sigma_d = h_sigma.data[idx_d]; // precomputed

    m_sigma_diff_a = (sigma_hat_ab * rsqab + sigma_hat_ac * rsqac + sigma_hat_ad * rsqad) * 0.25;
    m_sigma_diff_b = (sigma_hat_ab * rsqab + sigma_hat_bc * rsqbc + sigma_hat_bd * rsqbd) * 0.25;
    m_sigma_diff_c = (sigma_hat_ac * rsqac + sigma_hat_bc * rsqbc + sigma_hat_cd * rsqcd) * 0.25;
    m_sigma_diff_d = (sigma_hat_ad * rsqad + sigma_hat_bd * rsqbd + sigma_hat_cd * rsqcd) * 0.25;

    Scalar sigma_a_n = sigma_a + m_sigma_diff_a;
    Scalar sigma_b_n = sigma_b + m_sigma_diff_b;
    Scalar sigma_c_n = sigma_c + m_sigma_diff_c;
    Scalar sigma_d_n = sigma_d + m_sigma_diff_d;

    Scalar3 sigma_dash_a = h_sigma_dash.data[idx_a]; // precomputed
    Scalar3 sigma_dash_b = h_sigma_dash.data[idx_b]; // precomputed
    Scalar3 sigma_dash_c = h_sigma_dash.data[idx_c]; // precomputed
    Scalar3 sigma_dash_d = h_sigma_dash.data[idx_d]; // precomputed

    Scalar sigma_dash_a2 = dot(sigma_dash_a,sigma_dash_a);
    Scalar sigma_dash_b2 = dot(sigma_dash_b,sigma_dash_b);
    Scalar sigma_dash_c2 = dot(sigma_dash_c,sigma_dash_c);
    Scalar sigma_dash_d2 = dot(sigma_dash_d,sigma_dash_d);
    
    Scalar H0_a = h_params.data[type_a].H0;
    Scalar H0_b = h_params.data[type_b].H0;
    Scalar H0_c = h_params.data[type_c].H0;
    Scalar H0_d = h_params.data[type_d].H0;

    Scalar K_a = h_params.data[type_a].k;
    Scalar K_b = h_params.data[type_b].k;
    Scalar K_c = h_params.data[type_c].k;
    Scalar K_d = h_params.data[type_d].k;

    Scalar factor_a = h_normal.data[idx_a].x;
    Scalar factor_b = h_normal.data[idx_b].x;
    Scalar factor_c = h_normal.data[idx_c].x;
    Scalar factor_d = h_normal.data[idx_d].x;
    
    
    Scalar Curv_a = factor_a*sqrt(sigma_dash_a2)-H0_a*sigma_a;
    Scalar Curv_b = factor_b*sqrt(sigma_dash_b2)-H0_b*sigma_b;
    Scalar Curv_c = factor_c*sqrt(sigma_dash_c2)-H0_c*sigma_c;
    Scalar Curv_d = factor_d*sqrt(sigma_dash_d2)-H0_d*sigma_d;
    
    
    m_sigma_dash_diff_a = sigma_hat_ab * dab + sigma_hat_ac * dac + sigma_hat_ad * dad;
    m_sigma_dash_diff_b = -sigma_hat_ab * dab + sigma_hat_bc * dbc + sigma_hat_bd * dbd;
    m_sigma_dash_diff_c = -sigma_hat_ac * dac - sigma_hat_bc * dbc + sigma_hat_cd * dcd;
    m_sigma_dash_diff_d = -sigma_hat_ad * dad - sigma_hat_bd * dbd - sigma_hat_cd * dcd;

    Scalar3 sigma_dash_a_n = sigma_dash_a + m_sigma_dash_diff_a;
    Scalar3 sigma_dash_b_n = sigma_dash_b + m_sigma_dash_diff_b;
    Scalar3 sigma_dash_c_n = sigma_dash_c + m_sigma_dash_diff_c;
    Scalar3 sigma_dash_d_n = sigma_dash_d + m_sigma_dash_diff_d;

    Scalar sigma_dash_a2_n = dot(sigma_dash_a_n,sigma_dash_a_n);
    Scalar sigma_dash_b2_n = dot(sigma_dash_b_n,sigma_dash_b_n);
    Scalar sigma_dash_c2_n = dot(sigma_dash_c_n,sigma_dash_c_n);
    Scalar sigma_dash_d2_n = dot(sigma_dash_d_n,sigma_dash_d_n);

    m_factor_a = 1;
    m_factor_b = 1;
    m_factor_c = 1;
    m_factor_d = 1;
    
    if( dot(sigma_dash_a_n, normal_cda) < 0)
    	m_factor_a = -1;
    
    if( dot(sigma_dash_b_n, normal_cdb) < 0)
    	m_factor_b = -1;
    
    if( dot(sigma_dash_c_n, normal_cdb) < 0)
    	m_factor_c = -1;
    
    if( dot(sigma_dash_d_n, normal_cdb) < 0)
    	m_factor_d = -1;

    Scalar Curv_a_n = m_factor_a*sqrt(sigma_dash_a2_n)-H0_a*sigma_a_n;
    Scalar Curv_b_n = m_factor_b*sqrt(sigma_dash_b2_n)-H0_b*sigma_b_n;
    Scalar Curv_c_n = m_factor_c*sqrt(sigma_dash_c2_n)-H0_c*sigma_c_n;
    Scalar Curv_d_n = m_factor_d*sqrt(sigma_dash_d2_n)-H0_d*sigma_d_n;

    Scalar energy_old = K_a * Curv_a*Curv_a / sigma_a;
    energy_old += K_b * (Curv_b * Curv_b) / sigma_b;
    energy_old += K_c * (Curv_c * Curv_c) / sigma_c;
    energy_old += K_d * (Curv_d * Curv_d) / sigma_d;

    Scalar energy_new = K_a * Curv_a_n*Curv_a_n / sigma_a_n;
    energy_new += K_b * (Curv_b_n * Curv_b_n) / sigma_b_n;
    energy_new += K_c * (Curv_c_n * Curv_c_n) / sigma_c_n;
    energy_new += K_d * (Curv_d_n * Curv_d_n) / sigma_d_n;

    if (energy_new < 0)
        return DBL_MAX;

    return 0.5 * (energy_new - energy_old);
    }

void HelfrichGeneralMeshForceCompute::postcomputeParameter(unsigned int idx_a,
                                                    unsigned int idx_b,
                                                    unsigned int idx_c,
                                                    unsigned int idx_d,
                                                    unsigned int type_id)
    {
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_normal(m_normal, access_location::host, access_mode::readwrite);

    h_sigma.data[idx_a] += m_sigma_diff_a;
    h_sigma.data[idx_b] += m_sigma_diff_b;
    h_sigma.data[idx_c] += m_sigma_diff_c;
    h_sigma.data[idx_d] += m_sigma_diff_d;

    h_sigma_dash.data[idx_a] += m_sigma_dash_diff_a;
    h_sigma_dash.data[idx_b] += m_sigma_dash_diff_b;
    h_sigma_dash.data[idx_c] += m_sigma_dash_diff_c;
    h_sigma_dash.data[idx_d] += m_sigma_dash_diff_d;

    h_normal.data[idx_a].x = m_factor_a;
    h_normal.data[idx_b].x = m_factor_b;
    h_normal.data[idx_c].x = m_factor_c;
    h_normal.data[idx_d].x = m_factor_d;
    }

namespace detail
    {
void export_HelfrichGeneralMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<HelfrichGeneralMeshForceCompute,
                     MeshForceCompute,
                     std::shared_ptr<HelfrichGeneralMeshForceCompute>>(m, "HelfrichGeneralMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &HelfrichGeneralMeshForceCompute::setParamsPython)
        .def("getParams", &HelfrichGeneralMeshForceCompute::getParams)
        .def("getCurvatures", &HelfrichGeneralMeshForceCompute::getCurvaturesPython);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
