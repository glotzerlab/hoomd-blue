// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CurvatureHelfrichMeshForceCompute.h"
//#include "BendingRigidityMeshForceCompute.h"

#include <iostream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)


/*! \file CurvatureHelfrichMeshForceCompute.cc
    \brief Contains code for the CurvatureHelfrichMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
CurvatureHelfrichMeshForceCompute::CurvatureHelfrichMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing HelfrichMeshForceCompute" << endl;

    // allocate the parameters
    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    GPUArray<curvature_helfrich_param_t> params(n_types, m_exec_conf);
    m_params.swap(params);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_sigma_dash(m_pdata->getMaxN(), m_exec_conf);

    m_sigma_dash.swap(tmp_sigma_dash);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_norm(m_pdata->getMaxN(), m_exec_conf);

    m_norm.swap(tmp_norm);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar> tmp_sigma(m_pdata->getMaxN(), m_exec_conf);

    m_sigma.swap(tmp_sigma);
    }

CurvatureHelfrichMeshForceCompute::~CurvatureHelfrichMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CurvatureHelfrichMeshForceCompute" << endl;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void CurvatureHelfrichMeshForceCompute::setParams(unsigned int type,
		const curvature_helfrich_param_t& params)
    {
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = params;

    // check for some silly errors a user could make
    if (params.k <= 0)
        m_exec_conf->msg->warning() << "helfrich: specified K <= 0" << endl;
    }

void CurvatureHelfrichMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = curvature_helfrich_param_t(params);
    setParams(typ, _params);
    }

pybind11::dict CurvatureHelfrichMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.helfrich: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in CurvatureHelfrichMeshForceCompute");
        }
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);
    //pybind11::dict params;
    //params["k"] = h_params.data[typ];
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void CurvatureHelfrichMeshForceCompute::computeForces(uint64_t timestep)
    {
    precomputeParameter();

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::read);

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);
    assert(h_sigma.data);
    assert(h_sigma_dash.data);
    assert(h_norm.data);
    assert(h_triangles.data);

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

        unsigned int meshbond_type = m_mesh_data->getMeshBondData()->getTypeByIndex(i);

	Scalar C0_a = 0.0;
	Scalar C0_b = 0.0;
	Scalar C0_c = 0.0;
	Scalar C0_d = 0.0;
	Scalar eps_kT = 0.0;

	//if (h_tag.data[idx_a] > h_params.data[meshbond_type].tag_max)
	//{
	//	C0 = h_params.data[meshbond_type].C0;
	//}

	//if (h_tag.data[idx_b] > h_params.data[meshbond_type].tag_max)
	//{
	//	C0 = h_params.data[meshbond_type].C0;
	//}
	//
	if (h_tag.data[idx_a] > h_params.data[meshbond_type].tag_max)
	{
		C0_a = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_b] > h_params.data[meshbond_type].tag_max)
	{
		C0_b = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_c] > h_params.data[meshbond_type].tag_max)
	{
		C0_c = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_d] > h_params.data[meshbond_type].tag_max)
	{
		C0_d = h_params.data[meshbond_type].C0;
	}

	if ((h_tag.data[idx_a] > h_params.data[meshbond_type].tag_max) && (h_tag.data[idx_b] > h_params.data[meshbond_type].tag_max))
	{
		eps_kT = h_params.data[meshbond_type].eps_kT;
	}


        Scalar3 sigma_dash_a = h_sigma_dash.data[idx_a] ; // precomputed
        Scalar3 sigma_dash_b = h_sigma_dash.data[idx_b] ; // precomputed
        Scalar3 sigma_dash_c = h_sigma_dash.data[idx_c] ; // precomputed
        Scalar3 sigma_dash_d = h_sigma_dash.data[idx_d] ; // precomputed
							  //
        Scalar3 norm_a = h_norm.data[idx_a] ; // precomputed
        Scalar3 norm_b = h_norm.data[idx_b] ; // precomputed
        Scalar3 norm_c = h_norm.data[idx_c] ; // precomputed
        Scalar3 norm_d = h_norm.data[idx_d] ; // precomputed
	//std::cout << "tag a:" << h_tag.data[idx_a]<< " norm_a: " << norm_a.x << ","<< norm_a.y << ","<< norm_a.z << "," <<  std::endl;
					      //
        Scalar rsq_a =norm_a.x*norm_a.x +norm_a.y*norm_a.y+norm_a.z*norm_a.z; // precomputed
        Scalar rsq_b =norm_b.x*norm_b.x +norm_b.y*norm_b.y+norm_b.z*norm_b.z; // precomputed
        Scalar rsq_c =norm_c.x*norm_c.x +norm_c.y*norm_c.y+norm_c.z*norm_c.z; // precomputed
        Scalar rsq_d =norm_d.x*norm_d.x +norm_d.y*norm_d.y+norm_d.z*norm_d.z; // precomputed
									      //
        Scalar r_a =sqrt(rsq_a); // precomputed
        Scalar r_b =sqrt(rsq_b); // precomputed
        Scalar r_c =sqrt(rsq_c); // precomputed
        Scalar r_d =sqrt(rsq_d); // precomputed
				 //
        norm_a /= r_a; // precomputed
        norm_b /= r_b; // precomputed
        norm_c /= r_c; // precomputed
        norm_d /= r_d; // precomputed
				 //

        Scalar sigma_a = h_sigma.data[idx_a]; // precomputed
        Scalar sigma_b = h_sigma.data[idx_b]; // precomputed
        Scalar sigma_c = h_sigma.data[idx_c]; // precomputed
        Scalar sigma_d = h_sigma.data[idx_d]; // precomputed

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

        Scalar inv_sigma_a = 1.0 / sigma_a;
        Scalar inv_sigma_b = 1.0 / sigma_b;
        Scalar inv_sigma_c = 1.0 / sigma_c;
        Scalar inv_sigma_d = 1.0 / sigma_d;

	Scalar dot_norm_a = dot(norm_a,sigma_dash_a);
	Scalar dot_norm_b = dot(norm_b,sigma_dash_b);
	Scalar dot_norm_c = dot(norm_c,sigma_dash_c);
	Scalar dot_norm_d = dot(norm_d,sigma_dash_d);

	// Compute the square gaussian curvature
        Scalar sq_gauss_curv_H_a = dot(sigma_dash_a, sigma_dash_a);
        Scalar sq_gauss_curv_H_b = dot(sigma_dash_b, sigma_dash_b);
        Scalar sq_gauss_curv_H_c = dot(sigma_dash_c, sigma_dash_c);
        Scalar sq_gauss_curv_H_d = dot(sigma_dash_d, sigma_dash_d);

	// Compute the gaussian curvature
        Scalar gauss_curv_H_a = sqrt(sq_gauss_curv_H_a);
        Scalar gauss_curv_H_b = sqrt(sq_gauss_curv_H_b);
        Scalar gauss_curv_H_c = sqrt(sq_gauss_curv_H_c);
        Scalar gauss_curv_H_d = sqrt(sq_gauss_curv_H_d);

        Scalar sign_a = 1.0;
        Scalar sign_b = 1.0;
        Scalar sign_c = 1.0;
        Scalar sign_d = 1.0;

	if (dot_norm_a < 0.0)
	    {
	    sign_a *= -1.0;
	    }
	if (dot_norm_b < 0.0)
	    {
	    sign_b *= -1.0;
	    }
	if (dot_norm_c < 0.0)
	    {
	    sign_c *= -1.0;
	    }
	if (dot_norm_d < 0.0)
	    {
	    sign_d *= -1.0;
	    }

	Scalar C0_sq_a = 4.0 * C0_a * C0_a;
	Scalar C0_sq_b = 4.0 * C0_b * C0_b;
	Scalar C0_sq_c = 4.0 * C0_c * C0_c;
	Scalar C0_sq_d = 4.0 * C0_d * C0_d;

        Scalar sigma_dash_a2 = 0.5 * (sq_gauss_curv_H_a - 4 * sign_a * gauss_curv_H_a * C0_a + C0_sq_a*sigma_a) * inv_sigma_a * inv_sigma_a;
        Scalar sigma_dash_b2 = 0.5 * (sq_gauss_curv_H_b - 4 * sign_b * gauss_curv_H_b * C0_b + C0_sq_b*sigma_b) * inv_sigma_b * inv_sigma_b;
        Scalar sigma_dash_c2 = 0.5 * (sq_gauss_curv_H_c - 4 * sign_c * gauss_curv_H_c * C0_c + C0_sq_c*sigma_c) * inv_sigma_c * inv_sigma_c;
        Scalar sigma_dash_d2 = 0.5 * (sq_gauss_curv_H_d - 4 * sign_d * gauss_curv_H_d * C0_d + C0_sq_d*sigma_d) * inv_sigma_d * inv_sigma_d;

        //Scalar sigma_dash_a2 = 0.5 * (sq_gauss_curv_H_a) * inv_sigma_a * inv_sigma_a;
        //Scalar sigma_dash_b2 = 0.5 * (sq_gauss_curv_H_b) * inv_sigma_b * inv_sigma_b;
        //Scalar sigma_dash_c2 = 0.5 * (sq_gauss_curv_H_c) * inv_sigma_c * inv_sigma_c;
        //Scalar sigma_dash_d2 = 0.5 * (sq_gauss_curv_H_d) * inv_sigma_d * inv_sigma_d;

        Scalar3 Fa;
	Scalar3 grad_Ha;

        grad_Ha.x =  (sign_a*gauss_curv_H_a - 2 * C0_a)*dsigma_dash_a * inv_sigma_a * sigma_dash_a.x  /(gauss_curv_H_a*sign_a);
        grad_Ha.x += (sign_b*gauss_curv_H_b - 2 * C0_b)*(dsigma_dash_b * inv_sigma_b * sigma_dash_b.x)/(gauss_curv_H_b*sign_b);
        grad_Ha.x += (sign_c*gauss_curv_H_c - 2 * C0_c)*(dsigma_dash_c * inv_sigma_c * sigma_dash_c.x)/(gauss_curv_H_c*sign_c);
        grad_Ha.x += (sign_d*gauss_curv_H_d - 2 * C0_d)*(dsigma_dash_d * inv_sigma_d * sigma_dash_d.x)/(gauss_curv_H_d*sign_d);

        grad_Ha.y =  (sign_a*gauss_curv_H_a - 2 * C0_a)*dsigma_dash_a * inv_sigma_a * sigma_dash_a.y  /(gauss_curv_H_a*sign_a);
        grad_Ha.y += (sign_b*gauss_curv_H_b - 2 * C0_b)*(dsigma_dash_b * inv_sigma_b * sigma_dash_b.y)/(gauss_curv_H_b*sign_b);
        grad_Ha.y += (sign_c*gauss_curv_H_c - 2 * C0_c)*(dsigma_dash_c * inv_sigma_c * sigma_dash_c.y)/(gauss_curv_H_c*sign_c);
        grad_Ha.y += (sign_d*gauss_curv_H_d - 2 * C0_d)*(dsigma_dash_d * inv_sigma_d * sigma_dash_d.y)/(gauss_curv_H_d*sign_d);

        grad_Ha.z =  (sign_a*gauss_curv_H_a - 2 * C0_a)* dsigma_dash_a * inv_sigma_a * sigma_dash_a.z /(gauss_curv_H_a*sign_a);
        grad_Ha.z += (sign_b*gauss_curv_H_b - 2 * C0_b)*(dsigma_dash_b * inv_sigma_b * sigma_dash_b.z)/(gauss_curv_H_b*sign_b);
        grad_Ha.z += (sign_c*gauss_curv_H_c - 2 * C0_c)*(dsigma_dash_c * inv_sigma_c * sigma_dash_c.z)/(gauss_curv_H_c*sign_c);
        grad_Ha.z += (sign_d*gauss_curv_H_d - 2 * C0_d)*(dsigma_dash_d * inv_sigma_d * sigma_dash_d.z)/(gauss_curv_H_d*sign_d);

        //grad_Ha.x =  dsigma_dash_a * inv_sigma_a * sigma_dash_a.x ;
        //grad_Ha.x += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.x);
        //grad_Ha.x += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.x);
        //grad_Ha.x += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.x);

        //grad_Ha.y =  dsigma_dash_a * inv_sigma_a * sigma_dash_a.y  ;
        //grad_Ha.y += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.y);
        //grad_Ha.y += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.y);
        //grad_Ha.y += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.y);

        //grad_Ha.z =   dsigma_dash_a * inv_sigma_a * sigma_dash_a.z ;
        //grad_Ha.z += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.z);
        //grad_Ha.z += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.z);
        //grad_Ha.z += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.z);

        Fa.x =   (grad_Ha.x) - sigma_dash_a2 * dsigma_a.x;
        Fa.x -= (sigma_dash_b2 * dsigma_b.x);
        Fa.x -= (sigma_dash_c2 * dsigma_c.x);
        Fa.x -= (sigma_dash_d2 * dsigma_d.x);

        Fa.y =  (grad_Ha.y) - sigma_dash_a2 * dsigma_a.y;
        Fa.y -= (sigma_dash_b2 * dsigma_b.y);
        Fa.y -= (sigma_dash_c2 * dsigma_c.y);
        Fa.y -= (sigma_dash_d2 * dsigma_d.y);

        Fa.z =  (grad_Ha.z) - sigma_dash_a2 * dsigma_a.z;
        Fa.z -= (sigma_dash_b2 * dsigma_b.z);
        Fa.z -= (sigma_dash_c2 * dsigma_c.z);
        Fa.z -= (sigma_dash_d2 * dsigma_d.z);



        Fa *= h_params.data[meshbond_type].k;
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
            h_force.data[idx_a].w += (h_params.data[meshbond_type].k * 0.5
                                     * (sq_gauss_curv_H_a - 4 * C0_a + C0_sq_a)  * inv_sigma_a + eps_kT);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fa.x;
            h_force.data[idx_b].y -= Fa.y;
            h_force.data[idx_b].z -= Fa.z;
            h_force.data[idx_b].w += (h_params.data[meshbond_type].k * 0.5
                                     * (sq_gauss_curv_H_b - 4 * C0_b + C0_sq_b) * inv_sigma_b + eps_kT);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }
        }
    }

void CurvatureHelfrichMeshForceCompute::precomputeParameter()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::overwrite);

    m_sigma.zeroFill();
    m_sigma_dash.zeroFill();
    m_norm.zeroFill();

    // loop over mesh triangles
    const unsigned int size_tri = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size_tri; i++)
        {
	const typename Angle::members_t& triangle = h_triangles.data[i];
        assert(triangle.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[1] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[2] < m_pdata->getMaximumTag() + 1);

        unsigned int idx_a = h_rtag.data[triangle.tag[0]];
        unsigned int idx_b = h_rtag.data[triangle.tag[1]];
        unsigned int idx_c = h_rtag.data[triangle.tag[2]];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        //Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        //Scalar rab = sqrt(rsqab);
        //Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        //Scalar rac = sqrt(rsqac);

        Scalar3 nab, nac;
        nab = dab;// / rab;
        nac = dac;// / rac;

	Scalar3 local_norm;
	local_norm.x = dac.y * dab.z - dac.z * dab.y;
	local_norm.y = dac.z * dab.x - dac.x * dab.z;
	local_norm.z = dac.x * dab.y - dac.y * dab.x;
	//
        if (idx_a < m_pdata->getN())
	    {
            h_norm.data[idx_a].x += local_norm.x ;
            h_norm.data[idx_a].y += local_norm.y ;
            h_norm.data[idx_a].z += local_norm.z ;
	    }
        if (idx_b < m_pdata->getN())
	    {
            h_norm.data[idx_b].x += local_norm.x ;
            h_norm.data[idx_b].y += local_norm.y ;
            h_norm.data[idx_b].z += local_norm.z ;
	    }
        if (idx_c < m_pdata->getN())
	    {
            h_norm.data[idx_c].x += local_norm.x ;
            h_norm.data[idx_c].y += local_norm.y ;
            h_norm.data[idx_c].z += local_norm.z ;
	    }

        }


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
	//std::cout << "sigma_hat_ab:" << sigma_hat_ab << std::endl;

        Scalar sigma_a = sigma_hat_ab * rsqab * 0.25;


        if (idx_a < m_pdata->getN())
            {
            h_sigma.data[idx_a] += sigma_a;
            h_sigma_dash.data[idx_a].x += sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_a].y += sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_a].z += sigma_hat_ab * dab.z;
            }

        if (idx_b < m_pdata->getN())
            {
            h_sigma.data[idx_b] += sigma_a;
            h_sigma_dash.data[idx_b].x -= sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_b].y -= sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_b].z -= sigma_hat_ab * dab.z;
            }
        }
    }

Scalar CurvatureHelfrichMeshForceCompute::energyDiff(unsigned int idx_a,
                                            unsigned int idx_b,
                                            unsigned int idx_c,
                                            unsigned int idx_d,
                                            unsigned int type_id)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::read);

    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar C0_a = 0.0;
    Scalar C0_b = 0.0;
    Scalar C0_c = 0.0;
    Scalar C0_d = 0.0;

    Scalar eps_kT_old = 0.0;
    Scalar eps_kT_new = 0.0;

    if (h_rtag.data[idx_a] > h_params.data[type_id].tag_max)
    {
    	C0_a = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_b] > h_params.data[type_id].tag_max)
    {
    	C0_b = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_c] > h_params.data[type_id].tag_max)
    {
    	C0_c = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_d] > h_params.data[type_id].tag_max)
    {
    	C0_d = h_params.data[type_id].C0;
    }


    if ((h_rtag.data[idx_a] > h_params.data[type_id].tag_max) && (h_rtag.data[idx_b] > h_params.data[type_id].tag_max))
    {
    	eps_kT_old = h_params.data[type_id].eps_kT;
    }

    if ((h_rtag.data[idx_c] > h_params.data[type_id].tag_max) && (h_rtag.data[idx_d] > h_params.data[type_id].tag_max))
    {
    	eps_kT_new = h_params.data[type_id].eps_kT;
    }


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

    m_sigma_dash_diff_a = sigma_hat_ab * dab + sigma_hat_ac * dac + sigma_hat_ad * dad;
    m_sigma_dash_diff_b = -sigma_hat_ab * dab + sigma_hat_bc * dbc + sigma_hat_bd * dbd;
    m_sigma_dash_diff_c = -sigma_hat_ac * dac - sigma_hat_bc * dbc + sigma_hat_cd * dcd;
    m_sigma_dash_diff_d = -sigma_hat_ad * dad - sigma_hat_bd * dbd - sigma_hat_cd * dcd;

    Scalar3 sigma_dash_a_n = sigma_dash_a + m_sigma_dash_diff_a;
    Scalar3 sigma_dash_b_n = sigma_dash_b + m_sigma_dash_diff_b;
    Scalar3 sigma_dash_c_n = sigma_dash_c + m_sigma_dash_diff_c;
    Scalar3 sigma_dash_d_n = sigma_dash_d + m_sigma_dash_diff_d;

    Scalar sq_H_a = dot(sigma_dash_a, sigma_dash_a)       ;
    Scalar sq_H_b=       dot(sigma_dash_b, sigma_dash_b);
    Scalar sq_H_c=       dot(sigma_dash_c, sigma_dash_c);
    Scalar sq_H_d=       dot(sigma_dash_d, sigma_dash_d);

    Scalar H_a=sqrt(sq_H_a);
    Scalar H_b=sqrt(sq_H_b);
    Scalar H_c=sqrt(sq_H_c);
    Scalar H_d=sqrt(sq_H_d);

    Scalar sq_H_a_n= dot(sigma_dash_a_n, sigma_dash_a_n);
    Scalar sq_H_b_n= dot(sigma_dash_b_n, sigma_dash_b_n);
    Scalar sq_H_c_n= dot(sigma_dash_c_n, sigma_dash_c_n);
    Scalar sq_H_d_n= dot(sigma_dash_d_n, sigma_dash_d_n);

    Scalar H_a_n=sqrt(sq_H_a_n);
    Scalar H_b_n=sqrt(sq_H_b_n);
    Scalar H_c_n=sqrt(sq_H_c_n);
    Scalar H_d_n=sqrt(sq_H_d_n);

    // we now compute the sign of this contribution to the curvature based on the normal
    // of the face of the triangle associated with this bond - question of if this norm is enough
    // to calculate normal direction of the vertex or if this has to be accomplished with precompute
    // parameters
    Scalar3 norm_a = h_norm.data[idx_a] ; // precomputed
    Scalar3 norm_b = h_norm.data[idx_b] ; // precomputed
    Scalar3 norm_c = h_norm.data[idx_c] ; // precomputed
    Scalar3 norm_d = h_norm.data[idx_d] ; // precomputed
					  //
    Scalar3 norm_a_n = h_norm.data[idx_a] ; // precomputed
    Scalar3 norm_b_n = h_norm.data[idx_b] ; // precomputed
    Scalar3 norm_c_n = h_norm.data[idx_c] ; // precomputed
    Scalar3 norm_d_n = h_norm.data[idx_d] ; // precomputed
					      //
    Scalar3 norm_t1;
    Scalar3 norm_t2;
    Scalar3 norm_t3;
    Scalar3 norm_t4;

    norm_t1.x = dac.y * dab.z - dac.z * dab.y;
    norm_t1.y = dac.z * dab.x - dac.x * dab.z;
    norm_t1.z = dac.x * dab.y - dac.y * dab.x;

    norm_t2.x = dab.y * dad.z - dab.z * dad.y;
    norm_t2.y = dab.z * dad.x - dab.x * dad.z;
    norm_t2.z = dab.x * dad.y - dab.y * dad.x;

    norm_t3.x = dac.y * dad.z - dac.z * dad.y;
    norm_t3.y = dac.z * dad.x - dac.x * dad.z;
    norm_t3.z = dac.x * dad.y - dac.y * dad.x;

    norm_t4.x = dbd.y * dbc.z - dbd.z * dbc.y;
    norm_t4.y = dbd.z * dbc.x - dbd.x * dbc.z;
    norm_t4.z = dbd.x * dbc.y - dbd.y * dbc.x;

    //norm_t1.x = dac.x * dab.y - dac.y*dab.z;
    //norm_t1.y = dac.x * dab.z - dac.z*dab.x;
    //norm_t1.z = dac.y * dab.x - dac.x*dab.y;

    //norm_t2.x = dab.x * dad.y - dab.y*dad.z;
    //norm_t2.y = dab.x * dad.z - dab.z*dad.x;
    //norm_t2.z = dab.y * dad.x - dab.x*dad.y;

    //norm_t3.x = dac.x * dad.y - dac.y*dad.z;
    //norm_t3.y = dac.x * dad.z - dac.z*dad.x;
    //norm_t3.z = dac.y * dad.x - dac.x*dad.y;

    //norm_t4.x = dbd.x * dbc.y - dbd.y*dbc.z;
    //norm_t4.y = dbd.x * dbc.z - dbd.z*dbc.x;
    //norm_t4.z = dbd.y * dbc.x - dbd.x*dbc.y;

    norm_a_n += (norm_t3 - norm_t1 - norm_t2);
    norm_b_n += (norm_t4 - norm_t1 - norm_t2);
    norm_c_n += (norm_t4 + norm_t3 - norm_t1);
    norm_d_n += (norm_t4 + norm_t3 - norm_t2);

    Scalar dot_norm_a = dot(norm_a,sigma_dash_a);
    Scalar dot_norm_b = dot(norm_b,sigma_dash_b);
    Scalar dot_norm_c = dot(norm_c,sigma_dash_c);
    Scalar dot_norm_d = dot(norm_d,sigma_dash_d);

    Scalar dot_norm_a_n = dot(norm_a_n,sigma_dash_a_n);
    Scalar dot_norm_b_n = dot(norm_b_n,sigma_dash_b_n);
    Scalar dot_norm_c_n = dot(norm_c_n,sigma_dash_c_n);
    Scalar dot_norm_d_n = dot(norm_d_n,sigma_dash_d_n);

    if (dot_norm_a < 0.0)
        {
        H_a *= -1.0;
        }
    if (dot_norm_b < 0.0)
        {
        H_b *= -1.0;
        }
    if (dot_norm_c < 0.0)
        {
        H_c *= -1.0;
        }
    if (dot_norm_d < 0.0)
        {
        H_d *= -1.0;
        }

    if (dot_norm_a_n < 0.0)
        {
        H_a_n *= -1.0;
        }
    if (dot_norm_b_n < 0.0)
        {
        H_b_n *= -1.0;
        }
    if (dot_norm_c_n < 0.0)
        {
        H_c_n *= -1.0;
        }
    if (dot_norm_d_n < 0.0)
        {
        H_d_n *= -1.0;
        }

    Scalar C0_sq_a = 4 * C0_a * C0_a;
    Scalar C0_sq_b = 4 * C0_b * C0_b;
    Scalar C0_sq_c = 4 * C0_c * C0_c;
    Scalar C0_sq_d = 4 * C0_d * C0_d;


    Scalar energy_old =(sq_H_a - 4 * C0_a * H_a + C0_sq_a)/ sigma_a;
    energy_old +=     ((sq_H_b - 4 * C0_b * H_b + C0_sq_b)/ sigma_b);
    energy_old +=     ((sq_H_c - 4 * C0_c * H_c + C0_sq_c)/ sigma_c);
    energy_old +=     ((sq_H_d - 4 * C0_d * H_d + C0_sq_d)/ sigma_d);

    Scalar energy_new = (sq_H_a_n - 4 * C0_a * H_a_n + C0_sq_a) / sigma_a_n;
    energy_new +=      ((sq_H_b_n - 4 * C0_b * H_b_n + C0_sq_b) / sigma_b_n);
    energy_new +=      ((sq_H_c_n - 4 * C0_c * H_c_n + C0_sq_c) / sigma_c_n);
    energy_new +=      ((sq_H_d_n - 4 * C0_d * H_d_n + C0_sq_d) / sigma_d_n);

    if (energy_new < 0)
        return DBL_MAX;

    return h_params.data[type_id].k * 0.5 * (energy_new - energy_old) + eps_kT_new - eps_kT_old;
    }

void CurvatureHelfrichMeshForceCompute::postcomputeParameter(unsigned int idx_a,
                                                    unsigned int idx_b,
                                                    unsigned int idx_c,
                                                    unsigned int idx_d,
                                                    unsigned int type_id)
    {
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::readwrite);

    h_sigma.data[idx_a] += m_sigma_diff_a;
    h_sigma.data[idx_b] += m_sigma_diff_b;
    h_sigma.data[idx_c] += m_sigma_diff_c;
    h_sigma.data[idx_d] += m_sigma_diff_d;

    h_sigma_dash.data[idx_a] += m_sigma_dash_diff_a;
    h_sigma_dash.data[idx_b] += m_sigma_dash_diff_b;
    h_sigma_dash.data[idx_c] += m_sigma_dash_diff_c;
    h_sigma_dash.data[idx_d] += m_sigma_dash_diff_d;
    }

namespace detail
    {
void export_CurvatureHelfrichMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<CurvatureHelfrichMeshForceCompute,
                     MeshForceCompute,
                     std::shared_ptr<CurvatureHelfrichMeshForceCompute>>(m, "CurvatureHelfrichMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &CurvatureHelfrichMeshForceCompute::setParamsPython)
        .def("getParams", &CurvatureHelfrichMeshForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
