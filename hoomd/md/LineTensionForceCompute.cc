// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "LineTensionForceCompute.h"

#include <iostream>
#include <stdexcept>

using namespace std;

#define SMALL Scalar(0.001)

namespace hoomd
{
namespace md
{

LineTensionForceCompute::LineTensionForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef)
{
    m_exec_conf->msg->notice(5)
        << "Constructing LineTensionForceCompute"
        << endl;
	
	m_typpair_idx = Index2D(m_pdata->getNTypes());

	GPUArray<line_tension_param_t> params(
		m_typpair_idx.getNumElements(),
		m_exec_conf);

	m_params.swap(params);
}

LineTensionForceCompute::~LineTensionForceCompute()
{
    m_exec_conf->msg->notice(5)
        << "Destroying LineTensionForceCompute"
        << endl;
}

void LineTensionForceCompute::setParams(
    unsigned int typ1,
	unsigned int typ2,
    const line_tension_param_t& params
	)
{
    ArrayHandle<line_tension_param_t> h_params(
        m_params,
        access_location::host,
        access_mode::readwrite);
    
	unsigned int idx1 = m_typpair_idx(typ1, typ2);
	unsigned int idx2 = m_typpair_idx(typ2, typ1);

	h_params.data[idx1] = params;
	h_params.data[idx2] = params;

    if (params.l < Scalar(0.0))
    {
        m_exec_conf->msg->warning()
            << "LineTension: l < 0" << endl;
    }
}

void LineTensionForceCompute::setParamsPython(
    pybind11::tuple typ,
    pybind11::dict params)
{
    unsigned int typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    unsigned int typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());

    setParams(
        typ1,
		typ2,
        line_tension_param_t(params));
}

pybind11::dict LineTensionForceCompute::getParams(pybind11::tuple typ)
{
    unsigned int typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    unsigned int typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());

    if (typ1 >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error()
            << "mesh.line_tension: invalid type"
            << endl;

        throw runtime_error(
            "First type invalid in LineTensionForceCompute");
    }
    
	if (typ2 >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error()
            << "mesh.line_tension: invalid type"
            << endl;

        throw runtime_error(
            "Second type invalide in LineTensionForceCompute");
    }

	//Not in PotentialPair.h
	/*
    ArrayHandle<line_tension_param_t> h_params(
        m_params,
        access_location::host,
        access_mode::read);
	return h_params.data[typ].asDict();
	*/
	ArrayHandle<line_tension_param_t> h_params(
		m_params,
		access_location::host,
		access_mode::read);

	return h_params.data[m_typpair_idx(typ1, typ2)].asDict();
}

void LineTensionForceCompute::computeForces(uint64_t timestep)
{
    assert(m_pdata);

    ArrayHandle<Scalar4> h_pos(
        m_pdata->getPositions(),
        access_location::host,
        access_mode::read);

    ArrayHandle<unsigned int> h_rtag(
        m_pdata->getRTags(),
        access_location::host,
        access_mode::read);

    ArrayHandle<Scalar4> h_force(
        m_force,
        access_location::host,
        access_mode::overwrite);

    ArrayHandle<line_tension_param_t> h_params(
        m_params,
        access_location::host,
        access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    m_force.zeroFill();

    const BoxDim& box = m_pdata->getGlobalBox();

    const unsigned int n_bonds =
        (unsigned int)m_mesh_data->getMeshBondData()->getN();

	// Sum over bonds (as opposed to total particles in General Helfrich)
    for (unsigned int i = 0; i < n_bonds; i++)
    {
        const auto& bond = h_bonds.data[i];

        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];
		unsigned int tag_c = bond.tag[2];
		unsigned int tag_d = bond.tag[3];

		if (tag_c == tag_d)
			continue;

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];
        unsigned int idx_c = h_rtag.data[tag_c];
        unsigned int idx_d = h_rtag.data[tag_d];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());                                                   
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());
        
		Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;
		
		Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

		Scalar3 dad;
        dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
        dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
        dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

		Scalar3 dbd;
        dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
        dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
        dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

		Scalar3 dbc;
        dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
        dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
        dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

		unsigned int type_a =
            __scalar_as_int(h_pos.data[idx_a].w);
        unsigned int type_b =
            __scalar_as_int(h_pos.data[idx_b].w);
        unsigned int type_c =
            __scalar_as_int(h_pos.data[idx_c].w);
        unsigned int type_d =
            __scalar_as_int(h_pos.data[idx_d].w);

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

		// sigma_hat_ab is the norm of sigma_ij 
        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;

		unsigned int idx = m_typpair_idx(type_a, type_b);
		Scalar lam = h_params.data[idx].l;

		if (lam == Scalar(0.0))
			continue;
        
        Scalar3 F;
        F.x = -lam * sigma_hat_ab * nab.x;
        F.y = -lam * sigma_hat_ab * nab.y;
        F.z = -lam * sigma_hat_ab * nab.z;

        Scalar energy = Scalar(0.5) * lam * sigma_hat_ab;

        if (idx_a < m_pdata->getN())
        {
            h_force.data[idx_a].x += F.x;
            h_force.data[idx_a].y += F.y;
            h_force.data[idx_a].z += F.z;
            h_force.data[idx_a].w += energy;
        }

        if (idx_b < m_pdata->getN())
        {
            h_force.data[idx_b].x -= F.x;
            h_force.data[idx_b].y -= F.y;
            h_force.data[idx_b].z -= F.z;
            h_force.data[idx_b].w += energy;
        }
    }
}

namespace detail
{

void export_LineTensionForceCompute(pybind11::module& m)
{
    pybind11::class_<LineTensionForceCompute,
                     MeshForceCompute,
                     std::shared_ptr<LineTensionForceCompute>>(
        m,
        "LineTensionForceCompute")
        .def(pybind11::init<
             std::shared_ptr<SystemDefinition>,
             std::shared_ptr<MeshDefinition>>())
        .def("setParams", &LineTensionForceCompute::setParamsPython)
        .def("getParams", &LineTensionForceCompute::getParams);
}

} // namespace detail

} // namespace md
} // namespace hoomd
