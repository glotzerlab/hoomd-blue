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
	
	// Try to add per-particle energy fix
	unsigned int N = m_pdata->getN() + m_pdata->getNGhosts();

    GPUArray<Scalar> energy_array(N, m_exec_conf); //mem safe allocation
    m_particle_energy.swap(energy_array); //internal ownership is correct

    // zero initialize
    ArrayHandle<Scalar> h_energy(
        m_particle_energy,
        access_location::host,
        access_mode::overwrite);

    for (unsigned int i = 0; i < N; i++)
    {
        h_energy.data[i] = Scalar(0.0);
    }
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
        m_mesh_data->getMeshBondData()->getN();

    for (unsigned int i = 0; i < n_bonds; i++)
    {
        const auto& bond = h_bonds.data[i];

        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];

        if (idx_a >= m_pdata->getN() + m_pdata->getNGhosts()
         || idx_b >= m_pdata->getN() + m_pdata->getNGhosts())
            continue;

        unsigned int type_a =
            __scalar_as_int(h_pos.data[idx_a].w);
        unsigned int type_b =
            __scalar_as_int(h_pos.data[idx_b].w);

		unsigned int idx = m_typpair_idx(type_a, type_b);
		Scalar lam = h_params.data[idx].l;

		if (lam == Scalar(0.0))
			continue;
        
		Scalar3 dr;
        dr.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dr.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dr.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        dr = box.minImage(dr);

        Scalar rsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
        Scalar r = sqrt(rsq);

        if (r < SMALL)
            continue;

        Scalar inv_r = Scalar(1.0) / r;

        Scalar3 F;
        F.x = -lam * dr.x * inv_r;
        F.y = -lam * dr.y * inv_r;
        F.z = -lam * dr.z * inv_r;

        Scalar energy = Scalar(0.5) * lam * r;

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
