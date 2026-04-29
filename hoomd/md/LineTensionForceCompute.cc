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

    GPUArray<line_tension_param_t> params(
        m_pdata->getNTypes(),
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
    unsigned int type,
    const line_tension_param_t& params)
{
    ArrayHandle<line_tension_param_t> h_params(
        m_params,
        access_location::host,
        access_mode::readwrite);

    h_params.data[type] = params;

    if (params.l < Scalar(0.0))
    {
        m_exec_conf->msg->warning()
            << "LineTension: l < 0" << endl;
    }
}

void LineTensionForceCompute::setParamsPython(
    std::string type,
    pybind11::dict params)
{
    unsigned int typ =
        m_pdata->getTypeByName(type);

    setParams(
        typ,
        line_tension_param_t(params));
}

pybind11::dict LineTensionForceCompute::getParams(
    std::string type)
{
    unsigned int typ =
        m_pdata->getTypeByName(type);

    if (typ >= m_pdata->getNTypes())
    {
        m_exec_conf->msg->error()
            << "mesh.line_tension: invalid type"
            << endl;

        throw runtime_error(
            "Invalid type in LineTensionForceCompute");
    }

    ArrayHandle<line_tension_param_t> h_params(
        m_params,
        access_location::host,
        access_mode::read);

    return h_params.data[typ].asDict();
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

        Scalar lambda = 0.0;
        bool match = false;

        // Type-pair matching
        for (unsigned int t = 0; t < m_pdata->getNTypes(); t++)
        {
            const auto& p = h_params.data[t];

            if ((type_a == p.type_i && type_b == p.type_j) ||
                (type_a == p.type_j && type_b == p.type_i))
            {
                lambda = p.l;
                match = true;
                break;
            }
        }

        if (!match || lambda == Scalar(0.0))
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
        F.x = -lambda * dr.x * inv_r;
        F.y = -lambda * dr.y * inv_r;
        F.z = -lambda * dr.z * inv_r;

        Scalar energy = Scalar(0.5) * lambda * r;

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
