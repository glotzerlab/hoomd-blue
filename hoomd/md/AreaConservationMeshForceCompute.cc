// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AreaConservationMeshForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file AreaConservationMeshForceCompute.cc
    \brief Contains code for the AreaConservationMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
AreaConservationMeshForceCompute::AreaConservationMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_Amesh(NULL), m_mesh_data(meshdef), m_area(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing AreaConservationMeshForceCompute" << endl;

    // allocate the parameters
    m_K = new Scalar[m_pdata->getNTypes()];

    // allocate the parameters
    m_Amesh = new Scalar[m_pdata->getNTypes()];
    }

AreaConservationMeshForceCompute::~AreaConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying AreaConservationMeshForceCompute" << endl;

    delete[] m_K;
    delete[] m_Amesh;
    m_K = NULL;
    m_Amesh = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void AreaConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar A_mesh)
    {
    m_K[type] = K;
    m_Amesh[type] = A_mesh;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "area: specified K <= 0" << endl;
    if (A_mesh <= 0)
        m_exec_conf->msg->warning() << "area: specified A_mesh <= 0" << endl;
    }

void AreaConservationMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = aconstraint_params(params);
    setParams(typ, _params.k, _params.A_mesh);
    }

pybind11::dict AreaConservationMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.area: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in AreaConservationMeshForceCompute");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["A_mesh"] = m_Amesh[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void AreaConservationMeshForceCompute::computeForces(uint64_t timestep)
    {
    precomputeParameter(); // precompute area

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_triangles.data);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar area_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        area_virial[i] = Scalar(0.0);

    Scalar AreaDiff = m_area - m_Amesh[0];

    Scalar energy = m_K[0] * AreaDiff * AreaDiff / (2 * m_Amesh[0] * m_pdata->getN());

    AreaDiff = -m_K[0] / (2 * m_Amesh[0]) * AreaDiff;

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename MeshTriangle::members_t& triangle = h_triangles.data[i];

        unsigned int ttag_a = triangle.tag[0];
        assert(ttag_a < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_b = triangle.tag[1];
        assert(ttag_b < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_c = triangle.tag[2];
        assert(ttag_c < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[ttag_a];
        unsigned int idx_b = h_rtag.data[ttag_b];
        unsigned int idx_c = h_rtag.data[ttag_c];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
        dab.y = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
        dab.z = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_c].x - h_pos.data[idx_a].x;
        dac.y = h_pos.data[idx_c].y - h_pos.data[idx_a].y;
        dac.z = h_pos.data[idx_c].z - h_pos.data[idx_a].z;

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dac = box.minImage(dac);

        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;

        Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dab.z * dac.z;

        Scalar area2 = sqrt(rsqab * rsqac - rabrac * rabrac);

        Scalar3 Fa, Fb, Fc;

        Fa = AreaDiff / area2 * ((rabrac - rsqac) * dab + (rabrac - rsqab) * dac);

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_a].x * Fa.x; // xx
            area_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_a].y * Fa.x; // xy
            area_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.x; // xz
            area_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_a].y * Fa.y; // yy
            area_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.y; // yz
            area_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.z; // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w = energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += area_virial[j];
            }

        Fb = AreaDiff / area2 * (rsqac * dab - rabrac * dac);

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_b].x * Fb.x; // xx
            area_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_b].y * Fb.x; // xy
            area_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.x; // xz
            area_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_b].y * Fb.y; // yy
            area_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.y; // yz
            area_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x += Fb.x;
            h_force.data[idx_b].y += Fb.y;
            h_force.data[idx_b].z += Fb.z;
            h_force.data[idx_b].w = energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += area_virial[j];
            }

        Fc = AreaDiff / area2 * (rsqab * dac - rabrac * dab);

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_c].x * Fc.x; // xx
            area_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_c].y * Fc.x; // xy
            area_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.x; // xz
            area_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_c].y * Fc.y; // yy
            area_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.y; // yz
            area_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += Fc.x;
            h_force.data[idx_c].y += Fc.y;
            h_force.data[idx_c].z += Fc.z;
            h_force.data[idx_c].w = energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += area_virial[j];
            }
        }
    }

void AreaConservationMeshForceCompute::precomputeParameter()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();
    m_area = 0;

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename MeshTriangle::members_t& triangle = h_triangles.data[i];

        unsigned int ttag_a = triangle.tag[0];
        assert(ttag_a < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_b = triangle.tag[1];
        assert(ttag_b < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_c = triangle.tag[2];
        assert(ttag_c < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[ttag_a];
        unsigned int idx_b = h_rtag.data[ttag_b];
        unsigned int idx_c = h_rtag.data[ttag_c];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
        dab.y = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
        dab.z = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_c].x - h_pos.data[idx_a].x;
        dac.y = h_pos.data[idx_c].y - h_pos.data[idx_a].y;
        dac.z = h_pos.data[idx_c].z - h_pos.data[idx_a].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;

        Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dab.z * dac.z;

        Scalar area_tri = sqrt(rsqab * rsqac - rabrac * rabrac) / 2.0;

        m_area += area_tri;
        }
    }

Scalar AreaConservationMeshForceCompute::energyDiff(unsigned int idx_a,
                                                    unsigned int idx_b,
                                                    unsigned int idx_c,
                                                    unsigned int idx_d,
                                                    unsigned int type_id)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();

    Scalar3 dab;
    dab.x = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
    dab.y = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
    dab.z = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

    Scalar3 dac;
    dac.x = h_pos.data[idx_c].x - h_pos.data[idx_a].x;
    dac.y = h_pos.data[idx_c].y - h_pos.data[idx_a].y;
    dac.z = h_pos.data[idx_c].z - h_pos.data[idx_a].z;

    Scalar3 dad;
    dad.x = h_pos.data[idx_d].x - h_pos.data[idx_a].x;
    dad.y = h_pos.data[idx_d].y - h_pos.data[idx_a].y;
    dad.z = h_pos.data[idx_d].z - h_pos.data[idx_a].z;

    Scalar3 ddc;
    ddc.x = h_pos.data[idx_c].x - h_pos.data[idx_d].x;
    ddc.y = h_pos.data[idx_c].y - h_pos.data[idx_d].y;
    ddc.z = h_pos.data[idx_c].z - h_pos.data[idx_d].z;

    Scalar3 dbc;
    dbc.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
    dbc.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
    dbc.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

    dab = box.minImage(dab);
    dac = box.minImage(dac);
    dad = box.minImage(dad);
    ddc = box.minImage(ddc);
    dbc = box.minImage(dbc);

    // FLOPS: 42 / MEM TRANSFER: 6 Scalars
    Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
    Scalar rab = sqrt(rsqab);
    Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
    Scalar rac = sqrt(rsqac);

    Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
    Scalar rad = sqrt(rsqad);

    Scalar rsqdc = ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z;
    Scalar rdc = sqrt(rsqdc);

    Scalar rsqbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
    Scalar rbc = sqrt(rsqbc);

    Scalar3 nab, nac, nad, ndc, nbc;
    nab = dab / rab;
    nac = dac / rac;
    nad = dad / rad;
    ndc = ddc / rdc;
    nbc = dbc / rbc;

    Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;
    if (c_baac > 1.0)
        c_baac = 1.0;
    if (c_baac < -1.0)
        c_baac = -1.0;
    Scalar s_baac = sqrt(1.0 - c_baac * c_baac);

    Scalar c_baad = nab.x * nad.x + nab.y * nad.y + nab.z * nad.z;
    if (c_baad > 1.0)
        c_baad = 1.0;
    if (c_baad < -1.0)
        c_baad = -1.0;
    Scalar s_baad = sqrt(1.0 - c_baad * c_baad);

    Scalar c_dcca = ndc.x * nac.x + ndc.y * nac.y + ndc.z * nac.z;
    if (c_dcca > 1.0)
        c_dcca = 1.0;
    if (c_dcca < -1.0)
        c_dcca = -1.0;
    Scalar s_dcca = sqrt(1.0 - c_dcca * c_dcca);

    Scalar c_dccb = ndc.x * nbc.x + ndc.y * nbc.y + ndc.z * nbc.z;
    if (c_dccb > 1.0)
        c_dccb = 1.0;
    if (c_dccb < -1.0)
        c_dccb = -1.0;
    Scalar s_dccb = sqrt(1.0 - c_dccb * c_dccb);

    m_area_diff = rdc * (rac * s_dcca + rbc * s_dccb);
    m_area_diff -= rab * (rac * s_baac + rad * s_baad);

    m_area_diff /= 2.0;

    Scalar energy_old = m_area - m_Amesh[type_id];

    Scalar energy_new = energy_old + m_area_diff;

    energy_old = energy_old * energy_old;

    energy_new = energy_new * energy_new;

    return m_K[0] / (2.0 * m_Amesh[0]) * (energy_new - energy_old);
    }

namespace detail
    {
void export_AreaConservationMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<AreaConservationMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<AreaConservationMeshForceCompute>>(
        m,
        "AreaConservationMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &AreaConservationMeshForceCompute::setParamsPython)
        .def("getParams", &AreaConservationMeshForceCompute::getParams)
        .def("getArea", &AreaConservationMeshForceCompute::getArea);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
