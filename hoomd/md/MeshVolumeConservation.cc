// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshVolumeConservation.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file MeshVolumeConservation.cc
    \brief Contains code for the MeshVolumeConservation class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
MeshVolumeConservation::MeshVolumeConservation(std::shared_ptr<SystemDefinition> sysdef,
                                               std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_V0(NULL), m_mesh_data(meshdef), m_volume(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing MeshVolumeConservation" << endl;

    // allocate the parameters
    m_K = new Scalar[m_pdata->getNTypes()];

    // allocate the parameters
    m_V0 = new Scalar[m_pdata->getNTypes()];
    }

MeshVolumeConservation::~MeshVolumeConservation()
    {
    m_exec_conf->msg->notice(5) << "Destroying MeshVolumeConservation" << endl;

    delete[] m_K;
    delete[] m_V0;
    m_K = NULL;
    m_V0 = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void MeshVolumeConservation::setParams(unsigned int type, Scalar K, Scalar V0)
    {
    m_K[type] = K;
    m_V0[type] = V0;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "volume: specified K <= 0" << endl;
    if (V0 <= 0)
        m_exec_conf->msg->warning() << "volume: specified V0 <= 0" << endl;
    }

void MeshVolumeConservation::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = vconstraint_params(params);
    setParams(typ, _params.k, _params.V0);
    }

pybind11::dict MeshVolumeConservation::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.helfrich: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in MeshVolumeConservation");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["V0"] = m_V0[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void MeshVolumeConservation::computeForces(uint64_t timestep)
    {
    if (m_prof)
        m_prof->push("Harmonic Angle");

    computeVolume(); // precompute sigmas

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

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

    Scalar helfrich_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        helfrich_virial[i] = Scalar(0.0);

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename MeshTriangle::members_t& triangle = h_triangles.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        vec3<Scalar> pos_a(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        vec3<Scalar> pos_b(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);
        vec3<Scalar> pos_c(h_pos.data[idx_c].x, h_pos.data[idx_c].y, h_pos.data[idx_c].z);

        vec3<Scalar> dVol_a = cross(pos_b, pos_c);

        vec3<Scalar> dVol_b = cross(pos_c, pos_a);

        vec3<Scalar> dVol_c = cross(pos_a, pos_b);

        // calculate d\vec{r}
        Scalar3 dba;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dbc;
        dac.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dac.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dac.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        dba = box.minImage(dba);
        dbc = box.minImage(dbc);

        vec3<Scalar> normal
            = cross(vec3 < Scalar(dba.x, dba.y, dba.z), vec3<Scalar>(dbc.x, dbc.y, dbc.z));

        Scalar3 Fa;

        Fa.x = dsigma_dash_a * inv_sigma_a * sigma_dash_a.x - sigma_dash_a2 * dsigma_a.x;
        Fa.x += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.x - sigma_dash_b2 * dsigma_b.x);
        Fa.x += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.x - sigma_dash_c2 * dsigma_c.x);
        Fa.x += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.x - sigma_dash_d2 * dsigma_d.x);

        Fa.y = dsigma_dash_a * inv_sigma_a * sigma_dash_a.y - sigma_dash_a2 * dsigma_a.y;
        Fa.y += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.y - sigma_dash_b2 * dsigma_b.y);
        Fa.y += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.y - sigma_dash_c2 * dsigma_c.y);
        Fa.y += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.y - sigma_dash_d2 * dsigma_d.y);

        Fa.z = dsigma_dash_a * inv_sigma_a * sigma_dash_a.z - sigma_dash_a2 * dsigma_a.z;
        Fa.z += (dsigma_dash_b * inv_sigma_b * sigma_dash_b.z - sigma_dash_b2 * dsigma_b.z);
        Fa.z += (dsigma_dash_c * inv_sigma_c * sigma_dash_c.z - sigma_dash_c2 * dsigma_c.z);
        Fa.z += (dsigma_dash_d * inv_sigma_d * sigma_dash_d.z - sigma_dash_d2 * dsigma_d.z);

        Fa *= m_K[0];

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
            h_force.data[idx_a].w = m_K[0] * 0.5 * dot(sigma_dash_a, sigma_dash_a) * inv_sigma_a;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fa.x;
            h_force.data[idx_b].y -= Fa.y;
            h_force.data[idx_b].z -= Fa.z;
            h_force.data[idx_b].w = m_K[0] * 0.5 * dot(sigma_dash_b, sigma_dash_b) * inv_sigma_b;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }
        }

    if (m_prof)
        m_prof->pop();
    }

void MeshVolumeConservation::computeSigma()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::overwrite);

    memset((void*)h_sigma.data, 0, sizeof(Scalar) * m_sigma.getNumElements());
    memset((void*)h_sigma_dash.data, 0, sizeof(Scalar3) * m_sigma_dash.getNumElements());

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename MeshBond::members_t& bond = h_bonds.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];

        unsigned int tr_idx1 = bond.tag[2];
        unsigned int tr_idx2 = bond.tag[3];

        if (tr_idx1 == tr_idx2)
            continue;

        const typename MeshTriangle::members_t& triangle1 = h_triangles.data[tr_idx1];
        const typename MeshTriangle::members_t& triangle2 = h_triangles.data[tr_idx2];

        unsigned int idx_c = h_rtag.data[triangle1.tag[0]];

        unsigned int iterator = 1;
        while (idx_a == idx_c || idx_b == idx_c)
            {
            idx_c = h_rtag.data[triangle1.tag[iterator]];
            iterator++;
            }

        unsigned int idx_d = h_rtag.data[triangle2.tag[0]];

        iterator = 1;
        while (idx_a == idx_d || idx_b == idx_d)
            {
            idx_d = h_rtag.data[triangle2.tag[iterator]];
            iterator++;
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

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
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

        vec3<Scalar> nbac
            = cross(vec3<Scalar>(nab.x, nab.y, nab.z), vec3<Scalar>(nac.x, nac.y, nac.z));

        Scalar inv_nbac = 1.0 / sqrt(dot(nbac, nbac));

        vec3<Scalar> nbad
            = cross(vec3<Scalar>(nab.x, nab.y, nab.z), vec3<Scalar>(nad.x, nad.y, nad.z));

        Scalar inv_nbad = 1.0 / sqrt(dot(nbad, nbad));

        if (dot(nbac, nbad) * inv_nbad * inv_nbac > 0.9)
            {
            this->m_exec_conf->msg->error() << "helfrich calculations : triangles " << tr_idx1
                                            << " " << tr_idx2 << " overlap." << std::endl
                                            << std::endl;
            throw std::runtime_error("Error in bending energy calculation");
            }

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

        h_sigma.data[idx_a] += sigma_a;
        h_sigma.data[idx_b] += sigma_a;

        h_sigma_dash.data[idx_a].x += sigma_hat_ab * dab.x;
        h_sigma_dash.data[idx_a].y += sigma_hat_ab * dab.y;
        h_sigma_dash.data[idx_a].z += sigma_hat_ab * dab.z;

        h_sigma_dash.data[idx_b].x -= sigma_hat_ab * dab.x;
        h_sigma_dash.data[idx_b].y -= sigma_hat_ab * dab.y;
        h_sigma_dash.data[idx_b].z -= sigma_hat_ab * dab.z;
        }
    }

namespace detail
    {
void export_MeshVolumeConservation(pybind11::module& m)
    {
    pybind11::class_<MeshVolumeConservation, ForceCompute, std::shared_ptr<MeshVolumeConservation>>(
        m,
        "MeshVolumeConservation")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &MeshVolumeConservation::setParamsPython)
        .def("getParams", &MeshVolumeConservation::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
