// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichMeshForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file HelfrichMeshForceCompute.cc
    \brief Contains code for the HelfrichMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation 
    \post Memory is allocated, and forces are zeroed.
*/
HelfrichMeshForceCompute::HelfrichMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing HelfrichMeshForceCompute" << endl;

    // allocate the parameters
    m_K = new Scalar[m_mesh_data->getMeshBondData()->getNTypes()];

    // allocate memory for the per-type normal verctors
    GlobalVector<Scalar3> tmp_sigma_dash(m_pdata->getN(), m_exec_conf);

    m_sigma_dash.swap(tmp_sigma_dash);
    TAG_ALLOCATION(m_sigma_dash);

    // allocate memory for the per-type normal verctors
    GlobalVector<Scalar> tmp_sigma(m_pdata->getN(), m_exec_conf);

    m_sigma.swap(tmp_sigma);
    TAG_ALLOCATION(m_sigma);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_sigma_dash.get(),
                      sizeof(Scalar3) * m_sigma_dash.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);

        cudaMemAdvise(m_sigma.get(),
                      sizeof(Scalar) * m_sigma.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);
        }
#endif
    }

HelfrichMeshForceCompute::~HelfrichMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying HelfrichMeshForceCompute" << endl;

    delete[] m_K;
    m_K = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void HelfrichMeshForceCompute::setParams(unsigned int type, Scalar K)
    {
    m_K[type] = K;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "helfrich: specified K <= 0" << endl;
    }

void HelfrichMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = helfrich_params(params);
    setParams(typ, _params.k);
    }

pybind11::dict HelfrichMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.helfrich: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in HelfrichMeshForceCompute");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HelfrichMeshForceCompute::computeForces(uint64_t timestep)
    {
    computeSigma(); // precompute sigmas

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);
    assert(h_sigma.data);
    assert(h_sigma_dash.data);

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
    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
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

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];

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

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
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

        Scalar sigma_dash_a2 = 0.5 * dot(sigma_dash_a, sigma_dash_a) * inv_sigma_a * inv_sigma_a;
        Scalar sigma_dash_b2 = 0.5 * dot(sigma_dash_b, sigma_dash_b) * inv_sigma_b * inv_sigma_b;
        Scalar sigma_dash_c2 = 0.5 * dot(sigma_dash_c, sigma_dash_c) * inv_sigma_c * inv_sigma_c;
        Scalar sigma_dash_d2 = 0.5 * dot(sigma_dash_d, sigma_dash_d) * inv_sigma_d * inv_sigma_d;

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

        unsigned int bond_type = m_mesh_data->getMeshBondData()->getTypeByIndex(i);
        Scalar prefactor = m_K[bond_type];

        Fa *= prefactor;

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
            h_force.data[idx_a].w += prefactor * 0.5 * dot(sigma_dash_a, sigma_dash_a) * inv_sigma_a;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fa.x;
            h_force.data[idx_b].y -= Fa.y;
            h_force.data[idx_b].z -= Fa.z;
            h_force.data[idx_b].w += prefactor * 0.5 * dot(sigma_dash_b, sigma_dash_b) * inv_sigma_b;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }
        }
    }

void HelfrichMeshForceCompute::computeSigma()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
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
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);
        unsigned int btag_d = bond.tag[3];
        assert(btag_d < m_pdata->getMaximumTag() + 1);

        if (btag_c == btag_d)
            continue;

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];

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
            this->m_exec_conf->msg->error()
                << "helfrich calculations : triangles (" << idx_a << "," << idx_b << "," << idx_c
                << ") and (" << idx_a << "," << idx_b << "," << idx_d << ") overlap." << std::endl
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
void export_HelfrichMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<HelfrichMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<HelfrichMeshForceCompute>>(m, "HelfrichMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &HelfrichMeshForceCompute::setParamsPython)
        .def("getParams", &HelfrichMeshForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
