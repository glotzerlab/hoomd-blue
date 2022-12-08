// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BendingRigidityMeshForceCompute.h"

#include <float.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file BendingRigidityMeshForceCompute.cc
    \brief Contains code for the BendingRigidityMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
BendingRigidityMeshForceCompute::BendingRigidityMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing BendingRigidityMeshForceCompute" << endl;

    // allocate the parameters
    m_K = new Scalar[m_pdata->getNTypes()];
    }

BendingRigidityMeshForceCompute::~BendingRigidityMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying BendingRigidityMeshForceCompute" << endl;

    delete[] m_K;
    m_K = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void BendingRigidityMeshForceCompute::setParams(unsigned int type, Scalar K)
    {
    m_K[type] = K;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "rigidity: specified K <= 0" << endl;
    }

void BendingRigidityMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = rigidity_params(params);
    setParams(typ, _params.k);
    }

pybind11::dict BendingRigidityMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.rigidity: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in BendingRigidityMeshForceCompute");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void BendingRigidityMeshForceCompute::computeForces(uint64_t timestep)
    {
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
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);
    assert(h_triangles.data);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar rigidity_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        rigidity_virial[i] = Scalar(0.0);

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

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rqab = rsqab * rsqab;
        Scalar rcab = fast::sqrt(rsqab) * rsqab;
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;

        Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dac.z * dac.z;
        Scalar rabracsq = rabrac * rabrac;
        Scalar rabrad = dab.x * dad.x + dab.y * dad.y + dad.z * dad.z;
        Scalar rabradsq = rabrad * rabrad;
        Scalar racrad = dac.x * dad.x + dac.y * dad.y + dad.z * dad.z;

        Scalar numerator = rsqab * racrad - rabrac * rabrad;

        Scalar3 numerator_ab = 2 * racrad * dab - rabrac * dad - rabrad * dac;
        Scalar3 numerator_ac = rsqab * dad - rabrad * dab;
        Scalar3 numerator_ad = rsqab * dac - rabrac * dab;

        Scalar inv_d = fast::rsqrt(rsqac * rsqad * rqab - rabracsq * rsqab * rsqad
                                   - rabradsq * rsqab * rsqac + rabracsq * rabradsq);
        Scalar inv_d3 = inv_d * inv_d * inv_d;

        Scalar3 denominator_ab
            = dab * (2 * rsqac * rsqad * rcab - rabracsq * rsqad - rabradsq * rsqac)
              + dac * (rabrac * rabradsq - rabrac * rsqab * rsqad)
              + dad * (rabracsq * rabrad - rabrad * rsqab * rsqac);
        denominator_ab = -denominator_ab * inv_d3;
        Scalar3 denominator_ac = (rsqad * rqab - rabradsq * rsqab) * dac
                                 + (rabrac * rabradsq - rabrac * rsqab * rsqad) * dab;
        denominator_ac = -denominator_ac * inv_d3;
        Scalar3 denominator_ad = (rsqac * rqab - rabracsq * rsqab) * dad
                                 + (rabrad * rabracsq - rabrad * rsqab * rsqac) * dab;
        denominator_ad = -denominator_ad * inv_d3;

        Scalar cosinus = numerator * inv_d;

        Scalar3 cosinus_ab, cosinus_ac, cosinus_ad;

        cosinus_ab.x = numerator_ab.x * inv_d + numerator * denominator_ab.x;
        cosinus_ab.y = numerator_ab.y * inv_d + numerator * denominator_ab.y;
        cosinus_ab.z = numerator_ab.z * inv_d + numerator * denominator_ab.z;

        cosinus_ac.x = numerator_ac.x * inv_d + numerator * denominator_ac.x;
        cosinus_ac.y = numerator_ac.y * inv_d + numerator * denominator_ac.y;
        cosinus_ac.z = numerator_ac.z * inv_d + numerator * denominator_ac.z;

        cosinus_ad.x = numerator_ad.x * inv_d + numerator * denominator_ad.x;
        cosinus_ad.y = numerator_ad.y * inv_d + numerator * denominator_ad.y;
        cosinus_ad.z = numerator_ad.z * inv_d + numerator * denominator_ad.z;

        Scalar3 Fab, Fac, Fad;

        Fab.x = m_K[0] * cosinus_ab.x;
        Fab.y = m_K[0] * cosinus_ab.y;
        Fab.z = m_K[0] * cosinus_ab.z;

        Fac.x = m_K[0] * cosinus_ac.x;
        Fac.y = m_K[0] * cosinus_ac.y;
        Fac.z = m_K[0] * cosinus_ac.z;

        Fad.x = m_K[0] * cosinus_ad.x;
        Fad.y = m_K[0] * cosinus_ad.y;
        Fad.z = m_K[0] * cosinus_ad.z;

        if (compute_virial)
            {
            rigidity_virial[0]
                = Scalar(1. / 2.) * (dab.x * Fab.x + dac.x * Fac.x + dad.x * Fad.x); // xx
            rigidity_virial[1]
                = Scalar(1. / 2.) * (dab.y * Fab.x + dac.y * Fac.x + dad.y * Fad.x); // xy
            rigidity_virial[2]
                = Scalar(1. / 2.) * (dab.z * Fab.x + dac.z * Fac.x + dad.z * Fad.x); // xz
            rigidity_virial[3]
                = Scalar(1. / 2.) * (dab.y * Fab.y + dac.y * Fac.y + dad.y * Fad.y); // yy
            rigidity_virial[4]
                = Scalar(1. / 2.) * (dab.z * Fab.y + dac.z * Fac.y + dad.z * Fad.y); // yz
            rigidity_virial[5]
                = Scalar(1. / 2.) * (dab.z * Fab.z + dac.z * Fac.z + dad.z * Fad.z); // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += (Fab.x + Fac.x + Fad.x);
            h_force.data[idx_a].y += (Fab.y + Fac.y + Fad.y);
            h_force.data[idx_a].z += (Fab.z + Fac.z + Fad.z);
            h_force.data[idx_a].w
                = m_K[0] * 0.25 * (1 + cosinus); // the missing minus sign comes from the fact that
                                                 // we have to compare the normal directions
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += rigidity_virial[j];
            }

        if (compute_virial)
            {
            rigidity_virial[0] = Scalar(1. / 2.) * dab.x * Fab.x; // xx
            rigidity_virial[1] = Scalar(1. / 2.) * dab.y * Fab.x; // xy
            rigidity_virial[2] = Scalar(1. / 2.) * dab.z * Fab.x; // xz
            rigidity_virial[3] = Scalar(1. / 2.) * dab.y * Fab.y; // yy
            rigidity_virial[4] = Scalar(1. / 2.) * dab.z * Fab.y; // yz
            rigidity_virial[5] = Scalar(1. / 2.) * dab.z * Fab.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fab.x;
            h_force.data[idx_b].y -= Fab.y;
            h_force.data[idx_b].z -= Fab.z;
            h_force.data[idx_b].w = m_K[0] * 0.25 * (1 + cosinus);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += rigidity_virial[j];
            }

        if (compute_virial)
            {
            rigidity_virial[0] = Scalar(1. / 2.) * dac.x * Fac.x; // xx
            rigidity_virial[1] = Scalar(1. / 2.) * dac.y * Fac.x; // xy
            rigidity_virial[2] = Scalar(1. / 2.) * dac.z * Fac.x; // xz
            rigidity_virial[3] = Scalar(1. / 2.) * dac.y * Fac.y; // yy
            rigidity_virial[4] = Scalar(1. / 2.) * dac.z * Fac.y; // yz
            rigidity_virial[5] = Scalar(1. / 2.) * dac.z * Fac.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x -= Fac.x;
            h_force.data[idx_c].y -= Fac.y;
            h_force.data[idx_c].z -= Fac.z;
            h_force.data[idx_c].w = m_K[0] * 0.25 * (1 + cosinus);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += rigidity_virial[j];
            }

        if (compute_virial)
            {
            rigidity_virial[0] = Scalar(1. / 2.) * dad.x * Fad.x; // xx
            rigidity_virial[1] = Scalar(1. / 2.) * dad.y * Fad.x; // xy
            rigidity_virial[2] = Scalar(1. / 2.) * dad.z * Fad.x; // xz
            rigidity_virial[3] = Scalar(1. / 2.) * dad.y * Fad.y; // yy
            rigidity_virial[4] = Scalar(1. / 2.) * dad.z * Fad.y; // yz
            rigidity_virial[5] = Scalar(1. / 2.) * dad.z * Fad.z; // zz
            }

        if (idx_d < m_pdata->getN())
            {
            h_force.data[idx_d].x -= Fad.x;
            h_force.data[idx_d].y -= Fad.y;
            h_force.data[idx_d].z -= Fad.z;
            h_force.data[idx_d].w = m_K[0] * 0.25 * (1 + cosinus);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_d] += rigidity_virial[j];
            }
        }
    }

Scalar BendingRigidityMeshForceCompute::energyDiff(unsigned int idx_a,
                                                   unsigned int idx_b,
                                                   unsigned int idx_c,
                                                   unsigned int idx_d,
                                                   unsigned int type_id)
    {
    return 0;
    }

namespace detail
    {
void export_BendingRigidityMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<BendingRigidityMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<BendingRigidityMeshForceCompute>>(
        m,
        "BendingRigidityMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &BendingRigidityMeshForceCompute::setParamsPython)
        .def("getParams", &BendingRigidityMeshForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
