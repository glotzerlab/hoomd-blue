// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
BendingRigidityMeshForceCompute::BendingRigidityMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing BendingRigidityMeshForceCompute" << endl;

    // allocate the parameters
    m_K = new Scalar[m_mesh_data->getMeshBondData()->getNTypes()];
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

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);

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

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);

        Scalar3 z1;
        z1.x = dab.y * dac.z - dab.z * dac.y;
        z1.y = dab.z * dac.x - dab.x * dac.z;
        z1.z = dab.x * dac.y - dab.y * dac.x;

        Scalar3 z2;
        z2.x = dad.y * dab.z - dad.z * dab.y;
        z2.y = dad.z * dab.x - dad.x * dab.z;
        z2.z = dad.x * dab.y - dad.y * dab.x;

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar n1 = fast::rsqrt(z1.x * z1.x + z1.y * z1.y + z1.z * z1.z);
        Scalar n2 = fast::rsqrt(z2.x * z2.x + z2.y * z2.y + z2.z * z2.z);
        Scalar z1z2 = z1.x * z2.x + z1.y * z2.y + z1.z * z2.z;

        Scalar cosinus = z1z2 * n1 * n2;

        Scalar3 A1 = n1 * n2 * z2 - cosinus * n1 * n1 * z1;
        Scalar3 A2 = n1 * n2 * z1 - cosinus * n2 * n2 * z2;

        Scalar3 Fab, Fac, Fad;
        Fab.x = -A1.y * dac.z + A1.z * dac.y + A2.y * dad.z - A2.z * dad.y;
        Fab.y = A1.x * dac.z - A1.z * dac.x - A2.x * dad.z + A2.z * dad.x;
        Fab.z = -A1.x * dac.y + A1.y * dac.x + A2.x * dad.y - A2.y * dad.x;

        Fac.x = A1.y * dab.z - A1.z * dab.y;
        Fac.y = -A1.x * dab.z + A1.z * dab.x;
        Fac.z = A1.x * dab.y - A1.y * dab.x;

        Fad.x = -A2.y * dab.z + A2.z * dab.y;
        Fad.y = A2.x * dab.z - A2.z * dab.x;
        Fad.z = -A2.x * dab.y + A2.y * dab.x;

        unsigned int meshbond_type = m_mesh_data->getMeshBondData()->getTypeByIndex(i);

        Scalar prefactor = 0.5 * m_K[meshbond_type];

        Scalar prefactor_4 = 0.25 * prefactor;

        Fab = prefactor * Fab;

        Fac = prefactor * Fac;

        Fad = prefactor * Fad;

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
                += prefactor_4 * (1 - cosinus); // the missing minus sign comes from the fact that
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
            h_force.data[idx_b].w += prefactor_4 * (1 - cosinus);
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
            h_force.data[idx_c].w += prefactor_4 * (1 - cosinus);
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
            h_force.data[idx_d].w += prefactor_4 * (1 - cosinus);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_d] += rigidity_virial[j];
            }
        }
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
