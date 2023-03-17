// Copyright (c) 2009-2023 The Regents of the University of Michigan.
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
    : ForceCompute(sysdef), m_K(NULL), m_A0(NULL), m_mesh_data(meshdef), m_area(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing AreaConservationMeshForceCompute" << endl;

    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    // allocate the parameters
    m_K = new Scalar[n_types];

    // allocate the parameters
    m_A0 = new Scalar[n_types];

    m_area = new Scalar[n_types];
    }

AreaConservationMeshForceCompute::~AreaConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying AreaConservationMeshForceCompute" << endl;

    delete[] m_K;
    delete[] m_A0;
    delete[] m_area;
    m_K = NULL;
    m_A0 = NULL;
    m_area = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void AreaConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar A0)
    {
    m_K[type] = K;
    m_A0[type] = A0;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "area: specified K <= 0" << endl;
    if (A0 <= 0)
        m_exec_conf->msg->warning() << "area: specified A0 <= 0" << endl;
    }

void AreaConservationMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = aconstraint_params(params);
    setParams(typ, _params.k, _params.A0);
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
    params["A0"] = m_A0[typ];
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

    ArrayHandle<typename Angle::members_t> h_triangles(
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

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename Angle::members_t& triangle = h_triangles.data[i];

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
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);

        Scalar3 nab, nac;
        nab = dab / rab;
        nac = dac / rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        Scalar inv_s_baac = 1.0 / s_baac;

        Scalar3 dc_dra, dc_drb, dc_drc; // dcos_baac / dr_a
        dc_dra = -nac / rab - nab / rac + c_baac / rab * nab + c_baac / rac * nac;
        dc_drb = nac / rab - c_baac / rab * nab;
        dc_drc = nab / rac - c_baac / rac * nac;

        Scalar3 ds_dra, ds_drb, ds_drc; // dsin_baac / dr_a
        ds_dra = -c_baac * inv_s_baac * dc_dra;
        ds_drb = -c_baac * inv_s_baac * dc_drb;
        ds_drc = -c_baac * inv_s_baac * dc_drc;

        Scalar3 Fa, Fb, Fc;

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);

        Scalar AreaDiff = m_area[triangle_type] - m_A0[triangle_type];

        Scalar energy = m_K[triangle_type] * AreaDiff * AreaDiff
                        / (2 * m_A0[triangle_type] * m_pdata->getNGlobal());

        AreaDiff = -m_K[triangle_type] / m_A0[triangle_type] * AreaDiff / 2.0;

        Fa = AreaDiff * (-nab * rac * s_baac - nac * rab * s_baac + ds_dra * rab * rac);

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

        Fb = AreaDiff * (nab * rac * s_baac + ds_drb * rab * rac);

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

        Fc = AreaDiff * (nac * rab * s_baac + ds_drc * rab * rac);

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

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    const unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    std::vector<Scalar> global_area(n_types);

    for (unsigned int i = 0; i < n_types; i++)
        global_area[i] = 0;

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename Angle::members_t& triangle = h_triangles.data[i];

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
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);

        Scalar3 nab, nac;
        nab = dab / rab;
        nac = dac / rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        Scalar area_tri = rab * rac * s_baac / 2.0;

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);

#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            area_tri /= 3;

            if (idx_a < m_pdata->getN())
                global_area[triangle_type] += area_tri;
            if (idx_b < m_pdata->getN())
                global_area[triangle_type] += area_tri;
            if (idx_c < m_pdata->getN())
                global_area[triangle_type] += area_tri;
            }
        else
#endif
            {
            global_area[triangle_type] += area_tri;
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &global_area[0],
                      n_types,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    for (unsigned int i = 0; i < n_types; i++)
        m_area[i] = global_area[i];
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
