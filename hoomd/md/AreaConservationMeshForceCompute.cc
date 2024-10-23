// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AreaConservationMeshForceCompute.h"

#include <iostream>
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
    \param meshdef Mesh triangulation
    \param ignore_type boolean whether to ignore types
    \post Memory is allocated, and forces are zeroed.
*/
AreaConservationMeshForceCompute::AreaConservationMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef,
    bool ignore_type)
    : ForceCompute(sysdef), m_mesh_data(meshdef), m_ignore_type(ignore_type)
    {
    m_exec_conf->msg->notice(5) << "Constructing AreaConservationMeshForceCompute" << endl;

    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    if (m_ignore_type)
        n_types = 1;

    GPUArray<Scalar2> params(n_types, m_exec_conf);
    m_params.swap(params);

    GPUArray<Scalar> area(n_types, m_exec_conf);
    m_area.swap(area);
    }

AreaConservationMeshForceCompute::~AreaConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying AreaConservationMeshForceCompute" << endl;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param A0 desired surface area to maintain for the force computation

    Sets parameters for the potential of a particular mesh type
*/
void AreaConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar A0)
    {
    if (!m_ignore_type || type == 0)
        {
        ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
        // update the local copy of the memory
        h_params.data[type] = make_scalar2(K, A0);

        // check for some silly errors a user could make
        if (K <= 0)
            m_exec_conf->msg->warning() << "area: specified K <= 0" << endl;
        if (A0 <= 0)
            m_exec_conf->msg->warning() << "area: specified A0 <= 0" << endl;
        }
    }

void AreaConservationMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = area_conservation_params(params);
    setParams(typ, _params.k, _params.A0);
    }

pybind11::dict AreaConservationMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.conservation.Area: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in AreaConservationMeshForceCompute");
        }
    if (m_ignore_type)
        typ = 0;
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    pybind11::dict params;
    params["k"] = h_params.data[typ].x;
    params["A0"] = h_params.data[typ].y;
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void AreaConservationMeshForceCompute::computeForces(uint64_t timestep)
    {
    precomputeParameter(); // precompute area

    assert(m_pdata);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_area(m_area, access_location::host, access_mode::read);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_triangles.data);

    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<unsigned int> h_pts(m_mesh_data->getPerTypeSize(),
                                    access_location::host,
                                    access_mode::read);

    Scalar area_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        area_virial[i] = Scalar(0.0);

    unsigned int triN = m_mesh_data->getSize();

    // loop over mesh triangles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
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

        Scalar3 dc_drab, dc_drac; // dcos_baac / dr_a
        dc_drab = -nac / rab + c_baac / rab * nab;
        dc_drac = -nab / rac + c_baac / rac * nac;

        Scalar3 ds_drab, ds_drac; // dsin_baac / dr_a
        ds_drab = -c_baac * inv_s_baac * dc_drab;
        ds_drac = -c_baac * inv_s_baac * dc_drac;

        Scalar3 Fab, Fac;

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);

        if (m_ignore_type)
            triangle_type = 0;
        else
            triN = h_pts.data[triangle_type];

        Scalar AreaDiff = h_area.data[triangle_type] - h_params.data[triangle_type].y;

        Scalar energy = h_params.data[triangle_type].x * AreaDiff * AreaDiff
                        / (6 * h_params.data[triangle_type].y * triN);

        AreaDiff = h_params.data[triangle_type].x / h_params.data[triangle_type].y * AreaDiff / 2.0;

        Fab = AreaDiff * (-nab * rac * s_baac + ds_drab * rab * rac);
        Fac = AreaDiff * (-nac * rab * s_baac + ds_drac * rab * rac);

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * (dab.x * Fab.x + dac.x * Fac.x); // xx
            area_virial[1] = Scalar(1. / 2.) * (dab.y * Fab.x + dac.y * Fac.x); // xy
            area_virial[2] = Scalar(1. / 2.) * (dab.z * Fab.x + dac.z * Fac.x); // xz
            area_virial[3] = Scalar(1. / 2.) * (dab.y * Fab.y + dac.y * Fac.y); // yy
            area_virial[4] = Scalar(1. / 2.) * (dab.z * Fab.y + dac.z * Fac.y); // yz
            area_virial[5] = Scalar(1. / 2.) * (dab.z * Fab.z + dac.z * Fac.z); // zz
            }

        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += (Fab.x + Fac.x);
            h_force.data[idx_a].y += (Fab.y + Fac.y);
            h_force.data[idx_a].z += (Fab.z + Fac.z);
            h_force.data[idx_a].w += energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += area_virial[j];
            }

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * dab.x * Fab.x; // xx
            area_virial[1] = Scalar(1. / 2.) * dab.y * Fab.x; // xy
            area_virial[2] = Scalar(1. / 2.) * dab.z * Fab.x; // xz
            area_virial[3] = Scalar(1. / 2.) * dab.y * Fab.y; // yy
            area_virial[4] = Scalar(1. / 2.) * dab.z * Fab.y; // yz
            area_virial[5] = Scalar(1. / 2.) * dab.z * Fab.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fab.x;
            h_force.data[idx_b].y -= Fab.y;
            h_force.data[idx_b].z -= Fab.z;
            h_force.data[idx_b].w += energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += area_virial[j];
            }

        if (compute_virial)
            {
            area_virial[0] = Scalar(1. / 2.) * dac.x * Fac.x; // xx
            area_virial[1] = Scalar(1. / 2.) * dac.y * Fac.x; // xy
            area_virial[2] = Scalar(1. / 2.) * dac.z * Fac.x; // xz
            area_virial[3] = Scalar(1. / 2.) * dac.y * Fac.y; // yy
            area_virial[4] = Scalar(1. / 2.) * dac.z * Fac.y; // yz
            area_virial[5] = Scalar(1. / 2.) * dac.z * Fac.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x -= Fac.x;
            h_force.data[idx_c].y -= Fac.y;
            h_force.data[idx_c].z -= Fac.z;
            h_force.data[idx_c].w += energy;
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

    const BoxDim& box = m_pdata->getGlobalBox();

    const unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    std::vector<Scalar> global_area(n_types);

    for (unsigned int i = 0; i < n_types; i++)
        global_area[i] = 0;

    // loop over mesh triangles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
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
        dab.x = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
        dab.y = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
        dab.z = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_c].x - h_pos.data[idx_a].x;
        dac.y = h_pos.data[idx_c].y - h_pos.data[idx_a].y;
        dac.z = h_pos.data[idx_c].z - h_pos.data[idx_a].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

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

        if (m_ignore_type)
            triangle_type = 0;

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

    ArrayHandle<Scalar> h_area(m_area, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < n_types; i++)
        h_area.data[i] = global_area[i];
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
        .def(pybind11::
                 init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>, bool>())
        .def("setParams", &AreaConservationMeshForceCompute::setParamsPython)
        .def("getParams", &AreaConservationMeshForceCompute::getParams)
        .def("getArea", &AreaConservationMeshForceCompute::getArea);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
