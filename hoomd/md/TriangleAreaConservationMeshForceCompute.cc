// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TriangleAreaConservationMeshForceCompute.h"

#include <iostream>
#include <stdexcept>

using namespace std;

/*! \file AngleAreaConservationForceCompute.cc
    \brief Contains code for the AngleAreaConservationForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
TriangleAreaConservationMeshForceCompute::TriangleAreaConservationMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing TriangleAreaConservationhMeshForceCompute" << endl;

    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    GPUArray<Scalar2> params(n_types, m_exec_conf);
    m_params.swap(params);

    m_area = new Scalar[n_types];
    }

TriangleAreaConservationMeshForceCompute::~TriangleAreaConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying TriangleAreaConservationMeshForceCompute" << endl;

    delete[] m_area;
    m_area = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param A0 desired surface area to maintain for the force computation

    Sets parameters for the potential of a particular mesh type
*/
void TriangleAreaConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar A0)
    {
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, A0);

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "TriangleAreaConservation: specified K <= 0" << endl;

    if (A0 <= 0)
        m_exec_conf->msg->warning() << "TriangleAreaConservation: specified A0 <= 0" << endl;
    }

void TriangleAreaConservationMeshForceCompute::setParamsPython(std::string type,
                                                               pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    auto _params = triangle_area_conservation_params(params);
    setParams(typ, _params.k, _params.A0);
    }

pybind11::dict TriangleAreaConservationMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshTriangleData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.conservation.TriangleArea: Invalid mesh type specified"
                                  << endl;
        throw runtime_error("Error setting parameters in TriangleAreaConservationMeshForceCompute");
        }
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    pybind11::dict params;
    params["k"] = h_params.data[typ].x;
    params["A0"] = h_params.data[typ].y;
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void TriangleAreaConservationMeshForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_triangles.data);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box
    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar triangle_area_conservation_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        triangle_area_conservation_virial[i] = Scalar(0.0);

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

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);

        Scalar At = h_params.data[triangle_type].y;

        Scalar tri_area = rab * rac * s_baac / 6; // triangle area/3
        Scalar Ut = 3 * tri_area - At;

        Scalar prefactor = h_params.data[triangle_type].x / (2 * At) * Ut;

        Scalar3 Fab, Fac;
        Fab = prefactor * (-nab * rac * s_baac + ds_drab * rab * rac);
        Fac = prefactor * (-nac * rab * s_baac + ds_drac * rab * rac);

        if (compute_virial)
            {
            triangle_area_conservation_virial[0]
                = Scalar(1. / 2.) * (dab.x * Fab.x + dac.x * Fac.x); // xx
            triangle_area_conservation_virial[1]
                = Scalar(1. / 2.) * (dab.y * Fab.x + dac.y * Fac.x); // xy
            triangle_area_conservation_virial[2]
                = Scalar(1. / 2.) * (dab.z * Fab.x + dac.z * Fac.x); // xz
            triangle_area_conservation_virial[3]
                = Scalar(1. / 2.) * (dab.y * Fab.y + dac.y * Fac.y); // yy
            triangle_area_conservation_virial[4]
                = Scalar(1. / 2.) * (dab.z * Fab.y + dac.z * Fac.y); // yz
            triangle_area_conservation_virial[5]
                = Scalar(1. / 2.) * (dab.z * Fab.z + dac.z * Fac.z); // zz
            }

        if (idx_a < m_pdata->getN())
            {
            global_area[triangle_type] += tri_area;

            h_force.data[idx_a].x += (Fab.x + Fac.x);
            h_force.data[idx_a].y += (Fab.y + Fac.y);
            h_force.data[idx_a].z += (Fab.z + Fac.z);
            h_force.data[idx_a].w += h_params.data[triangle_type].x / (6.0 * At) * Ut
                                     * Ut; // divided by 3 because of three
                                           // particles sharing the energy

            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += triangle_area_conservation_virial[j];
            }

        if (compute_virial)
            {
            triangle_area_conservation_virial[0] = Scalar(1. / 2.) * dab.x * Fab.x; // xx
            triangle_area_conservation_virial[1] = Scalar(1. / 2.) * dab.y * Fab.x; // xy
            triangle_area_conservation_virial[2] = Scalar(1. / 2.) * dab.z * Fab.x; // xz
            triangle_area_conservation_virial[3] = Scalar(1. / 2.) * dab.y * Fab.y; // yy
            triangle_area_conservation_virial[4] = Scalar(1. / 2.) * dab.z * Fab.y; // yz
            triangle_area_conservation_virial[5] = Scalar(1. / 2.) * dab.z * Fab.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            global_area[triangle_type] += tri_area;

            h_force.data[idx_b].x -= Fab.x;
            h_force.data[idx_b].y -= Fab.y;
            h_force.data[idx_b].z -= Fab.z;
            h_force.data[idx_b].w += h_params.data[triangle_type].x / (6.0 * At) * Ut * Ut;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += triangle_area_conservation_virial[j];
            }

        if (compute_virial)
            {
            triangle_area_conservation_virial[0] = Scalar(1. / 2.) * dac.x * Fac.x; // xx
            triangle_area_conservation_virial[1] = Scalar(1. / 2.) * dac.y * Fac.x; // xchy
            triangle_area_conservation_virial[2] = Scalar(1. / 2.) * dac.z * Fac.x; // xz
            triangle_area_conservation_virial[3] = Scalar(1. / 2.) * dac.y * Fac.y; // yy
            triangle_area_conservation_virial[4] = Scalar(1. / 2.) * dac.z * Fac.y; // yz
            triangle_area_conservation_virial[5] = Scalar(1. / 2.) * dac.z * Fac.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            global_area[triangle_type] += tri_area;

            h_force.data[idx_c].x -= Fac.x;
            h_force.data[idx_c].y -= Fac.y;
            h_force.data[idx_c].z -= Fac.z;
            h_force.data[idx_c].w += h_params.data[triangle_type].x / (6.0 * At) * Ut * Ut;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += triangle_area_conservation_virial[j];
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
void export_TriangleAreaConservationMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<TriangleAreaConservationMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<TriangleAreaConservationMeshForceCompute>>(
        m,
        "TriangleAreaConservationMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &TriangleAreaConservationMeshForceCompute::setParamsPython)
        .def("getParams", &TriangleAreaConservationMeshForceCompute::getParams)
        .def("getArea", &TriangleAreaConservationMeshForceCompute::getArea);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
