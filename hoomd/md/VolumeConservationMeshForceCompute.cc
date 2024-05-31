// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "VolumeConservationMeshForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file VolumeConservationMeshForceCompute.cc
    \brief Contains code for the VolumeConservationMeshForceCompute class
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
VolumeConservationMeshForceCompute::VolumeConservationMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef,
    bool ignore_type)
    : ForceCompute(sysdef), m_K(NULL), m_V0(NULL), m_mesh_data(meshdef), m_volume(0), 
	m_ignore_type(ignore_type)
    {
    m_exec_conf->msg->notice(5) << "Constructing VolumeConservationMeshForceCompute" << endl;

    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    if(m_ignore_type) n_types = 1;

    // allocate the parameters
    m_K = new Scalar[n_types];

    // allocate the parameters
    m_V0 = new Scalar[n_types];

    m_volume = new Scalar[n_types];
    }

VolumeConservationMeshForceCompute::~VolumeConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying VolumeConservationMeshForceCompute" << endl;

    delete[] m_K;
    delete[] m_V0;
    delete[] m_volume;
    m_K = NULL;
    m_V0 = NULL;
    m_volume = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param V0 desired volume to maintain for the force computation

    Sets parameters for the potential of a particular angle type
*/
void VolumeConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar V0)
    {
    if(!m_ignore_type || type == 0 ) 
    	{
        m_K[type] = K;
        m_V0[type] = V0;

        // check for some silly errors a user could make
        if (K <= 0)
            m_exec_conf->msg->warning() << "volume: specified K <= 0" << endl;
        if (V0 <= 0)
            m_exec_conf->msg->warning() << "volume: specified V0 <= 0" << endl;
	}
    }

void VolumeConservationMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = vconstraint_params(params);
    setParams(typ, _params.k, _params.V0);
    }

pybind11::dict VolumeConservationMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.volume: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in VolumeConservationMeshForceCompute");
        }
    if(m_ignore_type) typ = 0;
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["V0"] = m_V0[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void VolumeConservationMeshForceCompute::computeForces(uint64_t timestep)
    {
    computeVolume(); // precompute volume

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

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

    ArrayHandle<unsigned int> h_pts(m_mesh_data->getPerTypeSize(), access_location::host, access_mode::read);

    Scalar helfrich_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        helfrich_virial[i] = Scalar(0.0);


    unsigned int triN = m_mesh_data->getSize();

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

        vec3<Scalar> pos_a(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        vec3<Scalar> pos_b(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);
        vec3<Scalar> pos_c(h_pos.data[idx_c].x, h_pos.data[idx_c].y, h_pos.data[idx_c].z);

        pos_a = box.shift(pos_a, h_image.data[idx_a]);
        pos_b = box.shift(pos_b, h_image.data[idx_b]);
        pos_c = box.shift(pos_c, h_image.data[idx_c]);

        vec3<Scalar> dVol_a = cross(pos_c, pos_b);

        vec3<Scalar> dVol_b = cross(pos_a, pos_c);

        vec3<Scalar> dVol_c = cross(pos_b, pos_a);

        Scalar3 Fa, Fb, Fc;

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);
  
	if(m_ignore_type) triangle_type = 0;
	else triN = h_pts.data[triangle_type];
  
        Scalar VolDiff = m_volume[triangle_type] - m_V0[triangle_type];

        Scalar energy = m_K[triangle_type] * VolDiff * VolDiff
                        / (6 * m_V0[triangle_type] * triN);

        VolDiff = -m_K[triangle_type] / m_V0[triangle_type] * VolDiff / 6.0;

        Fa.x = VolDiff * dVol_a.x;
        Fa.y = VolDiff * dVol_a.y;
        Fa.z = VolDiff * dVol_a.z;

        if (compute_virial)
            {
            helfrich_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_a].x * Fa.x; // xx
            helfrich_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_a].y * Fa.x; // xy
            helfrich_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.x; // xz
            helfrich_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_a].y * Fa.y; // yy
            helfrich_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.y; // yz
            helfrich_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_a].z * Fa.z; // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w += energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        Fb.x = VolDiff * dVol_b.x;
        Fb.y = VolDiff * dVol_b.y;
        Fb.z = VolDiff * dVol_b.z;

        if (compute_virial)
            {
            helfrich_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_b].x * Fb.x; // xx
            helfrich_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_b].y * Fb.x; // xy
            helfrich_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.x; // xz
            helfrich_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_b].y * Fb.y; // yy
            helfrich_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.y; // yz
            helfrich_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_b].z * Fb.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x += Fb.x;
            h_force.data[idx_b].y += Fb.y;
            h_force.data[idx_b].z += Fb.z;
            h_force.data[idx_b].w += energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }

        Fc.x = VolDiff * dVol_c.x;
        Fc.y = VolDiff * dVol_c.y;
        Fc.z = VolDiff * dVol_c.z;

        if (compute_virial)
            {
            helfrich_virial[0] = Scalar(1. / 2.) * h_pos.data[idx_c].x * Fc.x; // xx
            helfrich_virial[1] = Scalar(1. / 2.) * h_pos.data[idx_c].y * Fc.x; // xy
            helfrich_virial[2] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.x; // xz
            helfrich_virial[3] = Scalar(1. / 2.) * h_pos.data[idx_c].y * Fc.y; // yy
            helfrich_virial[4] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.y; // yz
            helfrich_virial[5] = Scalar(1. / 2.) * h_pos.data[idx_c].z * Fc.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += Fc.x;
            h_force.data[idx_c].y += Fc.y;
            h_force.data[idx_c].z += Fc.z;
            h_force.data[idx_c].w += energy;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += helfrich_virial[j];
            }
        }
    }

void VolumeConservationMeshForceCompute::computeVolume()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    if(m_ignore_type) n_types = 1;

    std::vector<Scalar> global_volume(n_types);
    for (unsigned int i = 0; i < n_types; i++)
        global_volume[i] = 0;

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

        vec3<Scalar> pos_a(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        vec3<Scalar> pos_b(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);
        vec3<Scalar> pos_c(h_pos.data[idx_c].x, h_pos.data[idx_c].y, h_pos.data[idx_c].z);

        pos_a = box.shift(pos_a, h_image.data[idx_a]);
        pos_b = box.shift(pos_b, h_image.data[idx_b]);
        pos_c = box.shift(pos_c, h_image.data[idx_c]);

        Scalar volume_tri = dot(cross(pos_c, pos_b), pos_a) / 6.0;

        unsigned int triangle_type = m_mesh_data->getMeshTriangleData()->getTypeByIndex(i);

	if(m_ignore_type) triangle_type = 0;

#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            volume_tri /= 3;

            if (idx_a < m_pdata->getN())
                global_volume[triangle_type] += volume_tri;
            if (idx_b < m_pdata->getN())
                global_volume[triangle_type] += volume_tri;
            if (idx_c < m_pdata->getN())
                global_volume[triangle_type] += volume_tri;
            }
        else
#endif
            {
            global_volume[triangle_type] += volume_tri;
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &global_volume[0],
                      n_types,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    for (unsigned int i = 0; i < n_types; i++)
        m_volume[i] = global_volume[i];
    }

namespace detail
    {
void export_VolumeConservationMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<VolumeConservationMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<VolumeConservationMeshForceCompute>>(
        m,
        "VolumeConservationMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>, bool>())
        .def("setParams", &VolumeConservationMeshForceCompute::setParamsPython)
        .def("getParams", &VolumeConservationMeshForceCompute::getParams)
        .def("getVolume", &VolumeConservationMeshForceCompute::getVolume);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
