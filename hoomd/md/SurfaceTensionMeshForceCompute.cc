// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "SurfaceTensionMeshForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file MeshSurfaceTensionForceCompute.cc
    \brief Contains code for the MeshSurfaceTensionForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
SurfaceTensionMeshForceCompute::SurfaceTensionMeshForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_sigma(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing SurfaceTensionMeshForceCompute" << endl;

    // allocate the parameters
    m_sigma = new Scalar[m_pdata->getNTypes()];
    }

SurfaceTensionMeshForceCompute::~SurfaceTensionMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying SurfaceTensionMeshForceCompute" << endl;

    delete[] m_sigma;
    m_sigma = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param sigma surface tension parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void SurfaceTensionMeshForceCompute::setParams(unsigned int type, Scalar sigma)
    {
    m_sigma[type] = sigma;

    // check for some silly errors a user could make
    if (sigma <= 0)
        m_exec_conf->msg->warning() << "SurfaceTension: specified sigma <= 0" << endl;
    }

void SurfaceTensionMeshForceCompute::setParamsPython(std::string type,
                                                               pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    auto _params = surface_tension_params(params);
    setParams(typ, _params.sigma);
    }

pybind11::dict SurfaceTensionMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshTriangleData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.surface_tension: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in SurfaceTensionMeshForceCompute");
        }
    pybind11::dict params;
    params["sigma"] = m_sigma[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void SurfaceTensionMeshForceCompute::computeForces(uint64_t timestep)
    {
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

    Scalar surface_tension_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        surface_tension_virial[i] = Scalar(0.0);

    // for each of the triangles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();

    // from whole surface area A_mesh to the surface of individual triangle A_mesh -> At

    m_area = 0;

    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the triangle
        const typename Angle::members_t& triangle = h_triangles.data[i];
        assert(triangle.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[1] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[2] < m_pdata->getMaximumTag() + 1);

        // transform a, b, and c into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[triangle.tag[0]];
        unsigned int idx_b = h_rtag.data[triangle.tag[1]];
        unsigned int idx_c = h_rtag.data[triangle.tag[2]];

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

        Scalar3 da, db, dc;
        da.x = h_pos.data[idx_a].x;
        da.y = h_pos.data[idx_a].y;
        da.z = h_pos.data[idx_a].z;
        db.x = h_pos.data[idx_b].x;
        db.y = h_pos.data[idx_b].y;
        db.z = h_pos.data[idx_b].z;
        dc.x = h_pos.data[idx_c].x;
        dc.y = h_pos.data[idx_c].y;
        dc.z = h_pos.data[idx_c].z;

        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;

        Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dab.z * dac.z;

        Scalar area2 = sqrt(rsqab * rsqac - rabrac * rabrac);

        Scalar prefactor = -m_sigma[0] / (2*area2);

        Scalar energy_pp = m_sigma[0] * area2 / 2;

        m_area += area2 / 2;

        Scalar3 Fa, Fb, Fc;
        Fa = prefactor * ((rabrac - rsqac) * dab + (rabrac - rsqab) * dac);
        Fb = prefactor * (rsqac * dab - rabrac * dac);
        Fc = prefactor * (rsqab * dac - rabrac * dab);

        if (compute_virial)
            {
            surface_tension_virial[0] = Scalar(1. / 2.) * da.x * Fa.x; // xx
            surface_tension_virial[1] = Scalar(1. / 2.) * da.y * Fa.x; // xy
            surface_tension_virial[2] = Scalar(1. / 2.) * da.z * Fa.x; // xz
            surface_tension_virial[3] = Scalar(1. / 2.) * da.y * Fa.y; // yy
            surface_tension_virial[4] = Scalar(1. / 2.) * da.z * Fa.y; // yz
            surface_tension_virial[5] = Scalar(1. / 2.) * da.z * Fa.z; // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w += energy_pp; // divided by 3 because of three
                                                // particles sharing the energy

            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += surface_tension_virial[j];
            }

        if (compute_virial)
            {
            surface_tension_virial[0] = Scalar(1. / 2.) * db.x * Fb.x; // xx
            surface_tension_virial[1] = Scalar(1. / 2.) * db.y * Fb.x; // xy
            surface_tension_virial[2] = Scalar(1. / 2.) * db.z * Fb.x; // xz
            surface_tension_virial[3] = Scalar(1. / 2.) * db.y * Fb.y; // yy
            surface_tension_virial[4] = Scalar(1. / 2.) * db.z * Fb.y; // yz
            surface_tension_virial[5] = Scalar(1. / 2.) * db.z * Fb.z; // zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x += Fb.x;
            h_force.data[idx_b].y += Fb.y;
            h_force.data[idx_b].z += Fb.z;
            h_force.data[idx_b].w += energy_pp;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += surface_tension_virial[j];
            }

        if (compute_virial)
            {
            surface_tension_virial[0] = Scalar(1. / 2.) * dc.x * Fc.x; // xx
            surface_tension_virial[1] = Scalar(1. / 2.) * dc.y * Fc.x; // xchy
            surface_tension_virial[2] = Scalar(1. / 2.) * dc.z * Fc.x; // xz
            surface_tension_virial[3] = Scalar(1. / 2.) * dc.y * Fc.y; // yy
            surface_tension_virial[4] = Scalar(1. / 2.) * dc.z * Fc.y; // yz
            surface_tension_virial[5] = Scalar(1. / 2.) * dc.z * Fc.z; // zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += Fc.x;
            h_force.data[idx_c].y += Fc.y;
            h_force.data[idx_c].z += Fc.z;
            h_force.data[idx_c].w += energy_pp;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += surface_tension_virial[j];
            }
        }
    }

Scalar SurfaceTensionMeshForceCompute::energyDiff(unsigned int idx_a,
                                                            unsigned int idx_b,
                                                            unsigned int idx_c,
                                                            unsigned int idx_d,
                                                            unsigned int type_id)
    {
    return 0;
    }

namespace detail
    {
void export_SurfaceTensionMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<SurfaceTensionMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<SurfaceTensionMeshForceCompute>>(
        m,
        "SurfaceTensionMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &SurfaceTensionMeshForceCompute::setParamsPython)
        .def("getParams", &SurfaceTensionMeshForceCompute::getParams)
        .def("getArea", &SurfaceTensionMeshForceCompute::getArea);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
