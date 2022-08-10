// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "SurfaceTensionMeshForceCompute2D.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file SurfaceTensionMeshForceCompute2D.cc
    \brief Contains code for the SurfaceTensionMeshForceCompute2D class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
SurfaceTensionMeshForceCompute2D::SurfaceTensionMeshForceCompute2D(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_sigma(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing SurfaceTensionMeshForceCompute2D" << endl;

    // allocate the parameters
    m_sigma = new Scalar[m_pdata->getNTypes()];
    }

SurfaceTensionMeshForceCompute2D::~SurfaceTensionMeshForceCompute2D()
    {
    m_exec_conf->msg->notice(5) << "Destroying SurfaceTensionMeshForceCompute2D" << endl;

    delete[] m_sigma;
    m_sigma = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param sigma surface tension parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void SurfaceTensionMeshForceCompute2D::setParams(unsigned int type, Scalar sigma)
    {
    m_sigma[type] = sigma;

    // check for some silly errors a user could make
    if (sigma <= 0)
        m_exec_conf->msg->warning() << "SurfaceTension: specified sigma <= 0" << endl;
    }

void SurfaceTensionMeshForceCompute2D::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = surface_tension_params(params);
    setParams(typ, _params.sigma);
    }

pybind11::dict SurfaceTensionMeshForceCompute2D::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);

    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.surface_tension2D: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in SurfaceTensionMeshForceCompute2D");
        }
    pybind11::dict params;
    params["sigma"] = m_sigma[typ];
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void SurfaceTensionMeshForceCompute2D::computeForces(uint64_t timestep)
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

    Scalar surface_tension_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        surface_tension_virial[i] = Scalar(0.0);

    // for each of the triangles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();

    // from whole surface area A_mesh to the surface of individual triangle A_mesh -> At

    m_circumference = 0;

    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a, b, and c into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());

        if (bond.tag[2] == bond.tag[3])
            {
            // calculate d\vec{r}
            Scalar3 dab;
            dab.x = h_pos.data[idx_b].x - h_pos.data[idx_a].x;
            dab.y = h_pos.data[idx_b].y - h_pos.data[idx_a].y;
            dab.z = h_pos.data[idx_b].z - h_pos.data[idx_a].z;

            // apply minimum image conventions to all 2 vectors
            dab = box.minImage(dab);

            Scalar3 da, db, dc;
            da.x = h_pos.data[idx_a].x;
            da.y = h_pos.data[idx_a].y;
            da.z = h_pos.data[idx_a].z;
            db.x = h_pos.data[idx_b].x;
            db.y = h_pos.data[idx_b].y;
            db.z = h_pos.data[idx_b].z;

            // FLOPS: 14 / MEM TRANSFER: 2 Scalars

            // FLOPS: 42 / MEM TRANSFER: 6 Scalars
            Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;

            Scalar rab = sqrt(rsqab);

            Scalar energy_pp = m_sigma[0] * rab / 2;

            m_circumference += rab;

            Scalar3 Fa, Fb;
            Fa = m_sigma[0] / rab * dab;
            Fb = -Fa;

            if (compute_virial)
                {
                surface_tension_virial[0] = Scalar(1. / 2.) * da.x * Fa.x; // xx
                surface_tension_virial[1] = Scalar(1. / 2.) * da.y * Fa.x; // xy
                surface_tension_virial[2] = Scalar(1. / 2.) * da.z * Fa.x; // xz
                surface_tension_virial[3] = Scalar(1. / 2.) * da.y * Fa.y; // yy
                surface_tension_virial[4] = Scalar(1. / 2.) * da.z * Fa.y; // yz
                surface_tension_virial[5] = Scalar(1. / 2.) * da.z * Fa.z; // zz
                }
            // Now, apply the force to each individual atom a,b, and accumulate the energy/virial
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
            }
        }
    }

Scalar SurfaceTensionMeshForceCompute2D::energyDiff(unsigned int idx_a,
                                                    unsigned int idx_b,
                                                    unsigned int idx_c,
                                                    unsigned int idx_d,
                                                    unsigned int type_id)
    {
    return 0;
    }

namespace detail
    {
void export_SurfaceTensionMeshForceCompute2D(pybind11::module& m)
    {
    pybind11::class_<SurfaceTensionMeshForceCompute2D,
                     ForceCompute,
                     std::shared_ptr<SurfaceTensionMeshForceCompute2D>>(
        m,
        "SurfaceTensionMeshForceCompute2D")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &SurfaceTensionMeshForceCompute2D::setParamsPython)
        .def("getParams", &SurfaceTensionMeshForceCompute2D::getParams)
        .def("getCircumference", &SurfaceTensionMeshForceCompute2D::getCircumference);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
