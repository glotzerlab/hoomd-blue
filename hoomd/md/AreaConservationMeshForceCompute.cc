// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: dnlebard

#include "AreaConservationMeshForceCompute.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file MeshAreaConservationForceCompute.cc
    \brief Contains code for the MeshAreaConservationForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
AreaConservationMeshForceCompute::AreaConservationMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_K(NULL), m_A0(NULL), m_mesh_data(meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing AreaConservationhMeshForceCompute" << endl;

    // allocate the parameters
    m_K = new Scalar[m_pdata->getNTypes()];
    m_A0 = new Scalar[m_pdata->getNTypes()];

    // // allocate memory for the per-type normal verctors
    // GlobalVector<Scalar3> tmp_sigma_dash(m_pdata->getN(), m_exec_conf);

    // m_sigma_dash.swap(tmp_sigma_dash);
    // TAG_ALLOCATION(m_sigma_dash);

    // allocate memory for the per-type normal verctors
    // GlobalVector<Scalar> tmp_numerator_base(m_mesh_data->getMeshTriangleData()->getN(), m_exec_conf);
    GlobalVector<Scalar> tmp_numerator_base(1, m_exec_conf); // The number of meshstructure?

    m_numerator_base.swap(tmp_numerator_base);
    TAG_ALLOCATION(m_numerator_base);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // cudaMemAdvise(m_sigma_dash.get(),
        //               sizeof(Scalar3) * m_sigma_dash.getNumElements(),
        //               cudaMemAdviseSetReadMostly,
        //               0);

        cudaMemAdvise(m_numerator_base.get(),
                      sizeof(Scalar) * m_numerator_base.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);
        }
#endif
    }

AreaConservationMeshForceCompute::~AreaConservationMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying AreaConservationMeshForceCompute" << endl;

    delete[] m_K;
    delete[] m_A0;
    m_K = NULL;
    m_A0 = NULL;

    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    Sets parameters for the potential of a particular angle type
    \param A0 desired surface area to maintain for the force computation
*/
void AreaConservationMeshForceCompute::setParams(unsigned int type, Scalar K, Scalar A0)
    {

    m_K[type] = K;
    m_A0[type] = A0;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "AreaConservation: specified K <= 0" << endl;

    if (A0 <= 0)
        m_exec_conf->msg->warning() << "AreaConservation: specified A0 <= 0" << endl;
    }

void AreaConservationMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    auto _params = area_conservation_params(params);
    setParams(typ, _params.k, _params.A0);
    }

pybind11::dict AreaConservationMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshTriangleData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshTriangleData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.area_conservation: Invalid mesh type specified" << endl;
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
    if (m_prof)
        m_prof->push("Area Conservation in Mesh");

    computeNumeratorBase(); // precompute base part of numerator in U(energy)

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    // ArrayHandle<typename MeshBond::members_t> h_bonds(m_mesh_data->getMeshBondData()->getMembersArray(),
    //                                                access_location::host,
    //                                                access_mode::read);
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(m_mesh_data->getMeshTriangleData()->getMembersArray(),
                                                   access_location::host,
                                                   access_mode::read);

    ArrayHandle<Scalar> h_numerator_base(m_numerator_base, access_location::host, access_mode::read);
    // ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    // assert(h_bonds.data);
    assert(h_triangles.data);
    assert(h_numerator_base.data);

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar area_conservation_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        area_conservation_virial[i] = Scalar(0.0);

    // for each of the triangles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    Scalar At = m_A0[0] / size; 
    for (unsigned int i = 0; i < size; i++)
{
        // lookup the tag of each of the particles participating in the triangle
        const typename MeshTriangle::members_t& triangle = h_triangles.data[i];
        assert(triangle.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[1] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[2] < m_pdata->getMaximumTag() + 1);

        // transform a, b, and c into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[triangle.tag[0]];
        unsigned int idx_b = h_rtag.data[triangle.tag[1]];
        unsigned int idx_c = h_rtag.data[triangle.tag[2]];

        //std::cout << i << ": " << idx_a << " " << idx_b << " " << idx_c << " " << idx_d << std::endl;

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x =  h_pos.data[idx_b].x - h_pos.data[idx_a].x;
        dab.y =  h_pos.data[idx_b].y - h_pos.data[idx_a].y;
        dab.z =  h_pos.data[idx_b].z - h_pos.data[idx_a].z;

        Scalar3 dac;
        dac.x =  h_pos.data[idx_c].x - h_pos.data[idx_a].x;
        dac.y =  h_pos.data[idx_c].y - h_pos.data[idx_a].y;
        dac.z =  h_pos.data[idx_c].z - h_pos.data[idx_a].z;

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

        da = box.minImage(da);
        db = box.minImage(db);
        dc = box.minImage(dc);

        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);

        Scalar3 nab, nac;
        nab = dab/rab;
        nac = dac/rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        if (s_baac < SMALL)
            s_baac = SMALL;
        
        Scalar3 dc_dra, dc_drb, dc_drc; // dcos_baac / dr_a 
        dc_dra = nac / rab * (-1.0) + nab /rac * (-1.0) - nab*nac / rab * (-1.0 * nab) - nab * nac / rac * (-1.0 * nac);
        dc_drb = nac / rab - rab * nac / rab * nab;
        dc_drc = nac / rac - nab * nac / rac * nac;

        Scalar3 ds_dra, ds_drb, ds_drc; // dsin_baac / dr_a 
        ds_dra = - 1.0 * c_baac / sqrt(1.0 - c_baac * c_baac) * dc_dra;
        ds_drb = - 1.0 * c_baac / sqrt(1.0 - c_baac * c_baac) * dc_drb;
        ds_drc = - 1.0 * c_baac / sqrt(1.0 - c_baac * c_baac) * dc_drc;

        // Scalar drab_dra, drab_drb, drab_drc, drac_dra, drac_drb, drac_drc;
        // d_rab_dra = -1.0;
        // d_rab_drb = 1.0;
        // d_rab_drc = 0.0;
        // d_rab_dra = -1.0;
        // d_rab_drb = 0.0;
        // d_rab_drc = 1.0;

        // Scalar dsrab_dra, dsrab_drb, dsrab_drc, dsrac_dra, dsrac_drb, dsrac_drc;
        // d_srab_dra = -1.0 * nab;
        // d_srab_drb = nab;
        // d_srab_drc = 0.0;
        // d_srac_dra = -1.0 * nac;
        // d_srac_drb = 0.0;
        // d_srac_drc = nac;

        Scalar numerator_base = h_numerator_base.data; //h_numerator_base.data[i]; //precomputed

        Scalar3 Fa, Fb, Fc;
        Fa = m_K[0] / (2 * At) * numerator_base * (- 1.0 * nab * rac * s_baac - nac * rab * s_baac + ds_dra * rab * rac);
        Fb = m_K[0] / (2 * At) * numerator_base * (nab * rac * s_baac + ds_drb * rab * rac);
        Fc = m_K[0] / (2 * At) * numerator_base * (nac * rab * s_baac + ds_drc * rab * rac);

        //std::cout << i << " " << idx_c << ": " << Fa.x << " " << Fa.y << " " << Fa.z << std::endl;

        if (compute_virial)
            {
            area_conservation_virial[0] = Scalar(1. / 2.) * da.x * Fa.x;// xx
            area_conservation_virial[1] = Scalar(1. / 2.) * da.y * Fa.x;// xy
            area_conservation_virial[2] = Scalar(1. / 2.) * da.z * Fa.x;// xz
            area_conservation_virial[3] = Scalar(1. / 2.) * da.y * Fa.y;// yy
            area_conservation_virial[4] = Scalar(1. / 2.) * da.z * Fa.y;// yz
            area_conservation_virial[5] = Scalar(1. / 2.) * da.z * Fa.z;// zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w = m_K[0]/(2.0*At)*(rab*rac*s_baac/2 - At);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += area_conservation_virial[j];
            }

        if (compute_virial)
            {
            area_conservation_virial[0] = Scalar(1. / 2.) * db.x * Fb.x;// xx
            area_conservation_virial[1] = Scalar(1. / 2.) * db.y * Fb.x;// xy
            area_conservation_virial[2] = Scalar(1. / 2.) * db.z * Fb.x;// xz
            area_conservation_virial[3] = Scalar(1. / 2.) * db.y * Fb.y;// yy
            area_conservation_virial[4] = Scalar(1. / 2.) * db.z * Fb.y;// yz
            area_conservation_virial[5] = Scalar(1. / 2.) * db.z * Fb.z;// zz
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x += Fb.x;
            h_force.data[idx_b].y += Fb.y;
            h_force.data[idx_b].z += Fb.z;
            h_force.data[idx_b].w = m_K[0]/(2.0*At)*(rab*rac*s_baac/2 - At);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += area_conservation_virial[j];
            }

        if (compute_virial)
            {
            area_conservation_virial[0] = Scalar(1. / 2.) * dc.x * Fc.x;// xx
            area_conservation_virial[1] = Scalar(1. / 2.) * dc.y * Fc.x;// xchy
            area_conservation_virial[2] = Scalar(1. / 2.) * dc.z * Fc.x;// xz
            area_conservation_virial[3] = Scalar(1. / 2.) * dc.y * Fc.y;// yy
            area_conservation_virial[4] = Scalar(1. / 2.) * dc.z * Fc.y;// yz
            area_conservation_virial[5] = Scalar(1. / 2.) * dc.z * Fc.z;// zz
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += Fc.x;
            h_force.data[idx_c].y += Fc.y;
            h_force.data[idx_c].z += Fc.z;
            h_force.data[idx_c].w = m_K[0]/(2.0*At)*(rab*rac*s_baac/2 - At);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += area_conservation_virial[j];
            }
        }

    if (m_prof)
        m_prof->pop();
    }

void AreaConservationMeshForceCompute::computeNumeratorBase()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // ArrayHandle<typename MeshBond::members_t> h_bonds(
    //     m_mesh_data->getMeshBondData()->getMembersArray(),
    //     access_location::host,
    //     access_mode::read);
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar> h_numerator_base(m_numerator_base, access_location::host, access_mode::overwrite);

    memset((void*)h_numerator_base.data, 0, sizeof(Scalar) * m_numerator_base.getNumElements());
    // memset((void*)h_sigma_dash.data, 0, sizeof(Scalar3) * m_sigma_dash.getNumElements());

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    Scalar At = m_A0[0] / size; 
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the triangle
        const typename MeshTriangle::members_t& triangle = h_triangles.data[i];
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

        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        rab = sqrt(rab);
        Scalar rac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        rac = sqrt(rac);

        Scalar3 nab, nac;
        nab = dab / rab;
        nac = dac / rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        if (s_baac < SMALL)
            s_baac = SMALL;

        Scalar numerator_base = rab * rac * s_baac / 2 - At;

        h_numerator_base.data += numerator_base; // h_numerator_base.data[i]
        }
    }

namespace detail
    {
void export_AreaConservationMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<AreaConservationMeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<AreaConservationMeshForceCompute>>(m, "AreaConservationMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &AreaConservationMeshForceCompute::setParamsPython)
        .def("getParams", &AreaConservationMeshForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
