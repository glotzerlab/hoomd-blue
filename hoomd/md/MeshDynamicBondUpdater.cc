// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file MeshDynamicBondUpdater.cc
    \brief Defines the MeshDynamicBondUpdater class
*/

#include "MeshDynamicBondUpdater.h"

#include <iostream>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System definition
 *  \param rotational_diffusion The diffusion across time
 *  \param group the particles to diffusion rotation on
 */

MeshDynamicBondUpdater::MeshDynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                               std::shared_ptr<Integrator> integrator,
                                               std::shared_ptr<MeshDefinition> mesh)
    : Updater(sysdef), m_integrator(integrator), m_mesh(mesh)
    {
    assert(m_pdata);
    assert(m_integrator);
    assert(m_mesh);
    m_exec_conf->msg->notice(5) << "Constructing MeshDynamicBondUpdater" << endl;
    }

MeshDynamicBondUpdater::~MeshDynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying MeshDynamicBondUpdater" << endl;
    }

/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void MeshDynamicBondUpdater::update(uint64_t timestep)
    {
    ArrayHandle<typename MeshBond::members_t> h_bonds(m_mesh->getMeshBondData()->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::readwrite);
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::readwrite);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typeval_t> h_typeval(m_mesh->getMeshBondData()->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    std::vector<std::shared_ptr<ForceCompute>> forces = m_integrator->getForces();

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh->getMeshBondData()->getN();

    m_mesh->MeshDataChange();

    unsigned int zahl = 0;
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int tag_b = bond.tag[1];
        unsigned int idx_b = h_rtag.data[tag_b];

        unsigned int tr_idx1 = bond.tag[2];
        unsigned int tr_idx2 = bond.tag[3];

        if (tr_idx1 == tr_idx2)
            continue;

        const typename MeshTriangle::members_t& triangle1 = h_triangles.data[tr_idx1];
        const typename MeshTriangle::members_t& triangle2 = h_triangles.data[tr_idx2];

        unsigned int tag_c = triangle1.tag[0];
        unsigned int idx_c = h_rtag.data[tag_c];

        unsigned int iterator = 0;

        while (idx_a == idx_c || idx_b == idx_c)
            {
            iterator++;
            tag_c = triangle1.tag[iterator];
            idx_c = h_rtag.data[tag_c];
            }

        unsigned int tag_d = triangle2.tag[0];
        unsigned int idx_d = h_rtag.data[tag_d];

        iterator = 0;
        while (idx_a == idx_d || idx_b == idx_d)
            {
            iterator++;
            tag_d = triangle2.tag[iterator];
            idx_d = h_rtag.data[tag_d];
            }

        unsigned int type_id = h_typeval.data[i].type;

        Scalar energyDifference = 0;

        for (auto& force : forces)
            {
            energyDifference += force->energyDiff(idx_a, idx_b, idx_c, idx_d, type_id);
            }

        if (energyDifference < 0)
            {
            // std::cout << "Hier triangle1 " << triangle1.tag[0] << " " << triangle1.tag[1] << " "
            // << triangle1.tag[2] << " " << triangle1.tag[3] << " " << triangle1.tag[4] << " " <<
            // triangle1.tag[5] << " " << std::endl; std::cout << i << std::endl;
            // std::cout << "tags " << tag_a  << " " << tag_b << " " << tag_c << " " << tag_d <<
            // std::endl; std::cout << "bond " << h_bonds.data[i].tag[0] << " " <<
            // h_bonds.data[i].tag[1] << " " << h_bonds.data[i].tag[2] << " " <<
            // h_bonds.data[i].tag[3] << std::endl; std::cout << "triangle1 " << tr_idx1 << " " <<
            // h_triangles.data[tr_idx1].tag[0] << " " << h_triangles.data[tr_idx1].tag[1] << " " <<
            // h_triangles.data[tr_idx1].tag[2] << " " << h_triangles.data[tr_idx1].tag[3] << " " <<
            // h_triangles.data[tr_idx1].tag[4] << " " << h_triangles.data[tr_idx1].tag[5] << " " <<
            // std::endl; int b1t1 = h_triangles.data[tr_idx1].tag[3]; std::cout << "bond_t1_1 " <<
            // b1t1 << " " << h_bonds.data[b1t1].tag[0] << " " << h_bonds.data[b1t1].tag[1] << " "
            // << h_bonds.data[b1t1].tag[2] << " " << h_bonds.data[b1t1].tag[3] << std::endl; int
            // b2t1 = h_triangles.data[tr_idx1].tag[4]; std::cout << "bond_t1_2 " << b2t1 << " " <<
            // h_bonds.data[b2t1].tag[0] << " " << h_bonds.data[b2t1].tag[1] << " " <<
            // h_bonds.data[b2t1].tag[2] << " " << h_bonds.data[b2t1].tag[3] << std::endl; int b3t1
            // = h_triangles.data[tr_idx1].tag[5]; std::cout << "bond_t1_3 " << b3t1 << " " <<
            // h_bonds.data[b3t1].tag[0] << " " << h_bonds.data[b3t1].tag[1] << " " <<
            // h_bonds.data[b3t1].tag[2] << " " << h_bonds.data[b3t1].tag[3] << std::endl; std::cout
            // << "triangle2 " << tr_idx2 << " " << h_triangles.data[tr_idx2].tag[0] << " " <<
            // h_triangles.data[tr_idx2].tag[1] << " " << h_triangles.data[tr_idx2].tag[2] << " " <<
            // h_triangles.data[tr_idx2].tag[3] << " " << h_triangles.data[tr_idx2].tag[4] << " " <<
            // h_triangles.data[tr_idx2].tag[5] << " " << std::endl; int b1t2 =
            // h_triangles.data[tr_idx2].tag[3]; std::cout << "bond_t2_1 " << b1t2 << " " <<
            // h_bonds.data[b1t2].tag[0] << " " << h_bonds.data[b1t2].tag[1] << " " <<
            // h_bonds.data[b1t2].tag[2] << " " << h_bonds.data[b1t2].tag[3] << std::endl; int b2t2
            // = h_triangles.data[tr_idx2].tag[4]; std::cout << "bond_t2_2 " << b2t2 << " " <<
            // h_bonds.data[b2t2].tag[0] << " " << h_bonds.data[b2t2].tag[1] << " " <<
            // h_bonds.data[b2t2].tag[2] << " " << h_bonds.data[b2t2].tag[3] << std::endl; int b3t2
            // = h_triangles.data[tr_idx2].tag[5]; std::cout << "bond_t2_3 " << b3t2 << " " <<
            // h_bonds.data[b3t2].tag[0] << " " << h_bonds.data[b3t2].tag[1] << " " <<
            // h_bonds.data[b3t2].tag[2] << " " << h_bonds.data[b3t2].tag[3] << std::endl;
            zahl++;
            typename MeshBond::members_t bond_n;
            typename MeshTriangle::members_t triangle1_n;
            typename MeshTriangle::members_t triangle2_n;

            bond_n.tag[0] = tag_c;
            bond_n.tag[1] = tag_d;
            bond_n.tag[2] = tr_idx1;
            bond_n.tag[3] = tr_idx2;

            h_bonds.data[i] = bond_n;

            bool needs_flipping = true;

            if (iterator < 2)
                {
                if (triangle2.tag[iterator + 1] == tag_a)
                    needs_flipping = false;
                }
            else
                {
                if (triangle2.tag[0] == tag_a)
                    needs_flipping = false;
                }

            triangle1_n.tag[0] = tag_a;
            triangle2_n.tag[0] = tag_b;

            if (needs_flipping)
                {
                triangle1_n.tag[2] = tag_c;
                triangle1_n.tag[1] = tag_d;
                triangle2_n.tag[2] = tag_d;
                triangle2_n.tag[1] = tag_c;
                }
            else
                {
                triangle1_n.tag[1] = tag_c;
                triangle1_n.tag[2] = tag_d;
                triangle2_n.tag[1] = tag_d;
                triangle2_n.tag[2] = tag_c;
                }

            for (int j = 3; j < 6; j++)
                {
                int k = triangle1.tag[j];
                if (k != i)
                    {
                    typename MeshBond::members_t& bond_s = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_s.tag[0] == tag_a || bond_s.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[3] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[3] = k;
                        }

                    if (bond_s.tag[2] == tr_idx1 || bond_s.tag[2] == tr_idx2)
                        bond_s.tag[2] = tr_idx;
                    else
                        bond_s.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_s;
                    }
                k = triangle2.tag[j];
                if (k != i)
                    {
                    bond_n = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_n.tag[0] == tag_a || bond_n.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[4] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[4] = k;
                        }

                    if (bond_n.tag[2] == tr_idx1 || bond_n.tag[2] == tr_idx2)
                        bond_n.tag[2] = tr_idx;
                    else
                        bond_n.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_n;
                    }
                }

            triangle1_n.tag[5] = i;
            triangle2_n.tag[5] = i;

            h_triangles.data[tr_idx1] = triangle1_n;
            h_triangles.data[tr_idx2] = triangle2_n;

            // std::cout << std::endl;
            // std::cout << "triangle1 " << tr_idx1 << " " << h_triangles.data[tr_idx1].tag[0] << "
            // " << h_triangles.data[tr_idx1].tag[1] << " " << h_triangles.data[tr_idx1].tag[2] << "
            // " << h_triangles.data[tr_idx1].tag[3] << " " << h_triangles.data[tr_idx1].tag[4] << "
            // " << h_triangles.data[tr_idx1].tag[5] << " " << std::endl; b1t1 =
            // h_triangles.data[tr_idx1].tag[3]; std::cout << "bond_t1_1 " << b1t1 << " " <<
            // h_bonds.data[b1t1].tag[0] << " " << h_bonds.data[b1t1].tag[1] << " " <<
            // h_bonds.data[b1t1].tag[2] << " " << h_bonds.data[b1t1].tag[3] << std::endl; b2t1 =
            // h_triangles.data[tr_idx1].tag[4]; std::cout << "bond_t1_2 " << b2t1 << " " <<
            // h_bonds.data[b2t1].tag[0] << " " << h_bonds.data[b2t1].tag[1] << " " <<
            // h_bonds.data[b2t1].tag[2] << " " << h_bonds.data[b2t1].tag[3] << std::endl; b3t1 =
            // h_triangles.data[tr_idx1].tag[5]; std::cout << "bond_t1_3 " << b3t1 << " " <<
            // h_bonds.data[b3t1].tag[0] << " " << h_bonds.data[b3t1].tag[1] << " " <<
            // h_bonds.data[b3t1].tag[2] << " " << h_bonds.data[b3t1].tag[3] << std::endl; std::cout
            // << "triangle2 " << tr_idx2 << " " << h_triangles.data[tr_idx2].tag[0] << " " <<
            // h_triangles.data[tr_idx2].tag[1] << " " << h_triangles.data[tr_idx2].tag[2] << " " <<
            // h_triangles.data[tr_idx2].tag[3] << " " << h_triangles.data[tr_idx2].tag[4] << " " <<
            // h_triangles.data[tr_idx2].tag[5] << " " << std::endl; b1t2 =
            // h_triangles.data[tr_idx2].tag[3]; std::cout << "bond_t2_1 " << b1t2 << " " <<
            // h_bonds.data[b1t2].tag[0] << " " << h_bonds.data[b1t2].tag[1] << " " <<
            // h_bonds.data[b1t2].tag[2] << " " << h_bonds.data[b1t2].tag[3] << std::endl; b2t2 =
            // h_triangles.data[tr_idx2].tag[4]; std::cout << "bond_t2_2 " << b2t2 << " " <<
            // h_bonds.data[b2t2].tag[0] << " " << h_bonds.data[b2t2].tag[1] << " " <<
            // h_bonds.data[b2t2].tag[2] << " " << h_bonds.data[b2t2].tag[3] << std::endl; b3t2 =
            // h_triangles.data[tr_idx2].tag[5]; std::cout << "bond_t2_3 " << b3t2 << " " <<
            // h_bonds.data[b3t2].tag[0] << " " << h_bonds.data[b3t2].tag[1] << " " <<
            // h_bonds.data[b3t2].tag[2] << " " << h_bonds.data[b3t2].tag[3] << std::endl; std::cout
            // << energyDifference << " " << h_bonds.data[i].tag[0] << " " << h_bonds.data[i].tag[1]
            // << " " << h_bonds.data[i].tag[2] << " " << h_bonds.data[i].tag[3] << std::endl;
            // exit(0);
            }
        }
    std::cout << float(zahl) / size << std::endl;
    }

namespace detail
    {
void export_MeshDynamicBondUpdater(pybind11::module& m)
    {
    pybind11::class_<MeshDynamicBondUpdater, Updater, std::shared_ptr<MeshDynamicBondUpdater>>(
        m,
        "MeshDynamicBondUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Integrator>,
                            std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
