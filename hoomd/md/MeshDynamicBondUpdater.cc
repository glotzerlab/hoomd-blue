// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshDynamicBondUpdater.cc
    \brief Defines the MeshDynamicBondUpdater class
*/

#include "MeshDynamicBondUpdater.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

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
                                               std::shared_ptr<Trigger> trigger,
                                               std::shared_ptr<Integrator> integrator,
                                               std::shared_ptr<MeshDefinition> mesh,
                                               Scalar T)
    : Updater(sysdef, trigger), m_integrator(integrator), m_mesh(mesh), m_inv_T(1.0 / T)
    {
    GlobalVector<bool> tmp_allowSwitch(m_pdata->getNTypes(), m_exec_conf);

    m_allowSwitch.swap(tmp_allowSwitch);
    TAG_ALLOCATION(m_allowSwitch);

    ArrayHandle<bool> h_allowSwitch(m_allowSwitch, access_location::host, access_mode::overwrite);

    for (unsigned int i = 0; i < m_allowSwitch.size(); i++)
        h_allowSwitch.data[i] = true;

    assert(m_pdata);
    assert(m_integrator);
    assert(m_mesh);
    m_exec_conf->msg->notice(5) << "Constructing MeshDynamicBondUpdater" << endl;
    }

MeshDynamicBondUpdater::~MeshDynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying MeshDynamicBondUpdater" << endl;
    }

void MeshDynamicBondUpdater::setAllowSwitch(const std::string& type_name, pybind11::bool_ v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    bool allowSwitch = pybind11::cast<bool>(v);

    ArrayHandle<bool> h_allowSwitch(m_allowSwitch, access_location::host, access_mode::readwrite);
    h_allowSwitch.data[typ] = allowSwitch;
    }

pybind11::bool_ MeshDynamicBondUpdater::getAllowSwitch(const std::string& type_name)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<bool> h_allowSwitch(m_allowSwitch, access_location::host, access_mode::read);

    bool allowSwitch = h_allowSwitch.data[typ];
    return pybind11::bool_(allowSwitch);
    }

/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void MeshDynamicBondUpdater::update(uint64_t timestep)
    {
    std::vector<std::shared_ptr<ForceCompute>> forces = m_integrator->getForces();

    uint16_t seed = m_sysdef->getSeed();

    for (auto& force : forces)
        {
        force->precomputeParameter();
        }

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

    ArrayHandle<bool> h_allowSwitch(m_allowSwitch, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh->getMeshBondData()->getN();

    bool changeDetected = false;

    std::vector<unsigned int> changed;

    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];

        unsigned int type_a = __scalar_as_int(h_pos.data[idx_a].w);
        unsigned int type_b = __scalar_as_int(h_pos.data[idx_b].w);

        bool allow_a = h_allowSwitch.data[type_a];
        bool allow_b = h_allowSwitch.data[type_b];

        if (!allow_a && !allow_b)
            continue;

        bool already_used = false;
        for (unsigned int j = 0; j < changed.size(); j++)
            {
            if (tag_a == changed[j] || tag_b == changed[j])
                {
                already_used = true;
                break;
                }
            }

        if (already_used)
            continue;

        unsigned int tr_idx1 = bond.tag[2];
        unsigned int tr_idx2 = bond.tag[3];

        if (tr_idx1 == tr_idx2)
            continue;

        const typename MeshTriangle::members_t& triangle1 = h_triangles.data[tr_idx1];
        const typename MeshTriangle::members_t& triangle2 = h_triangles.data[tr_idx2];

        unsigned int iterator = 0;

        bool a_before_b = true;

        while (idx_b == h_rtag.data[triangle1.tag[iterator]])
            iterator++;

        iterator = (iterator + 1) % 3;

        if (idx_a == h_rtag.data[triangle1.tag[iterator]])
            a_before_b = false;

        unsigned int tag_c = triangle1.tag[0];
        unsigned int idx_c = h_rtag.data[tag_c];

        iterator = 0;
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

        for (unsigned int j = 0; j < changed.size(); j++)
            {
            if (tag_c == changed[j] || tag_d == changed[j])
                {
                already_used = true;
                break;
                }
            }

        if (already_used)
            continue;

        unsigned int type_id = h_typeval.data[i].type;

        Scalar energyDifference = 0;

        unsigned int idx_cc = idx_d;
        unsigned int idx_dd = idx_c;

        if (a_before_b)
            {
	    idx_cc = idx_c;
	    idx_dd = idx_d;
	    }

	bool have_to_check_surrounding = false;
        for (auto& force : forces)
	    {
            energyDifference += force->energyDiff(idx_a, idx_b, idx_cc, idx_dd, type_id);

	    if(force->checkSurrounding())
		    have_to_check_surrounding = true;
	    }

	if(have_to_check_surrounding)
           {
           unsigned int idx_1, idx_2, idx_3, idx_4, idx_5;
	   for( int bo_i = 3; bo_i < 6; bo_i++)
	      {
	      unsigned int new_bo = triangle1.tag[bo_i];
	      if( new_bo == i) continue;
	      const typename MeshBond::members_t& bond1 = h_bonds.data[new_bo];

	      idx_1 = h_rtag.data[bond1.tag[0]];
	      idx_2 = h_rtag.data[bond1.tag[1]];

	      for( int tr_i = 2; tr_i < 4; tr_i++)
	      	{
	   	if(bond1.tag[tr_i] == tr_idx1) continue;

	      	const typename MeshTriangle::members_t& triangle1 = h_triangles.data[bond1.tag[tr_i]];
	      	for( int idx_i = 0; idx_i < 3; idx_i++)
	   	   {
	   	   if( triangle1.tag[idx_i] != bond1.tag[0] &&  triangle1.tag[idx_i] != bond1.tag[1] )
		      {
	   	      idx_3 = h_rtag.data[triangle1.tag[idx_i]];
		      break;
		      }
	   	   }
		break;
	   	}

	      idx_4 = idx_a;
	      if(idx_1 == idx_4 || idx_2 == idx_4) idx_4 = idx_b;
	      idx_5 = idx_d;

              for (auto& force : forces)
	      	energyDifference += force->energyDiffSurrounding(idx_1, idx_2, idx_3, idx_4, idx_5, type_id);

	      }

	   for( int bo_i = 3; bo_i < 6; bo_i++)
	      {
	      unsigned int new_bo = triangle2.tag[bo_i];
	      if( new_bo == i) continue;
	      const typename MeshBond::members_t& bond1 = h_bonds.data[new_bo];

	      idx_1 = h_rtag.data[bond1.tag[0]];
	      idx_2 = h_rtag.data[bond1.tag[1]];

	      for( int tr_i = 2; tr_i < 4; tr_i++)
	      	{
	   	if(bond1.tag[tr_i] == tr_idx2) continue;

	      	const typename MeshTriangle::members_t& triangle2 = h_triangles.data[bond1.tag[tr_i]];
	      	for( int idx_i = 0; idx_i < 3; idx_i++)
	   	   {
	   	   if( triangle2.tag[idx_i] != bond1.tag[0] &&  triangle2.tag[idx_i] != bond1.tag[1] )
		      {
	   	      idx_3 = h_rtag.data[triangle2.tag[idx_i]];
		      break;
		      }
	   	   }
		break;
                }

	      idx_4 = idx_a;
	      if(idx_1 == idx_4 || idx_2 == idx_4) idx_4 = idx_b;
	      idx_5 = idx_c;

              for (auto& force : forces)
	      	energyDifference += force->energyDiffSurrounding(idx_1, idx_2, idx_3, idx_4, idx_5, type_id);

	      }
	   }

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::MeshDynamicBondUpdater, timestep, seed),
                            hoomd::Counter(i));

        // compute the random force
        UniformDistribution<Scalar> uniform(0, Scalar(1));

        if (exp(-m_inv_T * energyDifference) > uniform(rng))
            {
            changeDetected = true;

            changed.push_back(tag_a);
            changed.push_back(tag_b);
            changed.push_back(tag_c);
            changed.push_back(tag_d);

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
                    typename MeshBond::members_t& bond_s = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_s.tag[0] == tag_a || bond_s.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[4] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[4] = k;
                        }

                    if (bond_s.tag[2] == tr_idx1 || bond_s.tag[2] == tr_idx2)
                        bond_s.tag[2] = tr_idx;
                    else
                        bond_s.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_s;
                    }
                }

            triangle1_n.tag[5] = i;
            triangle2_n.tag[5] = i;

            h_triangles.data[tr_idx1] = triangle1_n;
            h_triangles.data[tr_idx2] = triangle2_n;

            if (a_before_b)
                {
                for (auto& force : forces)
                    {
                    force->postcompute(idx_a, idx_b, idx_c, idx_d);
                    }
                }
            else
                {
                for (auto& force : forces)
                    {
                    force->postcompute(idx_a, idx_b, idx_d, idx_c);
                    }
                }
            }
        }

    if (changeDetected)
        {
        m_mesh->getMeshBondData()->meshChanged();
        m_mesh->getMeshTriangleData()->meshChanged();
        }
    }

namespace detail
    {
void export_MeshDynamicBondUpdater(pybind11::module& m)
    {
    pybind11::class_<MeshDynamicBondUpdater, Updater, std::shared_ptr<MeshDynamicBondUpdater>>(
        m,
        "MeshDynamicBondUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<Integrator>,
                            std::shared_ptr<MeshDefinition>,
                            Scalar>())
        .def("setAllowSwitch", &MeshDynamicBondUpdater::setAllowSwitch)
        .def("getAllowSwitch", &MeshDynamicBondUpdater::getAllowSwitch)
        .def_property("kT", &MeshDynamicBondUpdater::getT, &MeshDynamicBondUpdater::setT);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
