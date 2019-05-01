// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ?

#include "DynamicBond.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"

#include "hoomd/Saru.h"
using namespace hoomd;
namespace py = pybind11;

/*! \file DynamicBond.cc
    \brief Contains code for the DynamicBond class
*/

using namespace std;

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param nlist Neighborlist to use
    \param seed Random number generator seed
    \param period Period at which to call the updater
*/
DynamicBond::DynamicBond(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        std::shared_ptr<NeighborList> nlist,
                        int seed,
                        int period)
        : Updater(sysdef), m_group(group), m_nlist(nlist), m_seed(seed), m_r_cut(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBond" << endl;
    }


/*! \param r_cut cut off distance for computing bonds
    \param bond_type type of bond to be formed or broken
    \param prob_form probability that a bond will form
    \param prob_break probability that a bond will break

    Sets parameters for the dynamic bond updater
*/
void DynamicBond::setParams(Scalar r_cut,
                            std::string bond_type,
                            Scalar prob_form,
                            Scalar prob_break)
    {
    if (m_r_cut < 0)
        {
        m_exec_conf->msg->error() << "r_cut cannot be less than 0.\n" << std::endl;
        }
    m_r_cut = r_cut;
    m_prob_form = prob_form;
    m_prob_break = prob_break;
    }


DynamicBond::~DynamicBond()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBond" << endl;
    }


void DynamicBond::update(unsigned int timestep)
    {
    assert(m_pdata);
    assert(m_nlist);

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // get box dimensions
    const BoxDim& box = m_pdata->getGlobalBox();

    // start the profile for this compute
    if (m_prof) m_prof->push("DynamicBond");

    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // Access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    // Access bond data
    m_bond_data = m_sysdef->getBondData();

    // Access the bond table for reading
    const GPUArray<typename BondData::members_t>& gpu_bond_list = this->m_bond_data->getGPUTable();
    const Index2D& gpu_table_indexer = this->m_bond_data->getGPUTableIndexer();

    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);
    ArrayHandle<typeval_t> h_typeval(m_bond_data->getTypeValArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int>  h_bond_tags(m_bond_data->getTags(), access_location::host, access_mode::read);

    max_n_bonds = m_bond_data->getGPUTableIndexer().getW() // width of the table

    assert(h_pos);

    Scalar r_cut_sq = m_r_cut*m_r_cut;

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // initialize the RNG
        detail::Saru saru(i, timestep, m_seed);

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter of particle i
        Scalar di = Scalar(0.0);
        di = h_diameter.data[i];

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter of j
            Scalar dj = Scalar(0.0);
            dj = h_diameter.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // auto curr_b_type = m_bond_data->getTypeByIndex(i);

            // create a bond between particles i and j
            if (rsq < r_cut_sq)
                {
                Scalar rnd1 = saru.s<Scalar>(0,1);
                if (rnd1 < m_prob_form)
                    {
                    m_bond_data->addBondedGroup(Bond(0, i, j));
                    }
                }

                for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++) // bond index for all bonds on particle i
                    {
                    group_storage<2> cur_bond = blist[blist_idx(idx, bond_idx)];

                    int bonded_idx = cur_bond.idx[0];
                    int bonded_type = cur_bond.idx[1];

                Scalar rnd2 = saru.s<Scalar>(0,1);
                if (rnd2 < m_prob_break)
                    {
                    const unsigned int size = (unsigned int)m_bond_data->getN();
                    for (unsigned int bond_idx = 0; bond_idx < size; bond_idx++)
                        {
                        // lookup the tag of each of the particles participating in the bond
                        const typename BondData::members_t& bond = h_bonds.data[bond_idx];
                        assert(bond.tag[0] < m_pdata->getMaximumTag()+1);
                        assert(bond.tag[1] < m_pdata->getMaximumTag()+1);

                        // transform a and b into indices into the particle data arrays
                        // (MEM TRANSFER: 4 integers)
                        unsigned int idx_a = h_rtag.data[bond.tag[0]];
                        unsigned int idx_b = h_rtag.data[bond.tag[1]];
                        if ((idx_a == i & idx_b == j) | (idx_a == j & idx_b == i)) {
                            m_bond_data->removeBondedGroup(h_bond_tags[bond_idx]);

                            }
                        }
                    }
                }

            }
        }

    if (m_prof)
        m_prof->pop();
    }


void export_DynamicBond(py::module& m)
    {
    py::class_< DynamicBond, std::shared_ptr<DynamicBond> >(m, "DynamicBond", py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, int, int>())
    .def("setParams", &DynamicBond::setParams);
    }
