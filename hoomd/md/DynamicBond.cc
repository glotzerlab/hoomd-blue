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

double capfraction(double x)
    {
	double frac;
	double a0= 0.0925721;
    double a1= -0.00699901;
    double a2= 0.000378692;
    double a3= -1.55671e-05;
    double a4= 4.33718e-07;
    double a5= -7.41086e-09;
    double a6= 6.8603e-11;
    double a7= -2.61042e-13;

    frac=a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x+a5*x*x*x*x*x+a6*x*x*x*x*x*x+a7*x*x*x*x*x*x*x;

    if (frac<=0.0)
        {
        frac = 0.0;
        }
	else if (frac>=1.0)
        {
         frac = 1.0;
        }
    return frac;
    }

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
    const Index2D& gpu_table_index = this->m_bond_data->getGPUTableIndexer();

    ArrayHandle<BondData::members_t> h_gpu_bondlist(gpu_bond_list, access_location::host, access_mode::read);
    ArrayHandle<unsigned int > h_gpu_n_bonds(this->m_bond_data->getNGroupsArray(), access_location::host, access_mode::read);

    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int>  h_bond_tags(m_bond_data->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

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
            // access the index of neighbor particle j (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // access the type of particle j (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter of particle j
            Scalar dj = Scalar(0.0);
            dj = h_diameter.data[j];

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            if (rsq < r_cut_sq)
                {
                // check to see if a bond should be created between particles i and j
                Scalar rnd1 = saru.s<Scalar>(0,1);
                if (rnd1 < m_prob_form)
                    {
                    m_bond_data->addBondedGroup(Bond(0, i, j));
                    }

                // check to see if a bond should be broken between particles i and j
                Scalar rnd2 = saru.s<Scalar>(0,1);
                if (rnd2 < m_prob_break)
                    {

                    // for each of the bonds in the system
                    const unsigned int size = (unsigned int)m_bond_data->getN();
                    m_exec_conf->msg->notice(2) << "bonds in the system " << size << endl;
                    for (unsigned int bond_number = 0; bond_number < size; bond_number++)
                        {
                        // lookup the tag of each of the particles participating in the bond
                        const BondData::members_t bond = m_bond_data->getMembersByIndex(bond_number);
                        assert(bond.tag[0] < m_pdata->getN());
                        assert(bond.tag[1] < m_pdata->getN());

                        // transform a and b into indices into the particle data arrays
                        // (MEM TRANSFER: 4 integers)
                        unsigned int idx_a = h_rtag.data[bond.tag[0]];
                        unsigned int idx_b = h_rtag.data[bond.tag[1]];
                        assert(idx_a <= m_pdata->getMaximumTag());
                        assert(idx_b <= m_pdata->getMaximumTag());

                        if ((bond.tag[0] == i && bond.tag[1] == j) || (bond.tag[0] == j & bond.tag[1] == i))
                            {
                            m_exec_conf->msg->notice(2) << "Removing bond with tag: " << bond_number << endl;
                            m_exec_conf->msg->notice(2) << "between particles with tags: " << i << "," << j << endl;
                            m_bond_data->removeBondedGroup(bond_number);
                            break;
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
