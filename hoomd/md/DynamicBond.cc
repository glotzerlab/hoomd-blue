// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// TODO: debug by looking at dij - might be wrong??

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

Scalar capfraction(Scalar x)
    {
	Scalar frac;
	Scalar a0= 0.0925721;
    Scalar a1= -0.00699901;
    Scalar a2= 0.000378692;
    Scalar a3= -1.55671e-05;
    Scalar a4= 4.33718e-07;
    Scalar a5= -7.41086e-09;
    Scalar a6= 6.8603e-11;
    Scalar a7= -2.61042e-13;

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

Scalar feneEnergy(Scalar x, int nK)
    {
	Scalar UBi;
	UBi = nK * (-1.5 * log(1.0 - x * x));
	return UBi;
    }

Scalar feneForce(Scalar x) {
	Scalar FBi;
	FBi = (3.0 * x) / (1.0 - x * x);
	return FBi;
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
        : Updater(sysdef), m_group(group), m_nlist(nlist), m_seed(seed), m_r_cut(0.0), m_nK(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBond" << endl;
    int n_particles = m_pdata->getN();
    m_nloops.resize(n_particles);
    }


/*! \param r_cut cut off distance for computing bonds
    \param bond_type type of bond to be formed or broken
    \param delta_G sticker strength (kT)

    Sets parameters for the dynamic bond updater
*/
void DynamicBond::setParams(Scalar r_cut,
                            std::string bond_type,
                            Scalar delta_G,
                            int n_polymer,
                            int nK)
    {
    if (m_r_cut < 0)
        {
        m_exec_conf->msg->error() << "r_cut cannot be less than 0.\n" << std::endl;
        }
    m_r_cut = r_cut;
    m_delta_G = delta_G;
    std::fill(m_nloops.begin(), m_nloops.end(), 400);
    m_nK = nK;
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

    // Access the GPU bond table for reading
    const Index2D& gpu_table_indexer = this->m_bond_data->getGPUTableIndexer();
    ArrayHandle<BondData::members_t> h_gpu_bondlist(this->m_bond_data->getGPUTable(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int > h_gpu_n_bonds(this->m_bond_data->getNGroupsArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // Access the CPU bond table for reading
    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int>  h_bond_tags(m_bond_data->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

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
        assert(di > 0.0);

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
                // find number of bonds between i and j
                int n_bonds = h_gpu_n_bonds.data[i];
                int nbridges_ij = 0;
                for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
                    {
                    group_storage<2> cur_bond = h_gpu_bondlist.data[gpu_table_indexer(i, bond_idx)];
                    int bonded_tag = cur_bond.tag[0];
                    if (bonded_tag == h_tag.data[j])
                        {
                        nbridges_ij += 1;
                        }
                    }
                Scalar r = sqrt(rsq);
                Scalar surf_dist = r - (di+dj)/2;
                Scalar tstep = 0.05;
                Scalar omega = 4.68;  // natural thermal vibration frequency 1.2E0*3.9E-9

                // calculate probabilities
                Scalar chain_extension = surf_dist/m_nK;
                Scalar p0=tstep*omega*exp(-(m_delta_G+feneEnergy(chain_extension, m_nK)));
                Scalar q0=tstep*omega*exp(-(m_delta_G-feneEnergy(chain_extension, m_nK)+feneEnergy(chain_extension, m_nK)));

                Scalar p12 = p0*pow((1-p0),(m_nloops[i]*capfraction(surf_dist)-1.0))*m_nloops[i]*capfraction(surf_dist);
                Scalar p21 = p0*pow((1-p0),(m_nloops[j]*capfraction(surf_dist)-1.0))*m_nloops[j]*capfraction(surf_dist);
                Scalar q1 =  q0*pow((1-q0),(nbridges_ij-1.0))*nbridges_ij;

                // generate random numbers
                Scalar rnd1 = saru.s<Scalar>(0,1);
                Scalar rnd2 = saru.s<Scalar>(0,1);
                Scalar rnd3 = saru.s<Scalar>(0,1);
                Scalar rnd4 = saru.s<Scalar>(0,1);

                // check to see if a bond should be created between particles i and j
                if (rnd1 < p12)
                    {
                    m_bond_data->addBondedGroup(Bond(0, h_tag.data[i], h_tag.data[j]));
                    m_nloops[i] -= 1;
                    }

                if (rnd2 < p21)
                    {
                    m_bond_data->addBondedGroup(Bond(0, h_tag.data[i], h_tag.data[j]));
                    m_nloops[j] -= 1;
                    }

                // check to see if a bond should be broken between particles i and j
                if (rnd3 < q1)
                    {
                    // for each of the bonds in the *system*
                    const unsigned int size = (unsigned int)m_bond_data->getN();
                    // m_exec_conf->msg->notice(2) << "bonds in the system " << size << endl;

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
                            // m_exec_conf->msg->notice(2) << "Removing bond with tag: " << bond_number << endl;
                            // m_exec_conf->msg->notice(2) << "between particles with tags: " << i << "," << j << endl;
                            m_bond_data->removeBondedGroup(bond_number);
                            break;
                            }
                        }
                    if (rnd4 < 0.5)
                        {
                        m_nloops[i] += 1;
                        }
                    else if (rnd4 >=0.5)
                        {
                        m_nloops[j] +=1;
                        }
                    }
                }
            // TODO: sanity check number of loops and number of bridges
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
