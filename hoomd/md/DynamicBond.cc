// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ?

#include "DynamicBond.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Saru.h"

namespace py = pybind11;

using namespace std;

/*! \file DynamicBond.cc
    \brief Contains code for the DynamicBond class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
*/

DynamicBond::DynamicBond(std::shared_ptr<SystemDefinition> sysdef,
        std::shared_ptr<ParticleGroup> group,
        std::shared_ptr<NeighborList> nlist,
        int seed,
        int period)
        : Updater(sysdef),
        m_group(group),
        m_nlist(nlist)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBond" << endl;

    }
    // assert(m_pdata);
    // assert(m_nlist);

// void DynamicBond::set_params(Scalar r_cut,
//                         std::string bond_type,
//                         Scalar prob_create,
//                         Scalar prob_destroy)
//     {
//     unsigned int b_type = m_bond_data->getTypeByName(bond_type);
//     }

DynamicBond::~DynamicBond()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBond" << endl;
    }

void DynamicBond::update(unsigned int timestep)
    {

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("DynamicBond");

    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // Access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    // // Access bond data
    m_bond_data = m_sysdef->getBondData();

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

    //     // sanity check
    //     assert(typei < m_pdata->getNTypes());
    //
    //     // access diameter (if needed)
    //     Scalar di = Scalar(0.0);
    //     if (evaluator::needsDiameter())
    //         di = h_diameter.data[i];
    //
    //     // loop over all of the neighbors of this particle
    //     const unsigned int myHead = h_head_list.data[i];
    //     const unsigned int size = (unsigned int)h_n_neigh.data[i];
    //     for (unsigned int k = 0; k < size; k++)
    //         {
    //         // access the index of this neighbor (MEM TRANSFER: 1 scalar)
    //         unsigned int j = h_nlist.data[myHead + k];
    //         assert(j < m_pdata->getN() + m_pdata->getNGhosts());
    //
    //         // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
    //         Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
    //         Scalar3 dx = pi - pj;
    //
    //         // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
    //         unsigned int typej = __scalar_as_int(h_pos.data[j].w);
    //         assert(typej < m_pdata->getNTypes());
    //
    //         // access diameter and charge (if needed)
    //         Scalar dj = Scalar(0.0);
    //         if (evaluator::needsDiameter())
    //             dj = h_diameter.data[j];
    //
    //         // apply periodic boundary conditions
    //         dx = box.minImage(dx);
    //
    //         // calculate r_ij squared (FLOPS: 5)
    //         Scalar rsq = dot(dx, dx);
    //
    //         get parameters for this type pair
    //         unsigned int typpair_idx = m_typpair_idx(typei, typej);
    //         param_type param = h_params.data[typpair_idx];
    //         Scalar rcutsq = h_rcutsq.data[typpair_idx];
    //         Scalar ronsq = Scalar(0.0);
    //         if (m_shift_mode == xplor)
    //             ronsq = h_ronsq.data[typpair_idx];
    //         }
        }


    if (m_prof)
        m_prof->pop();
    }


void export_DynamicBond(py::module& m)
    {
    py::class_< DynamicBond, std::shared_ptr<DynamicBond> >(m, "DynamicBond", py::base<Updater>()).def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<NeighborList>, int, int>());
    }
