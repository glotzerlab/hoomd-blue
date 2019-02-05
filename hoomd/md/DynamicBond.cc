// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ?

#include "DynamicBond.h"

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
                    Scalar r_cut,
                    nlist,
                    Scalar period,
                    bond_type,
                    int seed,
                    Scalar prob_create,
                    Scalar prob_destroy)
        : Updater(sysdef), m_group(group), m_r_cut(r_cut), m_nlist(nlist), m_period(period), m_bond_type(bond_type), m_seed(seed), m_prob_create(prob_create), m_prob_destroy(prob_destroy)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBond" << endl;


    }

DynamicBond::~DynamicBond()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBond" << endl;
    }

void DynamicBond::update(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size ==0)
        return;

    if (m_prof) m_prof->push("DynamicBond");

    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        }
    if (m_prof)
        m_prof->pop();
    }

void export_DynamicBond(py::module& m)
    {
    py::class_< DynamicBond, std::shared_ptr<DynamicBond> >(m, "DynamicBond", py::base<Updater>()).def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
    ;
    }
