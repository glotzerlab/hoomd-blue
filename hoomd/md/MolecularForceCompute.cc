// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "MolecularForceCompute.h"

#include <string.h>
#include <map>

namespace py = pybind11;

/*! \file MolecularForceCompute.cc
    \brief Contains code for the MolecularForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
MolecularForceCompute::MolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceConstraint(sysdef), m_molecule_tag(m_exec_conf), m_n_molecules_global(0),
      m_molecule_list(m_exec_conf), m_molecule_length(m_exec_conf), m_molecule_order(m_exec_conf),
      m_molecule_idx(m_exec_conf), m_dirty(true)
    {
    // connect to the ParticleData to recieve notifications when particles change order in memory
    m_pdata->getParticleSortSignal().connect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);
    }

//! Destructor
MolecularForceCompute::~MolecularForceCompute()
    {
    m_pdata->getParticleSortSignal().disconnect<MolecularForceCompute, &MolecularForceCompute::setDirty>(this);
    }

void MolecularForceCompute::initMolecules()
    {
    // return early if no molecules are defined
    if (!m_n_molecules_global) return;

    if (m_prof) m_prof->push("init molecules");

    m_exec_conf->msg->notice(7) << "MolecularForceCompute initializing molecule table" << std::endl;

    // construct local molecule table
    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    std::map<unsigned int,unsigned int> local_molecule_tags;

    unsigned int n_local_molecules = 0;

    std::vector<unsigned int> local_molecule_idx(nptl_local, NO_MOLECULE);

    // resize molecule lookup to size of local particle data
    m_molecule_order.resize(m_pdata->getMaxN());

    // identify local molecules and assign local indices to global molecule tags
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        unsigned int tag = h_tag.data[i];
        assert(tag < m_molecule_tag.getNumElements());

        unsigned int mol_tag = h_molecule_tag.data[tag];
        if (mol_tag == NO_MOLECULE) continue;

        std::map<unsigned int,unsigned int>::iterator it = local_molecule_tags.find(mol_tag);
        if (it == local_molecule_tags.end())
            {
            // insert element
            it = local_molecule_tags.insert(std::make_pair(mol_tag,n_local_molecules++)).first;
            }

        local_molecule_idx[i] = it->second;
        }

    m_molecule_length.resize(n_local_molecules);

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::overwrite);

    // reset lengths
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // count molecule lengths
    for (unsigned int i = 0; i < nptl_local; ++i)
        {
        unsigned int molecule_i = local_molecule_idx[i];
        if (molecule_i != NO_MOLECULE)
            {
            h_molecule_length.data[molecule_i]++;
            }
        }

    // find maximum length
    unsigned nmax = 0;
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        if (h_molecule_length.data[imol] > nmax)
            {
            nmax = h_molecule_length.data[imol];
            }
        }

    // set up indexer
    m_molecule_indexer = Index2D(n_local_molecules, nmax);

    // resize molecule list
    m_molecule_list.resize(m_molecule_indexer.getNumElements());

    // reset lengths again
    for (unsigned int imol = 0; imol < n_local_molecules; ++imol)
        {
        h_molecule_length.data[imol] = 0;
        }

    // sort local molecules by ptl tag
    std::vector< std::set<unsigned int> > local_molecules_sorted_by_tag(n_local_molecules);

    for (unsigned int iptl = 0; iptl < nptl_local; ++iptl)
        {
        unsigned int i_mol = local_molecule_idx[iptl];

        if (i_mol != NO_MOLECULE)
            {
            local_molecules_sorted_by_tag[i_mol].insert(h_tag.data[iptl]);
            }
        }

    // reset molecule order
    ArrayHandle<unsigned int> h_molecule_order(m_molecule_order, access_location::host, access_mode::overwrite);
    memset(h_molecule_order.data, 0, sizeof(unsigned int)*(m_pdata->getN() + m_pdata->getNGhosts()));

    // resize reverse-lookup
    m_molecule_idx.resize(nptl_local);

    // fill molecule list
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_molecule_idx(m_molecule_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // reset reverse lookup
    memset(h_molecule_idx.data, 0, sizeof(unsigned int)*nptl_local);

    unsigned int i_mol = 0;
    for (std::vector< std::set<unsigned int> >::iterator it_mol = local_molecules_sorted_by_tag.begin();
        it_mol != local_molecules_sorted_by_tag.end(); ++it_mol)
        {
        for (std::set<unsigned int>::iterator it_tag = it_mol->begin(); it_tag != it_mol->end(); ++it_tag)
            {
            unsigned int n = h_molecule_length.data[i_mol]++;
            unsigned int ptl_idx = h_rtag.data[*it_tag];
            assert(ptl_idx < m_pdata->getN() + m_pdata->getNGhosts());
            h_molecule_list.data[m_molecule_indexer(i_mol, n)] = ptl_idx;
            h_molecule_idx.data[ptl_idx] = i_mol;
            h_molecule_order.data[ptl_idx] = n;
            }
        i_mol ++;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_MolecularForceCompute(py::module& m)
    {
    py::class_< MolecularForceCompute, std::shared_ptr<MolecularForceCompute> >(m, "MolecularForceCompute", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
